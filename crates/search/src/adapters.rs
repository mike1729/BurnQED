//! Bridges between search traits and real crate types (lean-repl, policy).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lean_repl::{LeanError, LeanPool, ProofHandleOwned, ProofState, TacticResult};
use policy::{GeneratedTactic, InferenceHandle};

use ebm::inference::EBMValueFn;

use crate::encode_batcher::BatchEncoder;
use crate::engine::{PolicyProvider, ProofEnvironment, SearchError, TacticRunner, ValueScorer};

// ---------------------------------------------------------------------------
// NullPolicyProvider — returns no candidates (probe-only search)
// ---------------------------------------------------------------------------

/// A policy provider that always returns empty candidates.
///
/// Used for probe-only search where only built-in probe tactics (simp, ring,
/// omega, etc.) are tried via the engine's `inject_probes()`. No LLM server needed.
pub struct NullPolicyProvider;

#[async_trait]
impl PolicyProvider for NullPolicyProvider {
    async fn generate_candidates(
        &self,
        _proof_state: &str,
        _n: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError> {
        Ok(vec![])
    }

    async fn generate_candidates_batch(
        &self,
        states: &[String],
        _n: usize,
    ) -> Result<Vec<Vec<GeneratedTactic>>, SearchError> {
        Ok(states.iter().map(|_| vec![]).collect())
    }
}

// ---------------------------------------------------------------------------
// ProofEnvironment for Arc<LeanPool>
// ---------------------------------------------------------------------------

#[async_trait]
impl ProofEnvironment for Arc<LeanPool> {
    async fn start_proof(
        &self,
        name: &str,
        statement: &str,
    ) -> Result<Box<dyn TacticRunner + Send>, SearchError> {
        // Try copyFrom(name) first — works for Mathlib theorems loaded in the environment.
        match self.start_proof_by_name_owned(name).await {
            Ok(handle) => return Ok(Box::new(handle)),
            Err(LeanError::LeanMessage(msg)) => {
                tracing::debug!(
                    name = name,
                    error = %msg,
                    "copyFrom failed, falling back to expr"
                );
            }
            Err(e) => return Err(SearchError::Lean(e)),
        }
        // Fallback: start proof from expression (works for simple type expressions).
        let handle = self
            .start_proof_owned(statement)
            .await
            .map_err(SearchError::Lean)?;
        Ok(Box::new(handle))
    }
}

// ---------------------------------------------------------------------------
// TacticRunner for ProofHandleOwned
// ---------------------------------------------------------------------------

#[async_trait]
impl TacticRunner for ProofHandleOwned {
    fn initial_state(&self) -> &ProofState {
        self.initial_state()
    }

    async fn apply_tactic(
        &mut self,
        state_id: u64,
        goal_id: Option<u64>,
        tactic: &str,
    ) -> Result<TacticResult, SearchError> {
        self.run_tactic(state_id, goal_id, tactic)
            .await
            .map_err(SearchError::Lean)
    }
}

// ---------------------------------------------------------------------------
// InferencePolicyProvider — wraps InferenceHandle (SGLang HTTP)
// ---------------------------------------------------------------------------

/// Policy provider backed by [`InferenceHandle`] (SGLang HTTP server).
pub struct InferencePolicyProvider {
    handle: InferenceHandle,
}

impl InferencePolicyProvider {
    /// Create a new `InferencePolicyProvider` from an inference handle.
    pub fn new(handle: InferenceHandle) -> Self {
        Self { handle }
    }

    /// Get a clone of the inference handle (for EBM encode closures, etc.).
    pub fn handle(&self) -> InferenceHandle {
        self.handle.clone()
    }
}

#[async_trait]
impl PolicyProvider for InferencePolicyProvider {
    async fn generate_candidates(
        &self,
        proof_state: &str,
        n: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError> {
        self.handle
            .generate_candidates(proof_state, n)
            .await
            .map_err(SearchError::Policy)
    }

    async fn generate_candidates_batch(
        &self,
        states: &[String],
        n: usize,
    ) -> Result<Vec<Vec<GeneratedTactic>>, SearchError> {
        self.handle
            .generate_candidates_batch(states, n)
            .await
            .map_err(SearchError::Policy)
    }
}

// ---------------------------------------------------------------------------
// ValueScorer for EBMValueFn
// ---------------------------------------------------------------------------

impl ValueScorer for EBMValueFn {
    fn score(&self, proof_state: &str) -> Result<f64, SearchError> {
        // block_in_place tells tokio this thread will block (on the internal
        // Mutex + potentially the encode HTTP call), so it spawns a replacement
        // worker. Falls back to direct call when not on a multi-threaded runtime.
        let score_fn = || self.score(proof_state).map_err(SearchError::Scorer);
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(score_fn)
            }
            _ => score_fn(),
        }
    }

    fn score_batch(&self, proof_states: &[&str]) -> Result<Vec<f64>, SearchError> {
        let batch_fn = || self.score_batch(proof_states).map_err(SearchError::Scorer);
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(batch_fn)
            }
            _ => batch_fn(),
        }
    }
}

// ---------------------------------------------------------------------------
// CachingEncoder — wraps InferenceHandle with an embedding cache
// ---------------------------------------------------------------------------

/// Batch encoder that caches embeddings to avoid redundant HTTP calls.
///
/// Checks a thread-safe `HashMap` before calling the underlying
/// `InferenceHandle::encode_batch`. Cache hits are returned directly;
/// only misses go to the server. Results are stored back in the cache.
pub struct CachingEncoder {
    handle: InferenceHandle,
    cache: Arc<std::sync::Mutex<HashMap<String, Vec<f32>>>>,
    hidden_size: usize,
}

impl CachingEncoder {
    /// Create a new `CachingEncoder`.
    ///
    /// - `handle`: the SGLang inference handle for HTTP encode calls
    /// - `hidden_size`: embedding dimension (used for zero-vector fallback)
    pub fn new(handle: InferenceHandle, hidden_size: usize) -> Self {
        Self {
            handle,
            cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            hidden_size,
        }
    }
}

#[async_trait]
impl BatchEncoder for CachingEncoder {
    async fn encode_batch(&self, states: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut results: Vec<Option<Vec<f32>>> = vec![None; states.len()];
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_texts: Vec<String> = Vec::new();

        // Phase 1: check cache for all states
        {
            let cache = self
                .cache
                .lock()
                .map_err(|e| anyhow::anyhow!("Embedding cache lock poisoned: {e}"))?;
            for (i, state) in states.iter().enumerate() {
                if let Some(cached) = cache.get(state.as_str()) {
                    results[i] = Some(cached.clone());
                } else {
                    miss_indices.push(i);
                    miss_texts.push(state.clone());
                }
            }
        }

        // Phase 2: batch-encode cache misses via HTTP
        if !miss_texts.is_empty() {
            let batch_results = self.handle.encode_batch(&miss_texts).await?;
            let mut cache = self
                .cache
                .lock()
                .map_err(|e| anyhow::anyhow!("Embedding cache lock poisoned: {e}"))?;
            for (j, &idx) in miss_indices.iter().enumerate() {
                match &batch_results[j] {
                    Ok(embedding) => {
                        cache.insert(states[idx].clone(), embedding.data.clone());
                        results[idx] = Some(embedding.data.clone());
                    }
                    Err(e) => {
                        tracing::warn!(
                            state_idx = idx,
                            error = %e,
                            "Batch encode failed for state, using zero embedding"
                        );
                        results[idx] = Some(vec![0.0; self.hidden_size]);
                    }
                }
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }
}
