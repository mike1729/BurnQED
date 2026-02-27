//! Bridges between search traits and real crate types (lean-repl, policy).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lean_repl::{LeanError, LeanPool, ProofHandleOwned, ProofState, TacticResult};
use policy::{GeneratedTactic, InferenceHandle};

#[cfg(feature = "burn-ebm")]
use ebm::inference::EBMValueFn;

#[cfg(feature = "burn-ebm")]
use crate::engine::ValueScorer;
use crate::engine::{PolicyProvider, ProofEnvironment, SearchError, TacticRunner};

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
    async fn generate_whole_proofs(
        &self,
        _proof_state: &str,
        _n: usize,
        _max_tokens: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError> {
        Ok(vec![])
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
        // Fallback: start proof from expression.
        // If the statement contains context (hypotheses before ⊢), extract just the goal type.
        // Full proof states like "x : Nat\n⊢ x = x" can't be parsed as expressions.
        let expr = if let Some(pos) = statement.find('⊢') {
            statement[pos + '⊢'.len_utf8()..].trim()
        } else {
            statement.trim()
        };
        match self.start_proof_owned(expr).await {
            Ok(handle) => Ok(Box::new(handle)),
            Err(e) => {
                tracing::warn!(
                    name = name,
                    error = %e,
                    "goal.start(expr) fallback also failed"
                );
                Err(SearchError::Lean(e))
            }
        }
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
    async fn generate_whole_proofs(
        &self,
        proof_state: &str,
        n: usize,
        max_tokens: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError> {
        self.handle
            .generate_whole_proofs(proof_state, n, max_tokens)
            .await
            .map_err(SearchError::Policy)
    }
}

// ---------------------------------------------------------------------------
// ValueScorer for EBMValueFn (burn-rs, v1 — behind feature gate)
// ---------------------------------------------------------------------------

#[cfg(feature = "burn-ebm")]
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
///
/// HTTP requests are chunked into sub-batches of `max_batch_size` to prevent
/// OOM on the encode server. Failed sub-batches are retried once individually.
pub struct CachingEncoder {
    handle: InferenceHandle,
    cache: Arc<std::sync::Mutex<HashMap<String, Vec<f32>>>>,
    hidden_size: usize,
    max_batch_size: usize,
}

impl CachingEncoder {
    /// Create a new `CachingEncoder`.
    ///
    /// - `handle`: the SGLang inference handle for HTTP encode calls
    /// - `hidden_size`: embedding dimension (used for zero-vector fallback)
    /// - `max_batch_size`: max states per HTTP request (prevents encode server OOM)
    pub fn new(handle: InferenceHandle, hidden_size: usize, max_batch_size: usize) -> Self {
        Self {
            handle,
            cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            hidden_size,
            max_batch_size,
        }
    }

    /// Encode multiple proof states into embedding vectors.
    ///
    /// Cache hits are returned directly; misses are sent to the encode server
    /// in sub-batches of `max_batch_size`. The encode server handles concurrent
    /// requests efficiently via server-side dynamic batching.
    pub async fn encode_batch(&self, states: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
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

        // Phase 2: encode cache misses in sub-batches.
        // The encode server coalesces concurrent requests server-side, so we
        // use batch encode calls (one per sub-batch) rather than sequential singles.
        for chunk_start in (0..miss_texts.len()).step_by(self.max_batch_size) {
            let chunk_end = (chunk_start + self.max_batch_size).min(miss_texts.len());
            let chunk_texts = &miss_texts[chunk_start..chunk_end];
            let chunk_indices = &miss_indices[chunk_start..chunk_end];

            let batch_result = self.handle.encode_batch(chunk_texts).await;

            match batch_result {
                Ok(batch_results) => {
                    let mut cache = self
                        .cache
                        .lock()
                        .map_err(|e| anyhow::anyhow!("Embedding cache lock poisoned: {e}"))?;
                    for (j, &idx) in chunk_indices.iter().enumerate() {
                        match &batch_results[j] {
                            Ok(embedding) => {
                                if embedding.data.iter().all(|&v| v == 0.0) {
                                    tracing::warn!(
                                        state_idx = idx,
                                        "Encode server returned all-zero embedding"
                                    );
                                }
                                cache.insert(states[idx].clone(), embedding.data.clone());
                                results[idx] = Some(embedding.data.clone());
                            }
                            Err(e) => {
                                tracing::warn!(
                                    state_idx = idx,
                                    error = %e,
                                    "Encode failed for individual state, using zero embedding"
                                );
                                results[idx] = Some(vec![0.0; self.hidden_size]);
                            }
                        }
                    }
                }
                Err(e) => {
                    // Sub-batch failed (likely OOM). Wait with jitter for server to recover,
                    // then retry individually.
                    // Jitter via nanosecond timestamp to avoid thundering herd
                    let jitter_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.subsec_nanos() as u64 % 1000)
                        .unwrap_or(500);
                    let wait = std::time::Duration::from_millis(1000 + jitter_ms);
                    tracing::warn!(
                        chunk_size = chunk_texts.len(),
                        error = %e,
                        wait_ms = wait.as_millis() as u64,
                        "Sub-batch encode failed, retrying individually"
                    );
                    tokio::time::sleep(wait).await;
                    for (j, &idx) in chunk_indices.iter().enumerate() {
                        let single = vec![chunk_texts[j].clone()];
                        match self.handle.encode_batch(&single).await {
                            Ok(single_results) => {
                                match &single_results[0] {
                                    Ok(embedding) => {
                                        let mut cache = self
                                            .cache
                                            .lock()
                                            .map_err(|e| anyhow::anyhow!("Embedding cache lock poisoned: {e}"))?;
                                        cache.insert(states[idx].clone(), embedding.data.clone());
                                        results[idx] = Some(embedding.data.clone());
                                    }
                                    Err(e) => {
                                        tracing::warn!(state_idx = idx, error = %e, "Individual encode failed");
                                        results[idx] = Some(vec![0.0; self.hidden_size]);
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(state_idx = idx, error = %e, "Individual encode retry failed");
                                results[idx] = Some(vec![0.0; self.hidden_size]);
                            }
                        }
                    }
                }
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }
}
