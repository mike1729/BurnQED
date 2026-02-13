//! Bridges between search traits and real crate types (lean-repl, policy).

use std::sync::Arc;

use async_trait::async_trait;
use lean_repl::{LeanError, LeanPool, ProofHandleOwned, ProofState, TacticResult};
use policy::{GeneratedTactic, GenerationServiceHandle, TacticGenerator};

use ebm::inference::EBMValueFn;

use crate::engine::{PolicyProvider, ProofEnvironment, SearchError, TacticRunner, ValueScorer};

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
// MutexPolicyProvider — wraps TacticGenerator with Mutex for Send + Sync
// ---------------------------------------------------------------------------

/// Thread-safe wrapper around `TacticGenerator` implementing `PolicyProvider`.
///
/// `TacticGenerator` requires `&mut self` for generation (due to KV cache),
/// so we wrap it in a `Mutex` to satisfy the `Send + Sync` bounds on
/// `PolicyProvider`. The `Arc` allows sharing the generator with other
/// components (e.g. the EBM encoder closure).
pub struct MutexPolicyProvider {
    generator: Arc<std::sync::Mutex<TacticGenerator>>,
}

impl MutexPolicyProvider {
    /// Create a new `MutexPolicyProvider` wrapping the given generator.
    pub fn new(generator: TacticGenerator) -> Self {
        Self {
            generator: Arc::new(std::sync::Mutex::new(generator)),
        }
    }

    /// Create a new `MutexPolicyProvider` from a pre-shared `Arc<Mutex<TacticGenerator>>`.
    ///
    /// Use this when the same generator needs to be shared with other components
    /// (e.g. an EBM encode closure that calls `encode_only()`).
    pub fn new_shared(generator: Arc<std::sync::Mutex<TacticGenerator>>) -> Self {
        Self { generator }
    }

    /// Get a clone of the internal `Arc<Mutex<TacticGenerator>>`.
    ///
    /// Useful for creating encode closures that share the generator.
    pub fn shared_generator(&self) -> Arc<std::sync::Mutex<TacticGenerator>> {
        self.generator.clone()
    }
}

impl PolicyProvider for MutexPolicyProvider {
    fn generate_candidates(
        &self,
        proof_state: &str,
        n: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError> {
        let wait_start = std::time::Instant::now();
        let mut gen = self
            .generator
            .lock()
            .map_err(|e| SearchError::Policy(anyhow::anyhow!("{e}")))?;
        let lock_wait_ms = wait_start.elapsed().as_millis() as u64;

        let gen_start = std::time::Instant::now();
        let result = gen
            .generate_candidates(proof_state, n)
            .map_err(SearchError::Policy);
        let gen_ms = gen_start.elapsed().as_millis() as u64;

        if lock_wait_ms > 5000 {
            tracing::warn!(lock_wait_ms, gen_ms, "High mutex contention on policy generator");
        } else if lock_wait_ms > 100 {
            tracing::debug!(lock_wait_ms, gen_ms, "Mutex wait on policy generator");
        }
        result
    }
}

// ---------------------------------------------------------------------------
// ServicePolicyProvider — channel-based, no mutex contention
// ---------------------------------------------------------------------------

/// Policy provider backed by the generation service (channel-based, no mutex).
///
/// Preferred over `MutexPolicyProvider` for concurrent search, as it eliminates
/// mutex contention by routing all generation requests through a dedicated
/// service task via an mpsc channel.
pub struct ServicePolicyProvider {
    handle: GenerationServiceHandle,
}

impl ServicePolicyProvider {
    /// Create a new `ServicePolicyProvider` from a service handle.
    pub fn new(handle: GenerationServiceHandle) -> Self {
        Self { handle }
    }

    /// Get a clone of the service handle (for EBM encode closures, etc.).
    pub fn handle(&self) -> GenerationServiceHandle {
        self.handle.clone()
    }
}

impl PolicyProvider for ServicePolicyProvider {
    fn generate_candidates(
        &self,
        proof_state: &str,
        n: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError> {
        self.handle
            .generate_candidates_blocking(proof_state, n)
            .map_err(SearchError::Policy)
    }
}

// ---------------------------------------------------------------------------
// ValueScorer for EBMValueFn
// ---------------------------------------------------------------------------

impl ValueScorer for EBMValueFn {
    fn score(&self, proof_state: &str) -> Result<f64, SearchError> {
        self.score(proof_state).map_err(SearchError::Scorer)
    }
}
