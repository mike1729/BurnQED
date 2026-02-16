//! Bridges between search traits and real crate types (lean-repl, policy).

use std::sync::Arc;

use async_trait::async_trait;
use lean_repl::{LeanError, LeanPool, ProofHandleOwned, ProofState, TacticResult};
use policy::{GeneratedTactic, InferenceHandle};

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
        let futs: Vec<_> = states
            .iter()
            .map(|s| self.handle.generate_candidates(s, n))
            .collect();
        let results = futures::future::join_all(futs).await;
        results
            .into_iter()
            .map(|r| r.map_err(SearchError::Policy))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ValueScorer for EBMValueFn
// ---------------------------------------------------------------------------

impl ValueScorer for EBMValueFn {
    fn score(&self, proof_state: &str) -> Result<f64, SearchError> {
        // block_in_place tells tokio this thread will block (on the internal
        // Mutex + encode HTTP call), so it spawns a replacement worker.
        // Without this, concurrent search tasks starve the thread pool.
        // Falls back to direct call when not on a multi-threaded runtime (tests).
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
