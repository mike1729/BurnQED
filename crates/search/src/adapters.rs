//! Bridges between search traits and real crate types (lean-repl, policy).

use std::sync::Arc;

use async_trait::async_trait;
use lean_repl::{LeanPool, ProofHandleOwned, ProofState, TacticResult};
use policy::{GeneratedTactic, TacticGenerator};

use ebm::inference::EBMValueFn;

use crate::engine::{PolicyProvider, ProofEnvironment, SearchError, TacticRunner, ValueScorer};

// ---------------------------------------------------------------------------
// ProofEnvironment for Arc<LeanPool>
// ---------------------------------------------------------------------------

#[async_trait]
impl ProofEnvironment for Arc<LeanPool> {
    async fn start_proof(
        &self,
        statement: &str,
    ) -> Result<Box<dyn TacticRunner + Send>, SearchError> {
        let handle = self.start_proof_owned(statement).await.map_err(SearchError::Lean)?;
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
/// `PolicyProvider`.
pub struct MutexPolicyProvider {
    generator: std::sync::Mutex<TacticGenerator>,
}

impl MutexPolicyProvider {
    /// Create a new `MutexPolicyProvider` wrapping the given generator.
    pub fn new(generator: TacticGenerator) -> Self {
        Self {
            generator: std::sync::Mutex::new(generator),
        }
    }
}

impl PolicyProvider for MutexPolicyProvider {
    fn generate_candidates(
        &self,
        proof_state: &str,
        n: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError> {
        let mut gen = self
            .generator
            .lock()
            .map_err(|e| SearchError::Policy(anyhow::anyhow!("{e}")))?;
        gen.generate_candidates(proof_state, n)
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
