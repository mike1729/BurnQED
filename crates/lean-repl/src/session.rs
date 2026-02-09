use crate::pool::{LeanPool, ProofHandle};
use crate::types::{Goal, LeanError, ProofState, TacticResult};

/// A proof session that tracks state across multiple tactic applications.
///
/// `ProofSession` provides a higher-level interface over `LeanPool` for
/// conducting a single proof attempt. It holds a [`ProofHandle`] that keeps
/// the same Pantograph worker for the entire session, ensuring state IDs
/// remain valid.
pub struct ProofSession<'a> {
    handle: ProofHandle<'a>,
    current_state: ProofState,
    history: Vec<(String, TacticResult)>,
    completed: bool,
}

impl<'a> ProofSession<'a> {
    /// Start a new proof session for the given expression.
    pub async fn new(pool: &'a LeanPool, expr: &str) -> Result<Self, LeanError> {
        let handle = pool.start_proof(expr).await?;
        let state = handle.initial_state().clone();

        Ok(Self {
            handle,
            current_state: state,
            history: Vec::new(),
            completed: false,
        })
    }

    /// Apply a tactic to the first open goal (goal_id = 0).
    ///
    /// Updates the session's current state on success. Returns a reference
    /// to the tactic result for inspection.
    pub async fn apply(&mut self, tactic: &str) -> Result<&TacticResult, LeanError> {
        self.apply_inner(None, tactic).await
    }

    /// Apply a tactic to a specific goal ID.
    pub async fn apply_to_goal(
        &mut self,
        goal_id: u64,
        tactic: &str,
    ) -> Result<&TacticResult, LeanError> {
        self.apply_inner(Some(goal_id), tactic).await
    }

    /// Shared implementation for `apply` and `apply_to_goal`.
    async fn apply_inner(
        &mut self,
        goal_id: Option<u64>,
        tactic: &str,
    ) -> Result<&TacticResult, LeanError> {
        if self.completed {
            return Err(LeanError::Protocol("Proof already complete".into()));
        }

        let result = self
            .handle
            .run_tactic(self.current_state.state_id, goal_id, tactic)
            .await?;

        match &result {
            TacticResult::Success { state_id, goals } => {
                self.current_state = ProofState {
                    state_id: *state_id,
                    goals: goals.clone(),
                };
            }
            TacticResult::ProofComplete { state_id } => {
                self.current_state = ProofState {
                    state_id: *state_id,
                    goals: Vec::new(),
                };
                self.completed = true;
            }
            TacticResult::Failed { .. } => {}
        }

        self.history.push((tactic.to_string(), result));
        Ok(&self.history.last().unwrap().1)
    }

    /// Whether the proof is complete (no remaining goals).
    pub fn is_complete(&self) -> bool {
        self.completed
    }

    /// Current goals in the proof state.
    pub fn current_goals(&self) -> &[Goal] {
        &self.current_state.goals
    }

    /// Current proof state.
    pub fn current_state(&self) -> &ProofState {
        &self.current_state
    }

    /// Number of tactics applied so far (proof depth).
    pub fn depth(&self) -> usize {
        self.history.len()
    }

    /// History of (tactic, result) pairs applied during this session.
    pub fn history(&self) -> &[(String, TacticResult)] {
        &self.history
    }
}
