//! Mock implementations of search traits for testing without Lean or LLM.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lean_repl::{ProofState, TacticResult};
use policy::GeneratedTactic;

use crate::engine::{PolicyProvider, ProofEnvironment, SearchError, TacticRunner};

/// Convenience constructor for a `GeneratedTactic`.
pub fn make_tactic(text: &str, log_prob: f64) -> GeneratedTactic {
    GeneratedTactic {
        text: text.to_string(),
        raw_text: text.to_string(),
        log_prob,
        tokens: vec![],
    }
}

// ---------------------------------------------------------------------------
// MockPolicy
// ---------------------------------------------------------------------------

/// Mock policy that returns canned tactic candidates based on state text.
pub struct MockPolicy {
    responses: HashMap<String, Vec<GeneratedTactic>>,
    contains_responses: Vec<(String, Vec<GeneratedTactic>)>,
    default_responses: Vec<GeneratedTactic>,
}

impl Default for MockPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl MockPolicy {
    /// Create an empty mock policy with no responses.
    pub fn new() -> Self {
        Self {
            responses: HashMap::new(),
            contains_responses: Vec::new(),
            default_responses: Vec::new(),
        }
    }

    /// Create a mock policy with default responses for any state.
    pub fn with_default(tactics: Vec<GeneratedTactic>) -> Self {
        Self {
            responses: HashMap::new(),
            contains_responses: Vec::new(),
            default_responses: tactics,
        }
    }

    /// Add a canned response for an exact state text match.
    pub fn add_response(&mut self, state_pattern: &str, tactics: Vec<GeneratedTactic>) {
        self.responses
            .insert(state_pattern.to_string(), tactics);
    }

    /// Add a canned response that matches if the proof state *contains* the pattern.
    ///
    /// Contains-matches are checked after exact matches. Useful for matching
    /// real Lean output where formatting may vary (e.g., `"n = n"` matches
    /// `"n : Nat\n⊢ n = n"` regardless of extra hypotheses).
    pub fn add_contains_response(&mut self, pattern: &str, tactics: Vec<GeneratedTactic>) {
        self.contains_responses
            .push((pattern.to_string(), tactics));
    }
}

#[async_trait]
impl PolicyProvider for MockPolicy {
    async fn generate_whole_proofs(
        &self,
        proof_state: &str,
        _n: usize,
        _max_tokens: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError> {
        // 1. Exact match
        if let Some(tactics) = self.responses.get(proof_state) {
            return Ok(tactics.clone());
        }
        // 2. Contains match (first match wins)
        for (pattern, tactics) in &self.contains_responses {
            if proof_state.contains(pattern.as_str()) {
                return Ok(tactics.clone());
            }
        }
        // 3. Default fallback
        if !self.default_responses.is_empty() {
            Ok(self.default_responses.clone())
        } else {
            Ok(vec![])
        }
    }
}

// ---------------------------------------------------------------------------
// MockEnvironment + MockTacticRunner
// ---------------------------------------------------------------------------

/// Mock proof environment that returns canned tactic results.
pub struct MockEnvironment {
    tactic_responses: Arc<HashMap<(u64, String), TacticResult>>,
    initial_state_id: u64,
}

impl Default for MockEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

impl MockEnvironment {
    /// Create a new mock environment with state_id 0 as initial.
    pub fn new() -> Self {
        Self {
            tactic_responses: Arc::new(HashMap::new()),
            initial_state_id: 0,
        }
    }

    /// Add a canned tactic result for a `(state_id, tactic)` pair.
    pub fn add_response(&mut self, state_id: u64, tactic: &str, result: TacticResult) {
        Arc::get_mut(&mut self.tactic_responses)
            .expect("cannot add responses after cloning")
            .insert((state_id, tactic.to_string()), result);
    }
}

#[async_trait]
impl ProofEnvironment for MockEnvironment {
    async fn start_proof(
        &self,
        _name: &str,
        _statement: &str,
    ) -> Result<Box<dyn TacticRunner + Send>, SearchError> {
        let initial = ProofState {
            state_id: self.initial_state_id,
            goals: Vec::new(), // goal.start returns empty goals, search synthesizes root
        };
        Ok(Box::new(MockTacticRunner {
            initial,
            responses: Arc::clone(&self.tactic_responses),
            next_state_id: self.initial_state_id + 1,
        }))
    }
}

/// Mock tactic runner that returns canned results by `(state_id, tactic)`.
struct MockTacticRunner {
    initial: ProofState,
    responses: Arc<HashMap<(u64, String), TacticResult>>,
    #[allow(dead_code)]
    next_state_id: u64,
}

#[async_trait]
impl TacticRunner for MockTacticRunner {
    fn initial_state(&self) -> &ProofState {
        &self.initial
    }

    async fn apply_tactic(
        &mut self,
        state_id: u64,
        _goal_id: Option<u64>,
        tactic: &str,
    ) -> Result<TacticResult, SearchError> {
        let key = (state_id, tactic.to_string());
        if let Some(result) = self.responses.get(&key) {
            Ok(result.clone())
        } else {
            Ok(TacticResult::Failed {
                message: format!("unknown tactic '{}' at state {}", tactic, state_id),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_tactic() {
        let t = make_tactic("intro n", -0.5);
        assert_eq!(t.text, "intro n");
        assert!((t.log_prob - (-0.5)).abs() < 1e-9);
        assert!(t.tokens.is_empty());
    }

    #[tokio::test]
    async fn test_mock_policy_exact_match() {
        let mut policy = MockPolicy::new();
        policy.add_response(
            "⊢ True",
            vec![make_tactic("trivial", -0.1)],
        );
        let result = policy.generate_whole_proofs("⊢ True", 32, 1024).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "trivial");
    }

    #[tokio::test]
    async fn test_mock_policy_default_fallback() {
        let policy = MockPolicy::with_default(vec![make_tactic("sorry", -5.0)]);
        let result = policy.generate_whole_proofs("anything", 32, 1024).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "sorry");
    }

    #[tokio::test]
    async fn test_mock_policy_empty() {
        let policy = MockPolicy::new();
        let result = policy.generate_whole_proofs("⊢ True", 32, 1024).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_mock_policy_contains_match() {
        let mut policy = MockPolicy::new();
        policy.add_contains_response("n = n", vec![make_tactic("rfl", -0.1)]);
        // Should match because "n : Nat\n⊢ n = n" contains "n = n"
        let result = policy
            .generate_whole_proofs("n : Nat\n⊢ n = n", 32, 1024)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "rfl");
    }

    #[tokio::test]
    async fn test_mock_policy_exact_before_contains() {
        let mut policy = MockPolicy::new();
        policy.add_response("⊢ True", vec![make_tactic("trivial", -0.1)]);
        policy.add_contains_response("True", vec![make_tactic("decide", -0.5)]);
        // Exact match should take priority over contains
        let result = policy.generate_whole_proofs("⊢ True", 32, 1024).await.unwrap();
        assert_eq!(result[0].text, "trivial");
    }

    #[tokio::test]
    async fn test_mock_environment_unknown_tactic() {
        let env = MockEnvironment::new();
        let mut runner = env.start_proof("test", "True").await.unwrap();
        let result = runner.apply_tactic(0, None, "nonexistent").await.unwrap();
        assert!(matches!(result, TacticResult::Failed { .. }));
    }

    #[tokio::test]
    async fn test_mock_environment_canned_response() {
        let mut env = MockEnvironment::new();
        env.add_response(
            0,
            "trivial",
            TacticResult::ProofComplete { state_id: 1 },
        );
        let mut runner = env.start_proof("test", "True").await.unwrap();
        let result = runner.apply_tactic(0, None, "trivial").await.unwrap();
        assert!(matches!(result, TacticResult::ProofComplete { state_id: 1 }));
    }
}
