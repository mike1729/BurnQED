//! Best-first search engine with priority queue, node expansion, and scoring.

use std::collections::BinaryHeap;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use ordered_float::OrderedFloat;

use lean_repl::{Goal, ProofState, TacticResult};
use policy::GeneratedTactic;
use trajectory::{SearchResult, SearchStats, TrajectoryLabel, TrajectoryRecord};

use crate::config::SearchConfig;
use crate::node::{extract_tactic_sequence, ScoredNode, SearchNode};

/// Errors that can occur during proof search.
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    /// Error from the Lean REPL.
    #[error("Lean error: {0}")]
    Lean(#[from] lean_repl::LeanError),
    /// Error from the policy (LLM) provider.
    #[error("Policy error: {0}")]
    Policy(#[source] anyhow::Error),
    /// Error from the value scorer (EBM).
    #[error("Scorer error: {0}")]
    Scorer(#[source] anyhow::Error),
    /// Failed to start a proof environment.
    #[error("Failed to start proof: {0}")]
    ProofStart(String),
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Proof environment that can start proofs and produce tactic runners.
#[async_trait]
pub trait ProofEnvironment: Send + Sync {
    /// Start a new proof, returning a runner that can apply tactics.
    ///
    /// `name` is the fully-qualified theorem name (e.g. `"Nat.add_comm"`).
    /// `statement` is a valid Lean expression or a pretty-printed proof state.
    /// Implementations may try `copyFrom(name)` first, falling back to `expr(statement)`.
    async fn start_proof(
        &self,
        name: &str,
        statement: &str,
    ) -> Result<Box<dyn TacticRunner + Send>, SearchError>;
}

/// Runner for applying tactics to a single proof (holds worker for lifetime).
#[async_trait]
pub trait TacticRunner: Send {
    /// The initial proof state returned by `goal.start`.
    fn initial_state(&self) -> &ProofState;

    /// Apply a tactic to the given state, returning the result.
    async fn apply_tactic(
        &mut self,
        state_id: u64,
        goal_id: Option<u64>,
        tactic: &str,
    ) -> Result<TacticResult, SearchError>;
}

/// LLM policy that generates tactic candidates for a proof state.
///
/// Sync trait — matches candle's synchronous inference API.
pub trait PolicyProvider: Send + Sync {
    /// Generate up to `n` candidate tactics for the given proof state text.
    fn generate_candidates(
        &self,
        proof_state: &str,
        n: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError>;
}

/// Value function that scores proof states (e.g., EBM energy).
///
/// Sync trait — matches burn's synchronous inference API.
pub trait ValueScorer: Send + Sync {
    /// Score a proof state. Convention: higher = more provable.
    fn score(&self, proof_state: &str) -> Result<f64, SearchError>;
}

// ---------------------------------------------------------------------------
// SearchEngine
// ---------------------------------------------------------------------------

/// Best-first proof search engine.
///
/// Expands nodes by priority (combined LLM + EBM score), generating
/// tactic candidates and verifying them against Lean.
#[derive(Clone)]
pub struct SearchEngine {
    config: SearchConfig,
}

impl SearchEngine {
    /// Create a new search engine with the given configuration.
    pub fn new(config: SearchConfig) -> Self {
        config.validate();
        Self { config }
    }

    /// Search for a proof of a single theorem.
    ///
    /// Returns a `SearchResult` with the proof tactics (if found) and
    /// all trajectory records for EBM training.
    pub async fn search_one(
        &self,
        env: &dyn ProofEnvironment,
        policy: &dyn PolicyProvider,
        scorer: Option<&dyn ValueScorer>,
        theorem_name: &str,
        statement: &str,
    ) -> Result<SearchResult, SearchError> {
        let start_time = std::time::Instant::now();
        let mut stats = SearchStats::default();

        let mut runner = env.start_proof(theorem_name, statement).await?;
        let initial_state = runner.initial_state().clone();

        // Build root node. goal.start returns empty goals, so we synthesize
        // the root state from the theorem statement.
        // If statement already contains '⊢' (e.g. from theorem_index proof states),
        // use it as-is; otherwise prepend "⊢ " (e.g. for simple expressions like "True").
        let root_pp = if statement.contains('⊢') {
            statement.to_string()
        } else {
            format!("⊢ {statement}")
        };
        let root_goals = vec![Goal::parse(0, &root_pp)];
        let root_terminal = root_goals.is_empty();

        let root = SearchNode {
            state_id: initial_state.state_id,
            state_pp: root_pp.clone(),
            goals: root_goals,
            parent: None,
            tactic_applied: String::new(),
            depth: 0,
            llm_log_prob: 0.0,
            ebm_score: 0.0,
            is_terminal: root_terminal,
        };

        let mut arena: Vec<SearchNode> = vec![root];

        // If root is already terminal (shouldn't happen for real theorems,
        // but handle gracefully)
        if root_terminal {
            let wall_time_ms = start_time.elapsed().as_millis() as u64;
            let records = build_trajectory_records(&arena, theorem_name);
            return Ok(SearchResult {
                theorem_name: theorem_name.to_string(),
                proved: true,
                proof_tactics: vec![],
                nodes_expanded: 0,
                total_states: 1,
                max_depth_reached: 0,
                wall_time_ms,
                all_records: records,
                stats,
            });
        }

        // Score root with EBM if available
        if let Some(scorer) = scorer {
            let score = scorer.score(&arena[0].state_pp)?;
            arena[0].ebm_score = score;
        }

        let root_score = arena[0].combined_score(self.config.alpha, self.config.beta);
        let mut frontier = BinaryHeap::new();
        frontier.push(ScoredNode {
            node_index: 0,
            score: OrderedFloat(root_score),
        });

        let mut nodes_expanded: u32 = 0;
        let mut max_depth_reached: u32 = 0;

        // Main search loop
        while let Some(current) = frontier.pop() {
            if nodes_expanded >= self.config.max_nodes {
                break;
            }

            let timeout_secs = self.config.timeout_per_theorem;
            if timeout_secs > 0 && start_time.elapsed().as_secs() >= timeout_secs {
                tracing::info!(
                    theorem = theorem_name,
                    elapsed_s = start_time.elapsed().as_secs(),
                    "Search timed out"
                );
                break;
            }

            let node_idx = current.node_index;
            let node_state_id = arena[node_idx].state_id;
            let goals_text = arena[node_idx].goals_as_text();
            let node_depth = arena[node_idx].depth;

            // Skip nodes at max depth — all children would exceed the limit,
            // so generating candidates and calling Lean would be wasted work.
            if node_depth >= self.config.max_depth {
                nodes_expanded += 1;
                stats.nodes_pruned += 1;
                continue;
            }

            tracing::info!(
                node = node_idx,
                state_id = node_state_id,
                depth = node_depth,
                score = %current.score,
                "Expanding node"
            );

            // Generate tactic candidates
            let gen_start = std::time::Instant::now();
            let mut candidates = policy.generate_candidates(&goals_text, self.config.num_candidates)?;
            stats.total_generate_time_ms += gen_start.elapsed().as_millis() as u64;

            // Inject fallback tactics when LLM returns nothing
            if candidates.is_empty() && !self.config.fallback_tactics.is_empty() {
                tracing::info!(
                    node = node_idx,
                    "LLM returned no candidates, using fallback tactics"
                );
                candidates = self
                    .config
                    .fallback_tactics
                    .iter()
                    .map(|t| GeneratedTactic {
                        text: t.clone(),
                        log_prob: -100.0,
                        tokens: vec![],
                    })
                    .collect();
            }

            for candidate in candidates {
                stats.total_tactic_attempts += 1;
                let tactic_start = std::time::Instant::now();
                let result = runner
                    .apply_tactic(node_state_id, None, &candidate.text)
                    .await?;
                stats.total_lean_time_ms += tactic_start.elapsed().as_millis() as u64;

                match result {
                    TacticResult::Success { state_id, goals } => {
                        let child_depth = node_depth + 1;

                        let state_pp = goals
                            .iter()
                            .map(|g| g.raw.as_str())
                            .collect::<Vec<_>>()
                            .join("\n\n");

                        let mut ebm_score = 0.0;
                        if let Some(scorer) = scorer {
                            ebm_score = scorer.score(&state_pp)?;
                        }

                        let child = SearchNode {
                            state_id,
                            state_pp,
                            goals,
                            parent: Some(node_idx),
                            tactic_applied: candidate.text.clone(),
                            depth: child_depth,
                            llm_log_prob: candidate.log_prob,
                            ebm_score,
                            is_terminal: false,
                        };

                        if child_depth > max_depth_reached {
                            max_depth_reached = child_depth;
                        }

                        let child_score =
                            child.combined_score(self.config.alpha, self.config.beta);
                        let child_idx = arena.len();
                        arena.push(child);

                        frontier.push(ScoredNode {
                            node_index: child_idx,
                            score: OrderedFloat(child_score),
                        });
                        stats.peak_frontier_size =
                            stats.peak_frontier_size.max(frontier.len());
                    }
                    TacticResult::ProofComplete { state_id } => {
                        let child_depth = node_depth + 1;
                        stats.nodes_terminal += 1;
                        let terminal = SearchNode {
                            state_id,
                            state_pp: String::new(),
                            goals: vec![],
                            parent: Some(node_idx),
                            tactic_applied: candidate.text.clone(),
                            depth: child_depth,
                            llm_log_prob: candidate.log_prob,
                            ebm_score: 0.0,
                            is_terminal: true,
                        };

                        if child_depth > max_depth_reached {
                            max_depth_reached = child_depth;
                        }

                        let terminal_idx = arena.len();
                        arena.push(terminal);

                        let proof_tactics = extract_tactic_sequence(&arena, terminal_idx);
                        let wall_time_ms = start_time.elapsed().as_millis() as u64;
                        let total_states = arena.len() as u32;
                        let records = build_trajectory_records(&arena, theorem_name);

                        stats.nodes_expanded = nodes_expanded;

                        tracing::info!(
                            theorem = theorem_name,
                            steps = proof_tactics.len(),
                            nodes = nodes_expanded,
                            states = total_states,
                            time_ms = wall_time_ms,
                            "Proof found"
                        );

                        return Ok(SearchResult {
                            theorem_name: theorem_name.to_string(),
                            proved: true,
                            proof_tactics,
                            nodes_expanded,
                            total_states,
                            max_depth_reached,
                            wall_time_ms,
                            all_records: records,
                            stats,
                        });
                    }
                    TacticResult::Failed { message } => {
                        stats.total_tactic_failures += 1;
                        tracing::debug!(
                            tactic = candidate.text,
                            error = message,
                            "Tactic failed"
                        );
                    }
                }
            }

            nodes_expanded += 1;
        }

        // Search exhausted without finding a proof
        let wall_time_ms = start_time.elapsed().as_millis() as u64;
        let total_states = arena.len() as u32;
        let records = build_trajectory_records(&arena, theorem_name);

        stats.nodes_expanded = nodes_expanded;

        tracing::info!(
            theorem = theorem_name,
            nodes = nodes_expanded,
            states = total_states,
            time_ms = wall_time_ms,
            "Search exhausted without proof"
        );

        Ok(SearchResult {
            theorem_name: theorem_name.to_string(),
            proved: false,
            proof_tactics: vec![],
            nodes_expanded,
            total_states,
            max_depth_reached,
            wall_time_ms,
            all_records: records,
            stats,
        })
    }
}

/// Convert the search arena into trajectory records for Parquet output.
fn build_trajectory_records(arena: &[SearchNode], theorem_name: &str) -> Vec<TrajectoryRecord> {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    arena
        .iter()
        .map(|node| {
            let parent_state_id = node.parent.map(|p_idx| arena[p_idx].state_id);
            TrajectoryRecord {
                theorem_name: theorem_name.to_string(),
                state_id: node.state_id,
                state_pp: node.state_pp.clone(),
                tactic_applied: node.tactic_applied.clone(),
                parent_state_id,
                label: TrajectoryLabel::Unknown,
                depth_from_root: node.depth,
                remaining_depth: -1,
                llm_log_prob: node.llm_log_prob,
                ebm_score: node.ebm_score,
                is_proof_complete: node.is_terminal,
                timestamp_ms: now_ms,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mocks::{make_tactic, MockEnvironment, MockPolicy};
    use lean_repl::Goal;

    #[tokio::test]
    async fn test_search_finds_one_step_proof() {
        // "True" proved by "trivial"
        let mut env = MockEnvironment::new();
        env.add_response(0, "trivial", TacticResult::ProofComplete { state_id: 1 });

        let mut policy = MockPolicy::new();
        policy.add_response("⊢ True", vec![make_tactic("trivial", -0.1)]);

        let engine = SearchEngine::new(SearchConfig::default());
        let result = engine
            .search_one(&env, &policy, None, "test_true", "True")
            .await
            .unwrap();

        assert!(result.proved);
        assert_eq!(result.proof_tactics, vec!["trivial"]);
        assert_eq!(result.theorem_name, "test_true");
    }

    #[tokio::test]
    async fn test_search_finds_two_step_proof() {
        // Two-step proof: "intro n" then "rfl"
        let mut env = MockEnvironment::new();
        env.add_response(
            0,
            "intro n",
            TacticResult::Success {
                state_id: 1,
                goals: vec![Goal::parse(0, "n : Nat\n⊢ n = n")],
            },
        );
        env.add_response(1, "rfl", TacticResult::ProofComplete { state_id: 2 });

        let mut policy = MockPolicy::new();
        policy.add_response("⊢ ∀ (n : Nat), n = n", vec![make_tactic("intro n", -0.3)]);
        policy.add_response("n : Nat\n⊢ n = n", vec![make_tactic("rfl", -0.1)]);

        let engine = SearchEngine::new(SearchConfig::default());
        let result = engine
            .search_one(&env, &policy, None, "nat_refl", "∀ (n : Nat), n = n")
            .await
            .unwrap();

        assert!(result.proved);
        assert_eq!(result.proof_tactics, vec!["intro n", "rfl"]);
    }

    #[tokio::test]
    async fn test_search_respects_node_budget() {
        // No working tactics → search should stop after max_nodes expansions
        let env = MockEnvironment::new();
        let policy = MockPolicy::with_default(vec![make_tactic("bad", -1.0)]);

        let config = SearchConfig {
            max_nodes: 3,
            ..SearchConfig::default()
        };
        let engine = SearchEngine::new(config);
        let result = engine
            .search_one(&env, &policy, None, "hard", "∀ x, x = x")
            .await
            .unwrap();

        assert!(!result.proved);
        assert!(result.nodes_expanded <= 3);
    }

    #[tokio::test]
    async fn test_search_respects_depth_limit() {
        // Chain: 0 -> 1 -> 2 -> 3, but max_depth=2 should prevent reaching depth 3
        let mut env = MockEnvironment::new();
        env.add_response(
            0,
            "step1",
            TacticResult::Success {
                state_id: 1,
                goals: vec![Goal::parse(0, "⊢ A")],
            },
        );
        env.add_response(
            1,
            "step2",
            TacticResult::Success {
                state_id: 2,
                goals: vec![Goal::parse(0, "⊢ B")],
            },
        );
        env.add_response(
            2,
            "step3",
            TacticResult::ProofComplete { state_id: 3 },
        );

        let mut policy = MockPolicy::new();
        policy.add_response("⊢ deep_theorem", vec![make_tactic("step1", -0.1)]);
        policy.add_response("⊢ A", vec![make_tactic("step2", -0.1)]);
        policy.add_response("⊢ B", vec![make_tactic("step3", -0.1)]);

        let config = SearchConfig {
            max_depth: 2,
            ..SearchConfig::default()
        };
        let engine = SearchEngine::new(config);
        let result = engine
            .search_one(&env, &policy, None, "deep", "deep_theorem")
            .await
            .unwrap();

        // step3 at depth 3 should be skipped
        assert!(!result.proved);
        assert!(result.max_depth_reached <= 2);
    }

    #[tokio::test]
    async fn test_search_handles_tactic_failure() {
        // "bad" fails, "good" succeeds
        let mut env = MockEnvironment::new();
        env.add_response(0, "good", TacticResult::ProofComplete { state_id: 1 });

        let mut policy = MockPolicy::new();
        policy.add_response(
            "⊢ True",
            vec![make_tactic("bad", -0.5), make_tactic("good", -0.1)],
        );

        let engine = SearchEngine::new(SearchConfig::default());
        let result = engine
            .search_one(&env, &policy, None, "resilient", "True")
            .await
            .unwrap();

        assert!(result.proved);
        assert_eq!(result.proof_tactics, vec!["good"]);
    }

    #[tokio::test]
    async fn test_search_empty_frontier() {
        // Policy returns no tactics and fallbacks disabled → frontier is empty after root
        let env = MockEnvironment::new();
        let policy = MockPolicy::new(); // returns empty for all states

        let config = SearchConfig {
            fallback_tactics: vec![], // disable fallbacks for this test
            ..SearchConfig::default()
        };
        let engine = SearchEngine::new(config);
        let result = engine
            .search_one(&env, &policy, None, "impossible", "∀ x, x = x")
            .await
            .unwrap();

        assert!(!result.proved);
        assert_eq!(result.proof_tactics.len(), 0);
    }

    #[tokio::test]
    async fn test_search_result_has_trajectory_records() {
        let mut env = MockEnvironment::new();
        env.add_response(
            0,
            "intro n",
            TacticResult::Success {
                state_id: 1,
                goals: vec![Goal::parse(0, "n : Nat\n⊢ n = n")],
            },
        );
        env.add_response(1, "rfl", TacticResult::ProofComplete { state_id: 2 });

        let mut policy = MockPolicy::new();
        policy.add_response("⊢ ∀ (n : Nat), n = n", vec![make_tactic("intro n", -0.3)]);
        policy.add_response("n : Nat\n⊢ n = n", vec![make_tactic("rfl", -0.1)]);

        let engine = SearchEngine::new(SearchConfig::default());
        let result = engine
            .search_one(&env, &policy, None, "nat_refl", "∀ (n : Nat), n = n")
            .await
            .unwrap();

        // Should have 3 records: root, intro n, rfl
        assert_eq!(result.all_records.len(), 3);
        assert_eq!(result.all_records[0].state_id, 0);
        assert!(result.all_records[0].parent_state_id.is_none());
        assert_eq!(result.all_records[1].tactic_applied, "intro n");
        assert_eq!(result.all_records[2].state_id, 2);
        assert!(result.all_records[2].is_proof_complete);
    }

    #[tokio::test]
    async fn test_search_timeout_zero_means_no_timeout() {
        // timeout_per_theorem=0 disables the timeout check (the `if timeout_secs > 0` guard).
        // Verify search still works normally.
        let mut env = MockEnvironment::new();
        env.add_response(0, "trivial", TacticResult::ProofComplete { state_id: 1 });

        let mut policy = MockPolicy::new();
        policy.add_response("⊢ True", vec![make_tactic("trivial", -0.1)]);

        let config = SearchConfig {
            timeout_per_theorem: 0,
            ..SearchConfig::default()
        };
        let engine = SearchEngine::new(config);
        let result = engine
            .search_one(&env, &policy, None, "no_timeout", "True")
            .await
            .unwrap();

        assert!(result.proved);
        assert_eq!(result.proof_tactics, vec!["trivial"]);
    }

    #[tokio::test]
    async fn test_search_policy_returns_no_candidates() {
        // Policy returns Ok(vec![]) for all states. With fallback_tactics disabled,
        // the search loop should skip expansion gracefully, increment nodes_expanded,
        // and eventually exhaust the frontier (root has no children).
        let env = MockEnvironment::new();
        let policy = MockPolicy::new(); // returns empty vec for all states

        let config = SearchConfig {
            max_nodes: 3,
            fallback_tactics: vec![], // disable fallbacks for this test
            ..SearchConfig::default()
        };
        let engine = SearchEngine::new(config);
        let result = engine
            .search_one(&env, &policy, None, "empty_candidates", "∀ x, x = x")
            .await
            .unwrap();

        assert!(!result.proved);
        assert!(result.nodes_expanded <= 3, "Budget should be respected");
    }

    /// A constant value scorer for testing.
    struct ConstScorer(f64);
    impl ValueScorer for ConstScorer {
        fn score(&self, _proof_state: &str) -> Result<f64, SearchError> {
            Ok(self.0)
        }
    }

    #[tokio::test]
    async fn test_search_with_value_scorer() {
        // Two-step proof with a ValueScorer active
        let mut env = MockEnvironment::new();
        env.add_response(
            0,
            "intro n",
            TacticResult::Success {
                state_id: 1,
                goals: vec![Goal::parse(0, "n : Nat\n⊢ n = n")],
            },
        );
        env.add_response(1, "rfl", TacticResult::ProofComplete { state_id: 2 });

        let mut policy = MockPolicy::new();
        policy.add_response("⊢ ∀ (n : Nat), n = n", vec![make_tactic("intro n", -0.3)]);
        policy.add_response("n : Nat\n⊢ n = n", vec![make_tactic("rfl", -0.1)]);

        let scorer = ConstScorer(1.0);
        let engine = SearchEngine::new(SearchConfig::default());
        let result = engine
            .search_one(&env, &policy, Some(&scorer), "nat_refl_scored", "∀ (n : Nat), n = n")
            .await
            .unwrap();

        assert!(result.proved, "Should prove with scorer active");
        assert_eq!(result.proof_tactics, vec!["intro n", "rfl"]);

        // Verify ebm_score fields are set in trajectory records
        // Root and child nodes (except terminal) should have been scored
        let scored_records: Vec<_> = result
            .all_records
            .iter()
            .filter(|r| r.ebm_score != 0.0)
            .collect();
        assert!(
            !scored_records.is_empty(),
            "At least one record should have a non-zero ebm_score when scorer is active"
        );
    }

    #[tokio::test]
    async fn test_search_stats_populated() {
        // Mix of good and bad tactics to exercise stats fields
        let mut env = MockEnvironment::new();
        env.add_response(
            0,
            "good_tactic",
            TacticResult::ProofComplete { state_id: 1 },
        );
        // "bad_tactic" will fall through to MockEnvironment's default (Failed)

        let mut policy = MockPolicy::new();
        policy.add_response(
            "⊢ True",
            vec![
                make_tactic("bad_tactic", -2.0),
                make_tactic("good_tactic", -0.1),
            ],
        );

        let config = SearchConfig {
            max_nodes: 10,
            ..SearchConfig::default()
        };
        let engine = SearchEngine::new(config);
        let result = engine
            .search_one(&env, &policy, None, "stats_test", "True")
            .await
            .unwrap();

        assert!(result.proved);

        // Verify SearchStats fields are populated
        let stats = &result.stats;
        assert!(
            stats.total_tactic_attempts > 0,
            "Should have attempted at least one tactic, got {}",
            stats.total_tactic_attempts
        );
        assert!(
            stats.total_tactic_failures > 0,
            "Should have at least one failure (bad_tactic), got {}",
            stats.total_tactic_failures
        );
        assert!(
            stats.nodes_terminal > 0,
            "Should have found at least one terminal node, got {}",
            stats.nodes_terminal
        );
        // generate_time and lean_time should be >= 0 (they are u64, so always true,
        // but we check they were touched by verifying tactic_attempts)
    }

    #[tokio::test]
    async fn test_search_result_proof_tactics() {
        let mut env = MockEnvironment::new();
        env.add_response(
            0,
            "intro h",
            TacticResult::Success {
                state_id: 1,
                goals: vec![Goal::parse(0, "h : P\n⊢ P")],
            },
        );
        env.add_response(1, "exact h", TacticResult::ProofComplete { state_id: 2 });

        let mut policy = MockPolicy::new();
        policy.add_response("⊢ P → P", vec![make_tactic("intro h", -0.2)]);
        policy.add_response("h : P\n⊢ P", vec![make_tactic("exact h", -0.1)]);

        let engine = SearchEngine::new(SearchConfig::default());
        let result = engine
            .search_one(&env, &policy, None, "p_implies_p", "P → P")
            .await
            .unwrap();

        assert!(result.proved);
        assert_eq!(result.proof_tactics, vec!["intro h", "exact h"]);
    }
}
