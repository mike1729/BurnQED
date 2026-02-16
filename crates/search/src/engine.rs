//! Best-first search engine with priority queue, node expansion, and scoring.

use std::collections::{BinaryHeap, HashSet};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
/// Async trait — generation involves HTTP calls to an inference server.
#[async_trait]
pub trait PolicyProvider: Send + Sync {
    /// Generate up to `n` candidate tactics for the given proof state text.
    async fn generate_candidates(
        &self,
        proof_state: &str,
        n: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError>;

    /// Generate candidates for multiple states. Default: sequential calls.
    async fn generate_candidates_batch(
        &self,
        states: &[String],
        n: usize,
    ) -> Result<Vec<Vec<GeneratedTactic>>, SearchError> {
        let mut results = Vec::with_capacity(states.len());
        for s in states {
            results.push(self.generate_candidates(s, n).await?);
        }
        Ok(results)
    }

    /// Return (hits, misses) cache counters, if this provider has a cache.
    fn cache_stats(&self) -> Option<(u32, u32)> {
        None
    }
}

/// Value function that scores proof states (e.g., EBM energy).
///
/// Sync trait — matches burn's synchronous inference API.
pub trait ValueScorer: Send + Sync {
    /// Score a proof state. Convention: higher = more provable.
    fn score(&self, proof_state: &str) -> Result<f64, SearchError>;

    /// Score multiple proof states in one call. Default: sequential fallback.
    fn score_batch(&self, proof_states: &[&str]) -> Result<Vec<f64>, SearchError> {
        proof_states.iter().map(|s| self.score(s)).collect()
    }
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
        let start_time = Instant::now();
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

        // Visited states for loop detection
        let mut visited_states: HashSet<String> = HashSet::new();
        visited_states.insert(root_pp.clone());

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
            let ebm_start = Instant::now();
            let score = scorer.score(&arena[0].state_pp)?;
            let ebm_us = ebm_start.elapsed().as_micros() as u64;
            stats.total_ebm_time_ms += ebm_us / 1000;
            stats.ebm_latencies_us.push(ebm_us);
            stats.ebm_score_calls += 1;
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
        loop {
            if frontier.is_empty() || nodes_expanded >= self.config.max_nodes {
                break;
            }

            let timeout_secs = self.config.timeout_per_theorem;
            if timeout_secs > 0 && start_time.elapsed().as_secs() >= timeout_secs {
                tracing::debug!(
                    theorem = theorem_name,
                    elapsed_s = start_time.elapsed().as_secs(),
                    "Search timed out"
                );
                break;
            }

            // Pop the best node from the frontier
            let sn = match frontier.pop() {
                Some(sn) => sn,
                None => break,
            };

            let node_idx = sn.node_index;
            let node_state_id = arena[node_idx].state_id;
            let node_depth = arena[node_idx].depth;

            // Skip nodes at max depth
            if node_depth >= self.config.max_depth {
                nodes_expanded += 1;
                stats.nodes_pruned += 1;
                continue;
            }

            tracing::trace!(
                node = node_idx,
                state_id = node_state_id,
                depth = node_depth,
                score = %sn.score,
                "Expanding node"
            );

            let goals_text = arena[node_idx].goals_as_text();
            let gen_start = Instant::now();
            let mut candidates = policy.generate_candidates(
                &goals_text,
                self.config.num_candidates,
            ).await?;
            let gen_us = gen_start.elapsed().as_micros() as u64;
            stats.total_generate_time_ms += gen_us / 1000;
            stats.gen_latencies_us.push(gen_us);

            // Collect children needing EBM scoring for deferred batch scoring.
            let mut pending_scores: Vec<(usize, String)> = Vec::new();

            {

                // Track LLM candidate count before probe injection
                let llm_count = candidates.len();
                stats.candidates_per_expansion.push(llm_count);

                // Inject fallback tactics when LLM returns nothing
                if candidates.is_empty() && !self.config.fallback_tactics.is_empty() {
                    tracing::debug!(
                        node = node_idx,
                        "LLM returned no candidates, using fallback tactics"
                    );
                    candidates = self
                        .config
                        .fallback_tactics
                        .iter()
                        .map(|t| GeneratedTactic {
                            text: t.clone(),
                            raw_text: t.clone(),
                            log_prob: -100.0,
                            tokens: vec![],
                        })
                        .collect();
                }

                // Append probe tactics (deduped against LLM candidates)
                let num_probes = inject_probes(&mut candidates, &self.config.probe_tactics);

                for (cand_idx, candidate) in candidates.iter().enumerate() {
                    let is_probe = cand_idx >= candidates.len() - num_probes;
                    if is_probe {
                        stats.probe_attempts += 1;
                    }

                    stats.total_tactic_attempts += 1;
                    let tactic_start = Instant::now();
                    let result = runner
                        .apply_tactic(node_state_id, None, &candidate.text)
                        .await?;
                    let tactic_us = tactic_start.elapsed().as_micros() as u64;
                    stats.total_lean_time_ms += tactic_us / 1000;
                    stats.lean_latencies_us.push(tactic_us);
                    if is_probe {
                        stats.total_probe_lean_time_ms += tactic_us / 1000;
                    } else {
                        stats.total_llm_lean_time_ms += tactic_us / 1000;
                    }

                    match result {
                        TacticResult::Success { state_id, goals } => {
                            let child_depth = node_depth + 1;

                            let state_pp = goals
                                .iter()
                                .map(|g| g.raw.as_str())
                                .collect::<Vec<_>>()
                                .join("\n\n");

                            // Loop detection: skip states we've already visited
                            if visited_states.contains(&state_pp) {
                                let looped = SearchNode {
                                    state_id,
                                    state_pp,
                                    goals,
                                    parent: Some(node_idx),
                                    tactic_applied: candidate.text.clone(),
                                    depth: child_depth,
                                    llm_log_prob: candidate.log_prob,
                                    ebm_score: 0.0,
                                    is_terminal: false,
                                };
                                arena.push(looped);
                                stats.loops_detected += 1;
                                continue;
                            }
                            visited_states.insert(state_pp.clone());

                            if is_probe {
                                stats.probe_successes += 1;
                            }

                            if child_depth > max_depth_reached {
                                max_depth_reached = child_depth;
                            }

                            // Defer EBM scoring: push child with ebm_score=0.0
                            let child_idx = arena.len();
                            let child = SearchNode {
                                state_id,
                                state_pp: state_pp.clone(),
                                goals,
                                parent: Some(node_idx),
                                tactic_applied: candidate.text.clone(),
                                depth: child_depth,
                                llm_log_prob: candidate.log_prob,
                                ebm_score: 0.0,
                                is_terminal: false,
                            };
                            arena.push(child);

                            if scorer.is_some() {
                                pending_scores.push((child_idx, state_pp));
                            } else {
                                // No scorer: push to frontier immediately
                                let child_score =
                                    arena[child_idx].combined_score(self.config.alpha, self.config.beta);
                                frontier.push(ScoredNode {
                                    node_index: child_idx,
                                    score: OrderedFloat(child_score),
                                });
                            }
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

                            // Sibling harvest: expand proof path ancestors for hard negatives
                            if self.config.harvest_siblings {
                                let harvest_start = Instant::now();
                                harvest_siblings(
                                    &mut arena,
                                    terminal_idx,
                                    &self.config,
                                    &mut *runner,
                                    policy,
                                    scorer,
                                    &mut stats,
                                )
                                .await?;
                                stats.total_harvest_time_ms +=
                                    harvest_start.elapsed().as_millis() as u64;
                            }

                            // Collect cache stats from policy provider
                            if let Some((hits, misses)) = policy.cache_stats() {
                                stats.cache_hits = hits;
                                stats.cache_misses = misses;
                            }

                            let proof_tactics = extract_tactic_sequence(&arena, terminal_idx);
                            let wall_time_ms = start_time.elapsed().as_millis() as u64;
                            let total_states = arena.len() as u32;
                            let records = build_trajectory_records(&arena, theorem_name);

                            stats.nodes_expanded = nodes_expanded;

                            tracing::debug!(
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
                            tracing::trace!(
                                tactic = candidate.text,
                                error = message,
                                "Tactic failed"
                            );
                        }
                    }
                }

            }

            nodes_expanded += 1;

            // Batch-score all deferred children from this expansion
            if let Some(scorer) = scorer {
                if !pending_scores.is_empty() {
                    let states: Vec<&str> = pending_scores.iter().map(|(_, s)| s.as_str()).collect();
                    let ebm_start = Instant::now();
                    let scores = scorer.score_batch(&states).unwrap_or_else(|e| {
                        tracing::warn!(error = %e, "Batch scoring failed during expansion, using 0.0 defaults");
                        vec![0.0; states.len()]
                    });
                    let ebm_us = ebm_start.elapsed().as_micros() as u64;
                    stats.total_ebm_time_ms += ebm_us / 1000;
                    stats.ebm_latencies_us.push(ebm_us);
                    stats.ebm_score_calls += scores.len() as u32;

                    for ((idx, _), score) in pending_scores.iter().zip(scores) {
                        arena[*idx].ebm_score = score;
                        let child_score =
                            arena[*idx].combined_score(self.config.alpha, self.config.beta);
                        frontier.push(ScoredNode {
                            node_index: *idx,
                            score: OrderedFloat(child_score),
                        });
                    }
                    stats.peak_frontier_size =
                        stats.peak_frontier_size.max(frontier.len());
                }
            }
        }

        // Collect cache stats from policy provider
        if let Some((hits, misses)) = policy.cache_stats() {
            stats.cache_hits = hits;
            stats.cache_misses = misses;
        }

        // Search exhausted without finding a proof
        let wall_time_ms = start_time.elapsed().as_millis() as u64;
        let total_states = arena.len() as u32;
        let records = build_trajectory_records(&arena, theorem_name);

        stats.nodes_expanded = nodes_expanded;

        tracing::debug!(
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

/// Inject probe tactics into candidates, deduped against existing LLM candidates.
/// Returns the number of probes actually appended.
fn inject_probes(candidates: &mut Vec<GeneratedTactic>, probes: &[String]) -> usize {
    if probes.is_empty() {
        return 0;
    }
    let llm_texts: HashSet<String> = candidates
        .iter()
        .map(|c| c.text.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    let mut count = 0;
    for probe in probes {
        let norm = probe.split_whitespace().collect::<Vec<_>>().join(" ");
        if !llm_texts.contains(&norm) {
            candidates.push(GeneratedTactic {
                text: probe.clone(),
                raw_text: probe.clone(),
                log_prob: -10.0,
                tokens: vec![],
            });
            count += 1;
        }
    }
    count
}

/// After finding a proof, expand sibling states on the proof path for hard negatives.
async fn harvest_siblings(
    arena: &mut Vec<SearchNode>,
    terminal_idx: usize,
    config: &SearchConfig,
    runner: &mut (dyn TacticRunner + Send),
    policy: &dyn PolicyProvider,
    scorer: Option<&dyn ValueScorer>,
    stats: &mut SearchStats,
) -> Result<(), SearchError> {
    // Collect ancestor node indices on the proof path
    let proof_path: HashSet<usize> = {
        let mut path = HashSet::new();
        let mut idx = terminal_idx;
        while let Some(parent) = arena[idx].parent {
            path.insert(parent);
            idx = parent;
        }
        path.insert(0); // root
        path
    };

    // Collect the tactic used from each ancestor to its proof-path child
    let proof_tactics_used: HashSet<(usize, String)> = {
        let mut set = HashSet::new();
        let mut idx = terminal_idx;
        while let Some(parent) = arena[idx].parent {
            set.insert((parent, arena[idx].tactic_applied.clone()));
            idx = parent;
        }
        set
    };

    // Collect ancestor data before mutating arena
    let ancestors: Vec<(usize, u64, u32, String)> = proof_path
        .iter()
        .filter(|&&idx| arena[idx].depth < config.max_depth)
        .map(|&idx| {
            (
                idx,
                arena[idx].state_id,
                arena[idx].depth,
                arena[idx].goals_as_text(),
            )
        })
        .collect();

    // Batch-generate candidates for all ancestors in one call
    let prompts: Vec<String> = ancestors.iter().map(|(_, _, _, g)| g.clone()).collect();
    let batch_candidates = policy
        .generate_candidates_batch(&prompts, config.num_candidates)
        .await?;

    // Collect sibling nodes needing EBM scoring for deferred batch scoring
    let mut pending_scores: Vec<(usize, String)> = Vec::new();

    for (i, (ancestor_idx, anc_state_id, anc_depth, _)) in ancestors.into_iter().enumerate() {
        let mut siblings = batch_candidates[i].clone();
        inject_probes(&mut siblings, &config.probe_tactics);

        for sib in &siblings {
            // Skip the tactic that's already on the proof path from this ancestor
            if proof_tactics_used.contains(&(ancestor_idx, sib.text.clone())) {
                continue;
            }

            let result = runner
                .apply_tactic(anc_state_id, None, &sib.text)
                .await?;
            match result {
                TacticResult::Success { state_id, goals } => {
                    let spp = goals
                        .iter()
                        .map(|g| g.raw.as_str())
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    let child_idx = arena.len();
                    arena.push(SearchNode {
                        state_id,
                        state_pp: spp.clone(),
                        goals,
                        parent: Some(ancestor_idx),
                        tactic_applied: sib.text.clone(),
                        depth: anc_depth + 1,
                        llm_log_prob: sib.log_prob,
                        ebm_score: 0.0,
                        is_terminal: false,
                    });
                    stats.sibling_states_mined += 1;
                    if scorer.is_some() {
                        pending_scores.push((child_idx, spp));
                    }
                }
                TacticResult::ProofComplete { state_id } => {
                    // Alternative proof — record as non-terminal
                    arena.push(SearchNode {
                        state_id,
                        state_pp: String::new(),
                        goals: vec![],
                        parent: Some(ancestor_idx),
                        tactic_applied: sib.text.clone(),
                        depth: anc_depth + 1,
                        llm_log_prob: sib.log_prob,
                        ebm_score: 0.0,
                        is_terminal: false,
                    });
                    stats.sibling_states_mined += 1;
                }
                TacticResult::Failed { .. } => {}
            }
        }
    }

    // Batch-score all deferred sibling states
    if let Some(scorer) = scorer {
        if !pending_scores.is_empty() {
            let states: Vec<&str> = pending_scores.iter().map(|(_, s)| s.as_str()).collect();
            let scores = scorer.score_batch(&states).unwrap_or_else(|e| {
                tracing::warn!(error = %e, "Batch scoring failed in harvest_siblings, using 0.0");
                vec![0.0; states.len()]
            });
            for ((idx, _), score) in pending_scores.iter().zip(scores) {
                arena[*idx].ebm_score = score;
            }
        }
    }

    Ok(())
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

    #[tokio::test]
    async fn test_search_with_probe_tactics() {
        // LLM returns no candidates, but probe "trivial" solves the goal.
        let mut env = MockEnvironment::new();
        env.add_response(0, "trivial", TacticResult::ProofComplete { state_id: 1 });

        let policy = MockPolicy::new(); // returns empty for all states

        let config = SearchConfig {
            probe_tactics: vec!["trivial".to_string()],
            fallback_tactics: vec![],
            ..SearchConfig::default()
        };
        let engine = SearchEngine::new(config);
        let result = engine
            .search_one(&env, &policy, None, "probe_test", "True")
            .await
            .unwrap();

        assert!(result.proved, "Probe tactic should find the proof");
        assert_eq!(result.proof_tactics, vec!["trivial"]);
        assert!(result.stats.probe_attempts > 0);
    }

    #[tokio::test]
    async fn test_loop_detection() {
        // Tactic "loop" leads back to the same state → should be detected as a loop
        let mut env = MockEnvironment::new();
        env.add_response(
            0,
            "loop",
            TacticResult::Success {
                state_id: 1,
                goals: vec![Goal::parse(0, "⊢ ∀ x, x = x")],
            },
        );

        let mut policy = MockPolicy::new();
        policy.add_response("⊢ ∀ x, x = x", vec![make_tactic("loop", -0.5)]);

        let config = SearchConfig {
            max_nodes: 5,
            probe_tactics: vec![],
            ..SearchConfig::default()
        };
        let engine = SearchEngine::new(config);
        let result = engine
            .search_one(&env, &policy, None, "loop_test", "∀ x, x = x")
            .await
            .unwrap();

        assert!(!result.proved);
        // The loop node should be in the arena but not lead to infinite expansion
        assert!(result.stats.loops_detected > 0, "Should detect at least one loop");
        // Arena should have root + first success + looped node
        assert!(result.all_records.len() >= 2);
    }

    #[tokio::test]
    async fn test_harvest_siblings() {
        // Two-step proof, harvest_siblings = true → extra sibling nodes
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
        // Sibling from root: "simp" produces a different state
        env.add_response(
            0,
            "simp",
            TacticResult::Success {
                state_id: 3,
                goals: vec![Goal::parse(0, "⊢ simp_result")],
            },
        );

        let mut policy = MockPolicy::new();
        policy.add_response(
            "⊢ ∀ (n : Nat), n = n",
            vec![make_tactic("intro n", -0.3), make_tactic("simp", -0.5)],
        );
        policy.add_response("n : Nat\n⊢ n = n", vec![make_tactic("rfl", -0.1)]);

        let config = SearchConfig {
            harvest_siblings: true,
            probe_tactics: vec![],
            ..SearchConfig::default()
        };
        let engine = SearchEngine::new(config);
        let result = engine
            .search_one(&env, &policy, None, "harvest_test", "∀ (n : Nat), n = n")
            .await
            .unwrap();

        assert!(result.proved);
        assert_eq!(result.proof_tactics, vec!["intro n", "rfl"]);
        // Should have mined siblings from the proof path ancestors
        assert!(
            result.stats.sibling_states_mined > 0,
            "Should mine at least one sibling state"
        );
        // Arena should have more than just the 3 proof nodes
        assert!(
            result.all_records.len() > 3,
            "Should have extra sibling nodes: got {}",
            result.all_records.len()
        );
    }

    #[test]
    fn test_inject_probes_dedup() {
        let mut candidates = vec![make_tactic("simp", -0.5)];
        let probes = vec!["simp".to_string(), "ring".to_string(), "omega".to_string()];
        let count = inject_probes(&mut candidates, &probes);
        // "simp" should be deduped, "ring" and "omega" added
        assert_eq!(count, 2);
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].text, "simp");
        assert_eq!(candidates[1].text, "ring");
        assert_eq!(candidates[2].text, "omega");
    }

    #[test]
    fn test_inject_probes_empty() {
        let mut candidates = vec![make_tactic("intro", -0.5)];
        let probes: Vec<String> = vec![];
        let count = inject_probes(&mut candidates, &probes);
        assert_eq!(count, 0);
        assert_eq!(candidates.len(), 1);
    }
}
