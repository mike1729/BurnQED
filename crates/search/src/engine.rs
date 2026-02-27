//! Hybrid whole-proof search engine with replay trie, Q-values, and UCB scoring.

use std::collections::{BinaryHeap, HashSet};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use ordered_float::OrderedFloat;

use lean_repl::{Goal, ProofState, TacticResult};
use policy::{extract_all_tactics_structured, GeneratedTactic};
use trajectory::{SearchResult, SearchStats, TrajectoryLabel, TrajectoryRecord};

use crate::config::SearchConfig;
use crate::node::{extract_tactic_sequence, ScoredNode, SearchNode};
use crate::trie::{normalize_tactic, ReplayTrie, TrieEntry};

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

/// LLM policy that generates whole-proof completions for a proof state.
///
/// Async trait — generation involves HTTP calls to an inference server.
#[async_trait]
pub trait PolicyProvider: Send + Sync {
    /// Generate N whole-proof completions for the given proof state.
    ///
    /// Returns the full raw text of each completion for trie-based replay.
    /// Each completion may contain multiple tactics that will be split by
    /// `extract_all_tactics_structured` during trie replay.
    async fn generate_whole_proofs(
        &self,
        proof_state: &str,
        n: usize,
        max_tokens: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError>;
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

    /// Search for a proof of a single theorem using hybrid whole-proof replay.
    ///
    /// Generates N whole proofs, splits into tactic sequences, replays through
    /// Lean building a trie (shared prefixes verified once), then adaptively
    /// expands the most promising unsolved leaf via UCB/EBM scoring.
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

        // Build root node
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
        let mut visited_states: HashSet<String> = HashSet::new();
        visited_states.insert(root_pp.clone());
        let mut trie = ReplayTrie::new();

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
                failure_reason: "proved".to_string(),
            });
        }

        // Score root with EBM if available
        if let Some(scorer) = scorer {
            if self.config.should_skip_ebm(0) {
                stats.ebm_skipped_by_depth += 1;
                arena[0].ebm_score = -1e6;
            } else {
                let ebm_start = Instant::now();
                match scorer.score(&arena[0].state_pp) {
                    Ok(score) => arena[0].ebm_score = score,
                    Err(e) => {
                        tracing::warn!(theorem = theorem_name, error = %e, "Root EBM scoring failed");
                        arena[0].ebm_score = 0.0;
                    }
                }
                let ebm_us = ebm_start.elapsed().as_micros() as u64;
                stats.total_ebm_time_ms += ebm_us / 1000;
                stats.ebm_latencies_us.push(ebm_us);
                stats.ebm_score_calls += 1;
            }
        }

        // Persistent frontier — persists across rounds so nodes from earlier
        // rounds compete with newly-scored nodes from later rounds.
        let mut frontier = BinaryHeap::new();
        let mut expanded: HashSet<usize> = HashSet::new();
        {
            let ebm = if scorer.is_some() && !self.config.should_skip_ebm(0) {
                Some(arena[0].ebm_score)
            } else {
                None
            };
            let root_score = trie.frontier_score(0, ebm, None);
            frontier.push(ScoredNode {
                node_index: 0,
                score: OrderedFloat(root_score),
            });
        }

        let mut total_proofs: u32 = 0;
        let mut round: u32 = 0;
        let mut max_depth_reached: u32 = 0;
        let mut nodes_expanded: u32 = 0;

        // Hybrid search loop
        let exit_reason;
        'outer: loop {
            // Check termination conditions
            let timeout_secs = self.config.timeout_per_theorem;
            if timeout_secs > 0 && start_time.elapsed().as_secs() >= timeout_secs {
                exit_reason = "timeout";
                tracing::debug!(theorem = theorem_name, "Search timed out");
                break;
            }
            if total_proofs >= self.config.hybrid_budget {
                exit_reason = "budget_exhausted";
                break;
            }
            if nodes_expanded >= self.config.max_nodes {
                exit_reason = "budget_exhausted";
                break;
            }
            if round >= self.config.hybrid_max_rounds {
                exit_reason = "max_rounds";
                break;
            }

            // Pop best node from persistent frontier (lazy deletion for stale entries)
            let leaf_idx = loop {
                match frontier.pop() {
                    None => {
                        exit_reason = "frontier_exhausted";
                        break 'outer;
                    }
                    Some(sn) => {
                        let idx = sn.node_index;
                        // Skip already-expanded, terminal, or depth-limited nodes
                        if expanded.contains(&idx)
                            || arena[idx].is_terminal
                            || arena[idx].depth >= self.config.max_depth
                        {
                            continue;
                        }
                        break idx;
                    }
                }
            };
            expanded.insert(leaf_idx);

            if frontier.len() + 1 > stats.peak_frontier_size {
                stats.peak_frontier_size = frontier.len() + 1;
            }

            // Track new nodes created during this expansion (probes + replay)
            let arena_before = arena.len();

            // Try probe tactics at this leaf BEFORE generating whole proofs
            if let Some(terminal_idx) = try_probes_at_node(
                leaf_idx,
                &mut arena,
                &mut *runner,
                &mut trie,
                &mut visited_states,
                &self.config,
                &mut stats,
                &mut max_depth_reached,
            )
            .await?
            {
                // Probe found a proof!
                trie.record_proof_success(terminal_idx, &arena);

                stats.nodes_expanded = nodes_expanded;
                stats.hybrid_rounds = round + 1;
                stats.trie_cache_hits = trie.cache_hits;
                let proof_tactics = extract_tactic_sequence(&arena, terminal_idx);
                let wall_time_ms = start_time.elapsed().as_millis() as u64;
                let records = build_trajectory_records(&arena, theorem_name);

                return Ok(SearchResult {
                    theorem_name: theorem_name.to_string(),
                    proved: true,
                    proof_tactics,
                    nodes_expanded,
                    total_states: arena.len() as u32,
                    max_depth_reached,
                    wall_time_ms,
                    all_records: records,
                    stats,
                    failure_reason: "proved".to_string(),
                });
            }

            let leaf_state = arena[leaf_idx].goals_as_text();

            // Generate whole proofs from this leaf
            let n = if round == 0 {
                self.config.hybrid_num_proofs
            } else {
                self.config.hybrid_expand_proofs
            };

            let gen_start = Instant::now();
            let raw_proofs = policy
                .generate_whole_proofs(&leaf_state, n, self.config.hybrid_max_tokens)
                .await?;
            let gen_us = gen_start.elapsed().as_micros() as u64;
            stats.total_generate_time_ms += gen_us / 1000;
            stats.gen_latencies_us.push(gen_us);
            total_proofs += n as u32;
            stats.hybrid_proofs_generated += n as u32;

            tracing::debug!(
                round,
                leaf = leaf_idx,
                depth = arena[leaf_idx].depth,
                n_proofs = raw_proofs.len(),
                "Hybrid round: generated whole proofs"
            );

            // Replay each proof through the trie
            for proof in &raw_proofs {
                let tactics = extract_all_tactics_structured(&proof.text);
                if tactics.is_empty() {
                    continue;
                }

                let mut current_idx = leaf_idx;

                for tactic in &tactics {
                    trie.record_visit(current_idx);

                    // Check trie cache first
                    let normalized = normalize_tactic(tactic);
                    let cached = trie.lookup(current_idx, &normalized).cloned();
                    if let Some(entry) = cached {
                        match entry {
                            TrieEntry::Success(child_idx) => {
                                stats.trie_cache_hits += 1;
                                trie.cache_hits += 1;
                                current_idx = child_idx;
                                continue;
                            }
                            TrieEntry::Failed => {
                                break;
                            }
                        }
                    }

                    // Novel tactic → send to Lean
                    let state_id = arena[current_idx].state_id;
                    let node_depth = arena[current_idx].depth;

                    stats.total_tactic_attempts += 1;
                    let tactic_start = Instant::now();
                    let result = runner.apply_tactic(state_id, None, tactic).await?;
                    let tactic_us = tactic_start.elapsed().as_micros() as u64;
                    stats.total_lean_time_ms += tactic_us / 1000;
                    stats.total_llm_lean_time_ms += tactic_us / 1000;
                    stats.lean_latencies_us.push(tactic_us);

                    match result {
                        TacticResult::Success { state_id: child_state_id, goals } => {
                            let child_depth = node_depth + 1;
                            let state_pp = goals
                                .iter()
                                .map(|g| g.raw.as_str())
                                .collect::<Vec<_>>()
                                .join("\n\n");

                            // Loop detection
                            if visited_states.contains(&state_pp) {
                                stats.loops_detected += 1;
                                let looped_idx = arena.len();
                                arena.push(SearchNode {
                                    state_id: child_state_id,
                                    state_pp,
                                    goals,
                                    parent: Some(current_idx),
                                    tactic_applied: tactic.clone(),
                                    depth: child_depth,
                                    llm_log_prob: proof.log_prob,
                                    ebm_score: 0.0,
                                    is_terminal: false,
                                });
                                trie.insert_success(current_idx, tactic, looped_idx);
                                break;
                            }
                            visited_states.insert(state_pp.clone());

                            if child_depth > max_depth_reached {
                                max_depth_reached = child_depth;
                            }

                            let child_idx = arena.len();
                            arena.push(SearchNode {
                                state_id: child_state_id,
                                state_pp: state_pp.clone(),
                                goals,
                                parent: Some(current_idx),
                                tactic_applied: tactic.clone(),
                                depth: child_depth,
                                llm_log_prob: proof.log_prob,
                                ebm_score: 0.0,
                                is_terminal: false,
                            });
                            trie.insert_success(current_idx, tactic, child_idx);

                            // Try probe tactics at this new node
                            if let Some(terminal) = try_probes_at_node(
                                child_idx,
                                &mut arena,
                                &mut *runner,
                                &mut trie,
                                &mut visited_states,
                                &self.config,
                                &mut stats,
                                &mut max_depth_reached,
                            )
                            .await?
                            {
                                // Probe found a proof!
                                trie.record_proof_success(terminal, &arena);

                                stats.nodes_expanded = nodes_expanded;
                                stats.hybrid_rounds = round + 1;
                                stats.trie_cache_hits = trie.cache_hits;
                                let proof_tactics = extract_tactic_sequence(&arena, terminal);
                                let wall_time_ms = start_time.elapsed().as_millis() as u64;
                                let records = build_trajectory_records(&arena, theorem_name);

                                return Ok(SearchResult {
                                    theorem_name: theorem_name.to_string(),
                                    proved: true,
                                    proof_tactics,
                                    nodes_expanded,
                                    total_states: arena.len() as u32,
                                    max_depth_reached,
                                    wall_time_ms,
                                    all_records: records,
                                    stats,
                                    failure_reason: "proved".to_string(),
                                });
                            }

                            current_idx = child_idx;
                        }
                        TacticResult::ProofComplete { state_id: terminal_state_id } => {
                            let child_depth = arena[current_idx].depth + 1;
                            stats.nodes_terminal += 1;

                            if child_depth > max_depth_reached {
                                max_depth_reached = child_depth;
                            }

                            let terminal_idx = arena.len();
                            arena.push(SearchNode {
                                state_id: terminal_state_id,
                                state_pp: String::new(),
                                goals: vec![],
                                parent: Some(current_idx),
                                tactic_applied: tactic.clone(),
                                depth: child_depth,
                                llm_log_prob: proof.log_prob,
                                ebm_score: 0.0,
                                is_terminal: true,
                            });
                            trie.insert_success(current_idx, tactic, terminal_idx);
                            trie.record_proof_success(terminal_idx, &arena);

                            stats.nodes_expanded = nodes_expanded;
                            stats.hybrid_rounds = round + 1;
                            stats.trie_cache_hits = trie.cache_hits;
                            let proof_tactics = extract_tactic_sequence(&arena, terminal_idx);
                            let wall_time_ms = start_time.elapsed().as_millis() as u64;
                            let records = build_trajectory_records(&arena, theorem_name);

                            tracing::debug!(
                                theorem = theorem_name,
                                steps = proof_tactics.len(),
                                rounds = round + 1,
                                total_proofs,
                                trie_hits = trie.cache_hits,
                                time_ms = wall_time_ms,
                                "Proof found via hybrid search"
                            );

                            return Ok(SearchResult {
                                theorem_name: theorem_name.to_string(),
                                proved: true,
                                proof_tactics,
                                nodes_expanded,
                                total_states: arena.len() as u32,
                                max_depth_reached,
                                wall_time_ms,
                                all_records: records,
                                stats,
                                failure_reason: "proved".to_string(),
                            });
                        }
                        TacticResult::Failed { message } => {
                            stats.total_tactic_failures += 1;
                            trie.insert_failure(current_idx, tactic);
                            tracing::trace!(
                                tactic = tactic.as_str(),
                                error = message,
                                "Tactic failed"
                            );
                            break;
                        }
                    }
                }
            }

            // Batch-score ALL new nodes from this expansion (probes + replay),
            // then push them to the persistent frontier immediately.
            let new_nodes: Vec<usize> = (arena_before..arena.len())
                .filter(|&i| !arena[i].is_terminal)
                .collect();

            if let Some(scorer) = scorer {
                let pending: Vec<(usize, String)> = new_nodes
                    .iter()
                    .filter(|&&i| !self.config.should_skip_ebm(arena[i].depth))
                    .map(|&i| (i, arena[i].state_pp.clone()))
                    .collect();

                if !pending.is_empty() {
                    let states: Vec<&str> = pending.iter().map(|(_, s)| s.as_str()).collect();
                    let ebm_start = Instant::now();
                    let scores = scorer.score_batch(&states).unwrap_or_else(|e| {
                        tracing::warn!(error = %e, "Batch scoring failed, using 0.0");
                        vec![0.0; states.len()]
                    });
                    let ebm_us = ebm_start.elapsed().as_micros() as u64;
                    stats.total_ebm_time_ms += ebm_us / 1000;
                    stats.ebm_latencies_us.push(ebm_us);
                    stats.ebm_score_calls += scores.len() as u32;

                    for ((idx, _), score) in pending.iter().zip(scores) {
                        arena[*idx].ebm_score = score;
                    }
                }

                // Mark skipped nodes
                for &i in &new_nodes {
                    if self.config.should_skip_ebm(arena[i].depth) {
                        stats.ebm_skipped_by_depth += 1;
                        arena[i].ebm_score = -1e6;
                    }
                }
            }

            // Push scored new nodes to the persistent frontier
            for &idx in &new_nodes {
                let node = &arena[idx];
                if node.depth < self.config.max_depth {
                    let ebm = if scorer.is_some() && !self.config.should_skip_ebm(node.depth) {
                        Some(node.ebm_score)
                    } else {
                        None
                    };
                    let score = trie.frontier_score(idx, ebm, node.parent);
                    frontier.push(ScoredNode {
                        node_index: idx,
                        score: OrderedFloat(score),
                    });
                }
            }

            nodes_expanded += 1;
            round += 1;
        }

        stats.nodes_expanded = nodes_expanded;
        stats.hybrid_rounds = round;
        stats.trie_cache_hits = trie.cache_hits;

        let wall_time_ms = start_time.elapsed().as_millis() as u64;
        let records = build_trajectory_records(&arena, theorem_name);

        tracing::debug!(
            theorem = theorem_name,
            rounds = round,
            total_proofs,
            trie_hits = trie.cache_hits,
            arena_size = arena.len(),
            time_ms = wall_time_ms,
            "Hybrid search exhausted without proof"
        );

        Ok(SearchResult {
            theorem_name: theorem_name.to_string(),
            proved: false,
            proof_tactics: vec![],
            nodes_expanded,
            total_states: arena.len() as u32,
            max_depth_reached,
            wall_time_ms,
            all_records: records,
            stats,
            failure_reason: exit_reason.to_string(),
        })
    }
}

/// Try probe tactics at a new node. Returns Some(terminal_idx) if a probe closes the proof.
async fn try_probes_at_node(
    node_idx: usize,
    arena: &mut Vec<SearchNode>,
    runner: &mut (dyn TacticRunner + Send),
    trie: &mut ReplayTrie,
    visited_states: &mut HashSet<String>,
    config: &SearchConfig,
    stats: &mut SearchStats,
    max_depth_reached: &mut u32,
) -> Result<Option<usize>, SearchError> {
    if config.probe_tactics.is_empty() {
        return Ok(None);
    }

    let state_id = arena[node_idx].state_id;
    let node_depth = arena[node_idx].depth;

    for probe in &config.probe_tactics {
        stats.probe_attempts += 1;
        stats.total_tactic_attempts += 1;

        let tactic_start = Instant::now();
        let result = runner.apply_tactic(state_id, None, probe).await?;
        let tactic_us = tactic_start.elapsed().as_micros() as u64;
        stats.total_lean_time_ms += tactic_us / 1000;
        stats.total_probe_lean_time_ms += tactic_us / 1000;
        stats.lean_latencies_us.push(tactic_us);

        match result {
            TacticResult::ProofComplete { state_id: terminal_state_id } => {
                let child_depth = node_depth + 1;
                stats.nodes_terminal += 1;
                stats.probe_successes += 1;

                if child_depth > *max_depth_reached {
                    *max_depth_reached = child_depth;
                }

                let terminal_idx = arena.len();
                arena.push(SearchNode {
                    state_id: terminal_state_id,
                    state_pp: String::new(),
                    goals: vec![],
                    parent: Some(node_idx),
                    tactic_applied: probe.clone(),
                    depth: child_depth,
                    llm_log_prob: -10.0,
                    ebm_score: 0.0,
                    is_terminal: true,
                });
                trie.insert_success(node_idx, probe, terminal_idx);
                return Ok(Some(terminal_idx));
            }
            TacticResult::Success { state_id: child_state_id, goals } => {
                stats.probe_successes += 1;
                let child_depth = node_depth + 1;
                let state_pp = goals
                    .iter()
                    .map(|g| g.raw.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n");

                if !visited_states.contains(&state_pp) {
                    visited_states.insert(state_pp.clone());
                    if child_depth > *max_depth_reached {
                        *max_depth_reached = child_depth;
                    }
                    let child_idx = arena.len();
                    arena.push(SearchNode {
                        state_id: child_state_id,
                        state_pp,
                        goals,
                        parent: Some(node_idx),
                        tactic_applied: probe.clone(),
                        depth: child_depth,
                        llm_log_prob: -10.0,
                        ebm_score: 0.0,
                        is_terminal: false,
                    });
                    trie.insert_success(node_idx, probe, child_idx);
                }
            }
            TacticResult::Failed { .. } => {
                stats.total_tactic_failures += 1;
                trie.insert_failure(node_idx, probe);
            }
        }
    }

    Ok(None)
}

/// After finding a proof, expand sibling states on the proof path for hard negatives.
///
/// TODO: Reimplement using `generate_whole_proofs` + trie replay. The old
/// implementation used `generate_candidates_batch` which has been removed.
/// For now, the trie-based search naturally creates sibling branches during
/// whole-proof replay, providing similar training signal.
#[allow(dead_code)]
async fn harvest_siblings(
    _arena: &mut Vec<SearchNode>,
    _terminal_idx: usize,
    _config: &SearchConfig,
    _runner: &mut (dyn TacticRunner + Send),
    _policy: &dyn PolicyProvider,
    _scorer: Option<&dyn ValueScorer>,
    _stats: &mut SearchStats,
) -> Result<(), SearchError> {
    tracing::warn!("harvest_siblings is not yet implemented for hybrid search; skipping");
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
        // Policy returns no tactics → frontier exhausted after root
        let env = MockEnvironment::new();
        let policy = MockPolicy::new(); // returns empty for all states

        let engine = SearchEngine::new(SearchConfig::default());
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
        // Policy returns Ok(vec![]) for all states. The search loop should
        // exhaust the frontier (root expanded, no children created).
        let env = MockEnvironment::new();
        let policy = MockPolicy::new(); // returns empty vec for all states

        let config = SearchConfig {
            max_nodes: 3,
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
    async fn test_persistent_frontier_branches() {
        // Two branches from root, one leads to proof at depth 2.
        // Tests that the persistent frontier correctly holds both branches.
        let mut env = MockEnvironment::new();
        env.add_response(
            0,
            "branch_a",
            TacticResult::Success {
                state_id: 1,
                goals: vec![Goal::parse(0, "⊢ A")],
            },
        );
        env.add_response(
            0,
            "branch_b",
            TacticResult::Success {
                state_id: 2,
                goals: vec![Goal::parse(0, "⊢ B")],
            },
        );
        env.add_response(1, "solve_a", TacticResult::ProofComplete { state_id: 3 });

        let mut policy = MockPolicy::new();
        policy.add_response(
            "⊢ ∀ x, x = x",
            vec![make_tactic("branch_a", -0.3), make_tactic("branch_b", -0.4)],
        );
        policy.add_response("⊢ A", vec![make_tactic("solve_a", -0.1)]);
        policy.add_response("⊢ B", vec![]);

        let config = SearchConfig {
            probe_tactics: vec![],
            ..SearchConfig::default()
        };
        let engine = SearchEngine::new(config);
        let result = engine
            .search_one(&env, &policy, None, "branch_test", "∀ x, x = x")
            .await
            .unwrap();

        assert!(result.proved);
        assert_eq!(result.proof_tactics, vec!["branch_a", "solve_a"]);
    }
}
