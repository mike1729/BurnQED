//! Replay trie for hybrid whole-proof search.
//!
//! Caches Lean tactic verification results keyed by `(parent_arena_idx, normalized_tactic)`.
//! Successful results map to child arena indices (free replay), failures are recorded
//! to avoid re-sending to Lean. Q-values (successes/visits) are backpropagated on
//! proof success for UCB-based frontier scoring.

use std::collections::{HashMap, HashSet};

use crate::node::SearchNode;

/// Result of a trie lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrieEntry {
    /// Tactic succeeded, child node is at the given arena index.
    Success(usize),
    /// Tactic was previously tried and failed.
    Failed,
}

/// Per-node visit/success statistics for Q-value computation.
#[derive(Debug, Clone, Default)]
pub struct TrieNodeStats {
    pub visits: u32,
    pub successes: u32,
}

/// Replay trie caching tactic results and tracking Q-values.
///
/// Keys are `(parent_arena_index, normalized_tactic_text)`. This allows
/// O(1) lookup when replaying a tactic sequence: if the same tactic from
/// the same parent was already verified, we skip the Lean call entirely.
pub struct ReplayTrie {
    /// Cached tactic results: (parent_idx, normalized_tactic) → entry.
    cache: HashMap<(usize, String), TrieEntry>,
    /// Per-node visit/success stats for Q-value tracking.
    stats: HashMap<usize, TrieNodeStats>,
    /// Total cache hit count.
    pub cache_hits: u32,
    /// Nodes whose stats were updated since last drain. Used to re-score
    /// existing frontier entries after replay rounds.
    updated_indices: HashSet<usize>,
}

impl ReplayTrie {
    /// Create a new empty trie.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            stats: HashMap::new(),
            cache_hits: 0,
            updated_indices: HashSet::new(),
        }
    }

    /// Look up a tactic result in the cache.
    pub fn lookup(&self, parent_idx: usize, tactic: &str) -> Option<&TrieEntry> {
        self.cache.get(&(parent_idx, normalize_tactic(tactic)))
    }

    /// Record a successful tactic application.
    pub fn insert_success(&mut self, parent_idx: usize, tactic: &str, child_idx: usize) {
        self.cache.insert(
            (parent_idx, normalize_tactic(tactic)),
            TrieEntry::Success(child_idx),
        );
    }

    /// Record a failed tactic application.
    pub fn insert_failure(&mut self, parent_idx: usize, tactic: &str) {
        self.cache.insert(
            (parent_idx, normalize_tactic(tactic)),
            TrieEntry::Failed,
        );
    }

    /// Increment visit count for a node.
    pub fn record_visit(&mut self, idx: usize) {
        self.stats.entry(idx).or_default().visits += 1;
        self.updated_indices.insert(idx);
    }

    /// After a proof is found at `terminal_idx`, walk the parent chain and
    /// increment success count for all ancestors (including root).
    pub fn record_proof_success(&mut self, terminal_idx: usize, arena: &[SearchNode]) {
        let mut current = Some(terminal_idx);
        while let Some(idx) = current {
            self.stats.entry(idx).or_default().successes += 1;
            self.updated_indices.insert(idx);
            current = arena[idx].parent;
        }
    }

    /// Q-value for a node: successes / visits. Returns 0.0 if unvisited.
    pub fn q_value(&self, idx: usize) -> f64 {
        match self.stats.get(&idx) {
            Some(s) if s.visits > 0 => s.successes as f64 / s.visits as f64,
            _ => 0.0,
        }
    }

    /// PUCT-style frontier score for a node.
    ///
    /// Blends a prior (EBM score or 0) with empirical Q-value, plus UCB exploration:
    ///
    /// ```text
    /// alpha_t = alpha / (1 + visits)          // trust prior when unvisited, shift to Q
    /// value   = alpha_t * prior + (1 - alpha_t) * q_value
    /// explore = c * sqrt(ln(parent_visits + 1) / (1 + visits))
    /// score   = value + explore
    /// ```
    ///
    /// This is AlphaZero's PUCT formula adapted for proof search.
    pub fn frontier_score(
        &self,
        idx: usize,
        ebm_score: Option<f64>,
        parent_idx: Option<usize>,
        alpha: f64,
        exploration_c: f64,
    ) -> f64 {
        let q = self.q_value(idx);
        let visits = self.stats.get(&idx).map_or(0, |s| s.visits);
        let parent_visits = parent_idx
            .and_then(|p| self.stats.get(&p))
            .map_or(0, |s| s.visits);

        let prior = ebm_score.unwrap_or(0.0);

        // Alpha blending: trust EBM prior when visits=0, shift to Q as visits grow
        let alpha_t = alpha / (1.0 + visits as f64);
        let value = alpha_t * prior + (1.0 - alpha_t) * q;

        // UCB exploration bonus
        let explore = exploration_c
            * ((parent_visits as f64 + 1.0).ln() / (1.0 + visits as f64)).sqrt();

        value + explore
    }

    /// Collect all non-terminal leaf nodes (nodes with no successful children in the trie).
    pub fn leaves(&self, arena: &[SearchNode]) -> Vec<usize> {
        let mut has_success_child: Vec<bool> = vec![false; arena.len()];
        for entry in self.cache.values() {
            if let TrieEntry::Success(child_idx) = entry {
                // The parent of child_idx has at least one success child
                if let Some(parent) = arena[*child_idx].parent {
                    has_success_child[parent] = true;
                }
            }
        }

        // Leaves: nodes that are in the arena, not terminal, and have no
        // successful children recorded in the trie. Also must have been
        // visited at least once (i.e., they're reachable).
        let mut leaves = Vec::new();
        for (idx, node) in arena.iter().enumerate() {
            if node.is_terminal {
                continue;
            }
            if has_success_child[idx] {
                continue;
            }
            // Must have goals (not a failed node placeholder)
            if node.goals.is_empty() && idx != 0 {
                continue;
            }
            leaves.push(idx);
        }
        leaves
    }

    /// Get stats for a node.
    pub fn get_stats(&self, idx: usize) -> Option<&TrieNodeStats> {
        self.stats.get(&idx)
    }

    /// Drain all node indices whose stats were updated since last drain.
    ///
    /// Drain the set of node indices whose trie stats changed since the last drain.
    ///
    /// Returns only indices strictly below `max_idx` — newly created nodes
    /// (idx >= max_idx) are excluded because they'll be pushed to the frontier
    /// separately when first discovered. Pre-existing nodes need re-scoring
    /// because their visit/success counts changed during replay.
    pub fn drain_updated_indices_below(&mut self, max_idx: usize) -> Vec<usize> {
        self.updated_indices
            .drain()
            .filter(|&idx| idx < max_idx)
            .collect()
    }

    /// Total number of cached entries.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

/// Normalize tactic whitespace for consistent cache keys.
pub fn normalize_tactic(tactic: &str) -> String {
    tactic.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::SearchNode;
    use lean_repl::Goal;

    fn make_node(
        parent: Option<usize>,
        tactic: &str,
        goals_raw: Vec<&str>,
    ) -> SearchNode {
        let goals: Vec<Goal> = goals_raw
            .iter()
            .enumerate()
            .map(|(i, raw)| Goal::parse(i, raw))
            .collect();
        SearchNode {
            state_id: 0,
            state_pp: goals_raw.join("\n\n"),
            goals,
            parent,
            tactic_applied: tactic.to_string(),
            depth: parent.map_or(0, |_| 1),
            llm_log_prob: 0.0,
            ebm_score: 0.0,
            is_terminal: goals_raw.is_empty(),
        }
    }

    #[test]
    fn test_normalize_tactic() {
        assert_eq!(normalize_tactic("intro  h"), "intro h");
        assert_eq!(normalize_tactic("  simp  [a,   b]  "), "simp [a, b]");
        assert_eq!(normalize_tactic("rfl"), "rfl");
    }

    #[test]
    fn test_trie_lookup_miss() {
        let trie = ReplayTrie::new();
        assert!(trie.lookup(0, "intro h").is_none());
    }

    #[test]
    fn test_trie_insert_and_lookup_success() {
        let mut trie = ReplayTrie::new();
        trie.insert_success(0, "intro h", 1);
        assert_eq!(trie.lookup(0, "intro h"), Some(&TrieEntry::Success(1)));
        // Normalized lookup
        assert_eq!(trie.lookup(0, "intro  h"), Some(&TrieEntry::Success(1)));
    }

    #[test]
    fn test_trie_insert_and_lookup_failure() {
        let mut trie = ReplayTrie::new();
        trie.insert_failure(0, "bad_tactic");
        assert_eq!(trie.lookup(0, "bad_tactic"), Some(&TrieEntry::Failed));
    }

    #[test]
    fn test_q_value_unvisited() {
        let trie = ReplayTrie::new();
        assert!((trie.q_value(0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_q_value_with_visits() {
        let mut trie = ReplayTrie::new();
        trie.record_visit(0);
        trie.record_visit(0);
        trie.record_visit(0);
        // No successes yet
        assert!((trie.q_value(0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_backprop_proof_success() {
        // Linear chain: root(0) → child(1) → terminal(2)
        let arena = vec![
            make_node(None, "", vec!["⊢ P"]),
            make_node(Some(0), "intro h", vec!["h : P\n⊢ Q"]),
            make_node(Some(1), "exact h", vec![]),
        ];

        let mut trie = ReplayTrie::new();
        trie.record_visit(0);
        trie.record_visit(1);
        trie.record_proof_success(2, &arena);

        // All ancestors should have 1 success
        assert_eq!(trie.stats[&0].successes, 1);
        assert_eq!(trie.stats[&1].successes, 1);
        assert_eq!(trie.stats[&2].successes, 1);

        // Q-values
        assert!((trie.q_value(0) - 1.0).abs() < 1e-9); // 1/1
        assert!((trie.q_value(1) - 1.0).abs() < 1e-9); // 1/1
    }

    #[test]
    fn test_frontier_score_puct() {
        let mut trie = ReplayTrie::new();
        // Parent visited 10 times, child visited 2 times, 1 success
        trie.stats.insert(0, TrieNodeStats { visits: 10, successes: 3 });
        trie.stats.insert(1, TrieNodeStats { visits: 2, successes: 1 });

        let alpha = 0.5;
        let c = 1.41;
        let score = trie.frontier_score(1, None, Some(0), alpha, c);
        let q = 0.5; // 1/2
        // alpha_t = 0.5 / (1 + 2) = 0.1667
        let alpha_t = alpha / (1.0 + 2.0);
        // prior = 0.0 (no EBM)
        let value = alpha_t * 0.0 + (1.0 - alpha_t) * q;
        let explore = c * ((10.0_f64 + 1.0).ln() / 3.0).sqrt();
        let expected = value + explore;
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn test_frontier_score_with_ebm() {
        let mut trie = ReplayTrie::new();
        trie.stats.insert(1, TrieNodeStats { visits: 4, successes: 2 });

        let alpha = 0.5;
        let c = 1.41;
        let score = trie.frontier_score(1, Some(0.8), Some(0), alpha, c);
        let q = 0.5; // 2/4
        // alpha_t = 0.5 / (1 + 4) = 0.1
        let alpha_t = alpha / 5.0;
        let value = alpha_t * 0.8 + (1.0 - alpha_t) * q;
        // parent_visits = 0 (no stats for 0)
        let explore = c * ((0.0_f64 + 1.0).ln() / 5.0).sqrt();
        let expected = value + explore;
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn test_frontier_score_unvisited_trusts_prior() {
        let trie = ReplayTrie::new();
        let alpha = 0.5;
        let c = 1.41;
        // Unvisited node with EBM score
        let score = trie.frontier_score(0, Some(0.8), None, alpha, c);
        // alpha_t = 0.5 / 1.0 = 0.5, q = 0.0
        let value = 0.5 * 0.8 + 0.5 * 0.0;
        let explore = c * ((1.0_f64).ln() / 1.0).sqrt(); // ln(1) = 0
        let expected = value + explore;
        assert!((score - expected).abs() < 1e-9);
        assert!(score > 0.0, "Unvisited node with good EBM should have positive score");
    }

    #[test]
    fn test_leaves_basic() {
        // root(0) → child(1), root(0) → child(2, terminal)
        let arena = vec![
            make_node(None, "", vec!["⊢ P"]),
            make_node(Some(0), "intro h", vec!["h : P\n⊢ Q"]),
            make_node(Some(0), "exact sorry", vec![]),
        ];

        let mut trie = ReplayTrie::new();
        trie.insert_success(0, "intro h", 1);
        trie.insert_success(0, "exact sorry", 2);

        let leaves = trie.leaves(&arena);
        // Node 1 has no children → leaf
        // Node 2 is terminal → not a leaf
        // Node 0 has success children → not a leaf
        assert_eq!(leaves, vec![1]);
    }

    #[test]
    fn test_leaves_root_only() {
        let arena = vec![make_node(None, "", vec!["⊢ P"])];
        let trie = ReplayTrie::new();
        let leaves = trie.leaves(&arena);
        assert_eq!(leaves, vec![0]);
    }

    #[test]
    fn test_cache_size() {
        let mut trie = ReplayTrie::new();
        assert_eq!(trie.cache_size(), 0);
        trie.insert_success(0, "intro", 1);
        trie.insert_failure(0, "bad");
        assert_eq!(trie.cache_size(), 2);
    }
}
