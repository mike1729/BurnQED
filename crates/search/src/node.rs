use lean_repl::Goal;
use ordered_float::OrderedFloat;

/// A node in the search tree.
///
/// Nodes are stored in a flat arena (`Vec<SearchNode>`) and reference
/// parents by index. Each node corresponds to a Pantograph proof state
/// reached by applying a tactic to the parent state.
#[derive(Debug, Clone)]
pub struct SearchNode {
    /// Pantograph state ID (process-local, monotonically increasing).
    pub state_id: u64,
    /// Pretty-printed proof state text.
    pub state_pp: String,
    /// Parsed goals in this proof state.
    pub goals: Vec<Goal>,
    /// Index of the parent node in the arena, `None` for root.
    pub parent: Option<usize>,
    /// Tactic that was applied to reach this state (empty for root).
    pub tactic_applied: String,
    /// Number of tactic applications from root.
    pub depth: u32,
    /// Sum of log-probabilities from LLM policy.
    pub llm_log_prob: f64,
    /// Energy score from the EBM (lower = more provable).
    pub ebm_score: f64,
    /// True if the proof is complete (no remaining goals).
    pub is_terminal: bool,
}

impl SearchNode {
    /// Combined score: `alpha * llm_log_prob + beta * ebm_score`.
    pub fn combined_score(&self, alpha: f64, beta: f64) -> f64 {
        alpha * self.llm_log_prob + beta * self.ebm_score
    }

    /// Concatenate all goal raw strings with `"\n\n"` separators.
    pub fn goals_as_text(&self) -> String {
        self.goals
            .iter()
            .map(|g| g.raw.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// A search node with an associated score for priority queue ordering.
///
/// Used with `BinaryHeap` (max-heap). Higher score = higher priority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoredNode {
    /// Index into the `Vec<SearchNode>` arena.
    pub node_index: usize,
    /// Combined score for priority ordering.
    pub score: OrderedFloat<f64>,
}

impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

/// Trace the parent chain from a terminal node back to the root,
/// returning arena indices in root-to-terminal order.
pub fn extract_proof_path(tree: &[SearchNode], terminal_index: usize) -> Vec<usize> {
    let mut path = Vec::new();
    let mut current = Some(terminal_index);
    while let Some(idx) = current {
        path.push(idx);
        current = tree[idx].parent;
    }
    path.reverse();
    path
}

/// Extract the sequence of tactics applied along the proof path
/// from root to terminal. Skips the root node's empty tactic.
pub fn extract_tactic_sequence(tree: &[SearchNode], terminal_index: usize) -> Vec<String> {
    let path = extract_proof_path(tree, terminal_index);
    path.iter()
        .map(|&idx| &tree[idx].tactic_applied)
        .filter(|t| !t.is_empty())
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(
        state_id: u64,
        parent: Option<usize>,
        tactic: &str,
        depth: u32,
        llm_log_prob: f64,
        ebm_score: f64,
        goals_raw: Vec<&str>,
    ) -> SearchNode {
        let goals = goals_raw
            .iter()
            .enumerate()
            .map(|(i, raw)| Goal::parse(i, raw))
            .collect();
        SearchNode {
            state_id,
            state_pp: goals_raw.join("\n\n"),
            goals,
            parent,
            tactic_applied: tactic.to_string(),
            depth,
            llm_log_prob,
            ebm_score,
            is_terminal: goals_raw.is_empty(),
        }
    }

    #[test]
    fn test_combined_score() {
        let node = make_node(0, None, "", 0, -1.0, -2.0, vec!["⊢ True"]);
        let score = node.combined_score(0.5, 0.5);
        assert!((score - (-1.5)).abs() < 1e-9);
    }

    #[test]
    fn test_combined_score_asymmetric() {
        let node = make_node(0, None, "", 0, -1.0, -3.0, vec!["⊢ True"]);
        let score = node.combined_score(0.7, 0.3);
        // 0.7 * (-1.0) + 0.3 * (-3.0) = -0.7 + -0.9 = -1.6
        assert!((score - (-1.6)).abs() < 1e-9);
    }

    #[test]
    fn test_scored_node_ordering() {
        let a = ScoredNode {
            node_index: 0,
            score: OrderedFloat(1.0),
        };
        let b = ScoredNode {
            node_index: 1,
            score: OrderedFloat(2.0),
        };
        assert!(b > a); // Higher score = Greater
    }

    #[test]
    fn test_goals_as_text_single() {
        let node = make_node(0, None, "", 0, 0.0, 0.0, vec!["⊢ True"]);
        assert_eq!(node.goals_as_text(), "⊢ True");
    }

    #[test]
    fn test_goals_as_text_multi() {
        let node = make_node(0, None, "", 0, 0.0, 0.0, vec!["⊢ P", "⊢ Q"]);
        assert_eq!(node.goals_as_text(), "⊢ P\n\n⊢ Q");
    }

    #[test]
    fn test_extract_proof_path_linear() {
        let tree = vec![
            make_node(0, None, "", 0, 0.0, 0.0, vec!["⊢ P"]),
            make_node(1, Some(0), "intro", 1, -0.5, 0.0, vec!["⊢ Q"]),
            make_node(2, Some(1), "rfl", 2, -1.0, 0.0, vec![]),
        ];
        let path = extract_proof_path(&tree, 2);
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn test_extract_proof_path_branching() {
        // Tree: 0 -> 1, 0 -> 2, 1 -> 3 (terminal)
        let tree = vec![
            make_node(0, None, "", 0, 0.0, 0.0, vec!["⊢ P"]),
            make_node(1, Some(0), "tac_a", 1, -0.5, 0.0, vec!["⊢ Q"]),
            make_node(2, Some(0), "tac_b", 1, -0.3, 0.0, vec!["⊢ R"]),
            make_node(3, Some(1), "rfl", 2, -1.0, 0.0, vec![]),
        ];
        let path = extract_proof_path(&tree, 3);
        assert_eq!(path, vec![0, 1, 3]);
    }

    #[test]
    fn test_extract_tactic_sequence() {
        let tree = vec![
            make_node(0, None, "", 0, 0.0, 0.0, vec!["⊢ P"]),
            make_node(1, Some(0), "intro n", 1, -0.5, 0.0, vec!["n : Nat\n⊢ Q"]),
            make_node(2, Some(1), "rfl", 2, -1.0, 0.0, vec![]),
        ];
        let tactics = extract_tactic_sequence(&tree, 2);
        assert_eq!(tactics, vec!["intro n", "rfl"]);
    }

    #[test]
    fn test_root_node() {
        let tree = vec![make_node(0, None, "", 0, 0.0, 0.0, vec!["⊢ True"])];
        let path = extract_proof_path(&tree, 0);
        assert_eq!(path, vec![0]);
        let tactics = extract_tactic_sequence(&tree, 0);
        assert!(tactics.is_empty());
    }
}
