//! Data types for trajectory records, search results, and theorem tasks.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

/// Label for a trajectory state: on the proof path, dead end, or not yet known.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrajectoryLabel {
    Positive,
    Negative,
    Unknown,
}

impl fmt::Display for TrajectoryLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Positive => write!(f, "positive"),
            Self::Negative => write!(f, "negative"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

impl TrajectoryLabel {
    /// Parse from string. Returns Unknown for unrecognized values.
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "positive" => Self::Positive,
            "negative" => Self::Negative,
            _ => Self::Unknown,
        }
    }
}

/// A single state visited during proof search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryRecord {
    /// Name of the theorem being proved.
    pub theorem_name: String,
    /// Pantograph state ID for this node.
    pub state_id: u64,
    /// Pretty-printed proof state from Pantograph.
    pub state_pp: String,
    /// Tactic used to reach this state (empty for root).
    pub tactic_applied: String,
    /// State ID of the parent node. None for the root node.
    pub parent_state_id: Option<u64>,
    /// Label: positive (on proof path), negative (dead end), or unknown.
    pub label: TrajectoryLabel,
    /// Number of tactic applications from root to this state.
    pub depth_from_root: u32,
    /// Steps remaining to QED on proof path, -1 if unknown/off path.
    pub remaining_depth: i32,
    /// Log probability from the LLM policy for the tactic.
    pub llm_log_prob: f64,
    /// Energy score from the EBM (0.0 if EBM not used).
    pub ebm_score: f64,
    /// Whether this state has no remaining goals (proof complete).
    pub is_proof_complete: bool,
    /// Unix timestamp in milliseconds when this state was created.
    pub timestamp_ms: u64,
}

/// Detailed statistics from a single proof search.
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    /// Number of nodes removed from the frontier and expanded.
    pub nodes_expanded: u32,
    /// Number of child nodes skipped because they exceeded max depth.
    pub nodes_pruned: u32,
    /// Number of proof-complete nodes found during search.
    pub nodes_terminal: u32,
    /// Maximum size of the frontier priority queue during search.
    pub peak_frontier_size: usize,
    /// Total number of tactics sent to the Lean REPL.
    pub total_tactic_attempts: u32,
    /// Number of tactics that Lean rejected (TacticResult::Failed).
    pub total_tactic_failures: u32,
    /// Cumulative wall time in ms for `apply_tactic` calls.
    pub total_lean_time_ms: u64,
    /// Cumulative wall time in ms for `generate_candidates` calls.
    pub total_generate_time_ms: u64,
    /// Number of unique candidates returned per expansion (after dedup).
    pub candidates_per_expansion: Vec<usize>,
    /// Number of probe tactic attempts (subset of total_tactic_attempts).
    pub probe_attempts: u32,
    /// Number of probe tactics that produced valid states.
    pub probe_successes: u32,
    /// Number of states detected as loops (visited before).
    pub loops_detected: u32,
    /// Number of sibling states mined after proof found.
    pub sibling_states_mined: u32,

    // --- Timing breakdown (added for profiling) ---

    /// Cumulative wall time in ms for EBM scorer.score() calls.
    pub total_ebm_time_ms: u64,
    /// Cumulative wall time in ms for harvest_siblings post-proof overhead.
    pub total_harvest_time_ms: u64,
    /// Cumulative wall time in ms for Lean apply_tactic calls on probe tactics only.
    pub total_probe_lean_time_ms: u64,
    /// Cumulative wall time in ms for Lean apply_tactic calls on LLM candidates only.
    pub total_llm_lean_time_ms: u64,
    /// Number of EBM score() invocations.
    pub ebm_score_calls: u32,

    // --- Cache stats ---

    /// Number of policy cache hits.
    pub cache_hits: u32,
    /// Number of policy cache misses.
    pub cache_misses: u32,

    // --- Per-call latency vectors (microseconds, transient — not serialized) ---

    /// Per apply_tactic call latency in microseconds.
    pub lean_latencies_us: Vec<u64>,
    /// Per generate_candidates_batch call latency in microseconds.
    pub gen_latencies_us: Vec<u64>,
    /// Per EBM score() call latency in microseconds.
    pub ebm_latencies_us: Vec<u64>,
}

/// Result of searching for a proof of a single theorem.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Name of the theorem that was searched.
    pub theorem_name: String,
    /// Whether a complete proof was found.
    pub proved: bool,
    /// Tactic sequence if proved, empty otherwise.
    pub proof_tactics: Vec<String>,
    /// Number of nodes expanded during search.
    pub nodes_expanded: u32,
    /// Total number of states created during search.
    pub total_states: u32,
    /// Maximum depth reached in the search tree.
    pub max_depth_reached: u32,
    /// Wall-clock time in milliseconds for the search.
    pub wall_time_ms: u64,
    /// All trajectory records from the search.
    pub all_records: Vec<TrajectoryRecord>,
    /// Detailed search statistics (timing, pruning, frontier size).
    pub stats: SearchStats,
}

/// A theorem to attempt to prove.
#[derive(Debug, Clone, Deserialize)]
pub struct TheoremTask {
    /// Human-readable name for the theorem.
    pub name: String,
    /// Lean 4 expression for goal.start.
    pub statement: String,
    /// Optional source file path.
    #[serde(default)]
    pub file_path: Option<String>,
    /// Optional line number in source file.
    #[serde(default)]
    pub line_number: Option<u32>,
}

/// Collection of theorems loaded from a JSON file.
#[derive(Debug, Clone, Deserialize)]
pub struct TheoremIndex {
    /// List of theorem tasks.
    pub theorems: Vec<TheoremTask>,
}

impl TheoremIndex {
    /// Load theorem index from a JSON file.
    pub fn from_json(path: &Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let index: Self = serde_json::from_str(&contents)?;
        tracing::info!(count = index.theorems.len(), path = %path.display(), "Loaded theorem index");
        Ok(index)
    }

    /// Number of theorems in the index.
    pub fn len(&self) -> usize {
        self.theorems.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.theorems.is_empty()
    }
}

/// Quick statistics from a trajectory file.
#[derive(Debug, Clone)]
pub struct TrajectorySummary {
    /// Total number of trajectory records.
    pub total_records: usize,
    /// Number of records labeled positive.
    pub positive_count: usize,
    /// Number of records labeled negative.
    pub negative_count: usize,
    /// Number of unique theorem names.
    pub unique_theorems: usize,
    /// Number of theorems with at least one is_proof_complete=true record.
    pub proved_theorems: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_display() {
        assert_eq!(TrajectoryLabel::Positive.to_string(), "positive");
        assert_eq!(TrajectoryLabel::Negative.to_string(), "negative");
        assert_eq!(TrajectoryLabel::Unknown.to_string(), "unknown");
    }

    #[test]
    fn test_label_from_str_lossy() {
        assert_eq!(TrajectoryLabel::from_str_lossy("positive"), TrajectoryLabel::Positive);
        assert_eq!(TrajectoryLabel::from_str_lossy("negative"), TrajectoryLabel::Negative);
        assert_eq!(TrajectoryLabel::from_str_lossy("unknown"), TrajectoryLabel::Unknown);
        assert_eq!(TrajectoryLabel::from_str_lossy("garbage"), TrajectoryLabel::Unknown);
        assert_eq!(TrajectoryLabel::from_str_lossy(""), TrajectoryLabel::Unknown);
    }

    #[test]
    fn test_label_serde_roundtrip() {
        for label in [TrajectoryLabel::Positive, TrajectoryLabel::Negative, TrajectoryLabel::Unknown] {
            let json = serde_json::to_string(&label).unwrap();
            let parsed: TrajectoryLabel = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, label);
        }
    }

    #[test]
    fn test_theorem_task_deserialize() {
        // With all fields
        let json = r#"{"name": "nat_add_zero", "statement": "∀ n, n + 0 = n", "file_path": "Mathlib/Nat.lean", "line_number": 42}"#;
        let task: TheoremTask = serde_json::from_str(json).unwrap();
        assert_eq!(task.name, "nat_add_zero");
        assert_eq!(task.statement, "∀ n, n + 0 = n");
        assert_eq!(task.file_path.as_deref(), Some("Mathlib/Nat.lean"));
        assert_eq!(task.line_number, Some(42));

        // Without optional fields
        let json = r#"{"name": "true_is_true", "statement": "True"}"#;
        let task: TheoremTask = serde_json::from_str(json).unwrap();
        assert_eq!(task.name, "true_is_true");
        assert!(task.file_path.is_none());
        assert!(task.line_number.is_none());
    }

    #[test]
    fn test_theorem_index_deserialize() {
        let json = r#"{
            "theorems": [
                {"name": "t1", "statement": "True"},
                {"name": "t2", "statement": "∀ n, n = n"},
                {"name": "t3", "statement": "False → False"}
            ]
        }"#;
        let index: TheoremIndex = serde_json::from_str(json).unwrap();
        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
        assert_eq!(index.theorems[0].name, "t1");
        assert_eq!(index.theorems[2].statement, "False → False");
    }

    #[test]
    fn test_trajectory_record_default_fields() {
        let record = TrajectoryRecord {
            theorem_name: "test".to_string(),
            state_id: 0,
            state_pp: "⊢ True".to_string(),
            tactic_applied: String::new(),
            parent_state_id: None,
            label: TrajectoryLabel::Unknown,
            depth_from_root: 0,
            remaining_depth: -1,
            llm_log_prob: 0.0,
            ebm_score: 0.0,
            is_proof_complete: false,
            timestamp_ms: 1700000000000,
        };
        assert_eq!(record.state_id, 0);
        assert!(record.parent_state_id.is_none());
        assert_eq!(record.label, TrajectoryLabel::Unknown);
        assert!(!record.is_proof_complete);
    }
}
