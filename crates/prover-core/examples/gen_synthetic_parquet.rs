//! Generate a synthetic baseline_raw.parquet for testing the train-ebm pipeline.
//!
//! Usage: cargo run -p prover-core --example gen_synthetic_parquet -- trajectories/baseline_raw.parquet

use search::SearchStats;
use trajectory::{SearchResult, TrajectoryLabel, TrajectoryRecord, TrajectoryWriter};

fn make_record(
    theorem: &str,
    state_id: u64,
    depth: u32,
    remaining: i32,
    complete: bool,
    parent: Option<u64>,
) -> TrajectoryRecord {
    TrajectoryRecord {
        theorem_name: theorem.to_string(),
        state_id,
        state_pp: format!("⊢ state_{theorem}_{state_id}"),
        tactic_applied: if parent.is_some() {
            format!("tac_{state_id}")
        } else {
            String::new()
        },
        parent_state_id: parent,
        label: TrajectoryLabel::Unknown,
        depth_from_root: depth,
        remaining_depth: remaining,
        llm_log_prob: -0.5,
        ebm_score: 0.0,
        is_proof_complete: complete,
        timestamp_ms: 1700000000000 + state_id,
    }
}

fn make_proved_result(theorem: &str) -> SearchResult {
    let root = make_record(theorem, 0, 0, 2, false, None);
    let n1 = make_record(theorem, 1, 1, 1, false, Some(0));
    let n2 = make_record(theorem, 2, 2, 0, true, Some(1));
    let n3 = make_record(theorem, 3, 1, -1, false, Some(0));

    SearchResult {
        theorem_name: theorem.to_string(),
        proved: true,
        proof_tactics: vec!["tac_1".into(), "tac_2".into()],
        nodes_expanded: 4,
        total_states: 4,
        max_depth_reached: 2,
        wall_time_ms: 100,
        all_records: vec![root, n1, n2, n3],
        stats: SearchStats::default(),
    }
}

fn make_unproved_result(theorem: &str) -> SearchResult {
    let root = make_record(theorem, 0, 0, -1, false, None);
    let n1 = make_record(theorem, 1, 1, -1, false, Some(0));

    SearchResult {
        theorem_name: theorem.to_string(),
        proved: false,
        proof_tactics: vec![],
        nodes_expanded: 2,
        total_states: 2,
        max_depth_reached: 1,
        wall_time_ms: 50,
        all_records: vec![root, n1],
        stats: SearchStats::default(),
    }
}

fn main() {
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "trajectories/baseline_raw.parquet".to_string());
    let path = std::path::PathBuf::from(&output);

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }

    let mut writer = TrajectoryWriter::new(path.clone());

    // 5 proved theorems
    for name in &["thm_true", "thm_refl", "thm_imp", "thm_and", "thm_or"] {
        let result = make_proved_result(name);
        let labeled = TrajectoryWriter::from_search_result(&result);
        writer.record_all(labeled);
    }

    // 3 unproved theorems
    for name in &["thm_hard_1", "thm_hard_2", "thm_hard_3"] {
        let result = make_unproved_result(name);
        let labeled = TrajectoryWriter::from_search_result(&result);
        writer.record_all(labeled);
    }

    writer.finish().unwrap();

    let summary = trajectory::TrajectoryReader::read_summary(&path).unwrap();
    println!("Wrote synthetic trajectory to: {output}");
    println!("  Records:     {}", summary.total_records);
    println!("  Positive:    {}", summary.positive_count);
    println!("  Negative:    {}", summary.negative_count);
    println!("  Theorems:    {}", summary.unique_theorems);
    println!("  Proved:      {}", summary.proved_theorems);
}
