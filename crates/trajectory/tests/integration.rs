//! Integration tests for the trajectory crate.
//!
//! These test full pipelines: labeling → writing → reading → verification.
//! No external dependencies needed (no Lean, no model).

use std::collections::HashMap;
use tempfile::TempDir;
use trajectory::{
    SearchResult, SearchStats, TheoremIndex, TrajectoryLabel, TrajectoryReader, TrajectoryRecord,
    TrajectoryWriter,
};

fn make_record(
    theorem: &str,
    state_id: u64,
    parent: Option<u64>,
    depth: u32,
    complete: bool,
) -> TrajectoryRecord {
    TrajectoryRecord {
        theorem_name: theorem.to_string(),
        state_id,
        state_pp: format!("⊢ goal_{state_id}"),
        tactic_applied: if parent.is_some() {
            format!("tactic_{state_id}")
        } else {
            String::new()
        },
        parent_state_id: parent,
        label: TrajectoryLabel::Unknown,
        depth_from_root: depth,
        remaining_depth: -1,
        llm_log_prob: -0.3 * state_id as f64,
        ebm_score: 0.0,
        is_proof_complete: complete,
        timestamp_ms: 1700000000000 + state_id,
        q_value: 0.0,
        visits: 0,
    }
}

/// Full pipeline: from_search_result → write → read → verify labels survived.
#[test]
fn test_label_roundtrip_through_parquet() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("labeled.parquet");

    // Build a search tree:
    //   0 (root) → 1 → 2 (QED)
    //            ↘ 3 → 4 (dead end)
    //                 ↘ 5 (dead end)
    let records = vec![
        make_record("thm_a", 0, None, 0, false),
        make_record("thm_a", 1, Some(0), 1, false),
        make_record("thm_a", 2, Some(1), 2, true),
        make_record("thm_a", 3, Some(0), 1, false),
        make_record("thm_a", 4, Some(3), 2, false),
        make_record("thm_a", 5, Some(3), 2, false),
    ];

    let result = SearchResult {
        theorem_name: "thm_a".to_string(),
        proved: true,
        proof_tactics: vec!["tactic_1".into(), "tactic_2".into()],
        nodes_expanded: 6,
        total_states: 6,
        max_depth_reached: 2,
        wall_time_ms: 500,
        all_records: records,
        stats: SearchStats::default(),
        failure_reason: String::new(),
    };

    // Label → write → read
    let labeled = TrajectoryWriter::from_search_result(&result);
    let mut writer = TrajectoryWriter::new(path.clone());
    writer.record_all(labeled);
    writer.finish().unwrap();

    let read_back = TrajectoryReader::read_all(&path).unwrap();
    assert_eq!(read_back.len(), 6);

    let by_id: HashMap<u64, &TrajectoryRecord> =
        read_back.iter().map(|r| (r.state_id, r)).collect();

    // Proof path: 0 → 1 → 2
    assert_eq!(by_id[&0].label, TrajectoryLabel::Positive);
    assert_eq!(by_id[&0].remaining_depth, 2);

    assert_eq!(by_id[&1].label, TrajectoryLabel::Positive);
    assert_eq!(by_id[&1].remaining_depth, 1);

    assert_eq!(by_id[&2].label, TrajectoryLabel::Positive);
    assert_eq!(by_id[&2].remaining_depth, 0);
    assert!(by_id[&2].is_proof_complete);

    // Dead ends: 3, 4, 5
    assert_eq!(by_id[&3].label, TrajectoryLabel::Negative);
    assert_eq!(by_id[&3].remaining_depth, -1);

    assert_eq!(by_id[&4].label, TrajectoryLabel::Negative);
    assert_eq!(by_id[&5].label, TrajectoryLabel::Negative);
}

/// TheoremIndex::from_json with an actual file on disk.
#[test]
fn test_theorem_index_from_json_file() {
    let tmp = TempDir::new().unwrap();
    let json_path = tmp.path().join("theorems.json");

    let json_content = r#"{
        "theorems": [
            {"name": "nat_add_zero", "statement": "∀ (n : Nat), n + 0 = n", "file_path": "Mathlib/Nat.lean", "line_number": 42},
            {"name": "true_trivial", "statement": "True"},
            {"name": "imp_refl", "statement": "∀ (P : Prop), P → P", "file_path": "Mathlib/Logic.lean"}
        ]
    }"#;
    std::fs::write(&json_path, json_content).unwrap();

    let index = TheoremIndex::from_json(&json_path).unwrap();
    assert_eq!(index.len(), 3);
    assert!(!index.is_empty());

    assert_eq!(index.theorems[0].name, "nat_add_zero");
    assert_eq!(index.theorems[0].file_path.as_deref(), Some("Mathlib/Nat.lean"));
    assert_eq!(index.theorems[0].line_number, Some(42));

    assert_eq!(index.theorems[1].name, "true_trivial");
    assert!(index.theorems[1].file_path.is_none());
    assert!(index.theorems[1].line_number.is_none());

    assert_eq!(index.theorems[2].name, "imp_refl");
    assert_eq!(index.theorems[2].file_path.as_deref(), Some("Mathlib/Logic.lean"));
    assert!(index.theorems[2].line_number.is_none());
}

/// TheoremIndex::from_json returns an error for invalid JSON.
#[test]
fn test_theorem_index_from_json_invalid() {
    let tmp = TempDir::new().unwrap();
    let json_path = tmp.path().join("bad.json");
    std::fs::write(&json_path, "not valid json {{{").unwrap();

    let result = TheoremIndex::from_json(&json_path);
    assert!(result.is_err());
}

/// TheoremIndex::from_json returns an error for missing file.
#[test]
fn test_theorem_index_from_json_missing_file() {
    let result = TheoremIndex::from_json(std::path::Path::new("/nonexistent/theorems.json"));
    assert!(result.is_err());
}

/// read_multiple concatenates records from multiple Parquet files.
#[test]
fn test_read_multiple_files() {
    let tmp = TempDir::new().unwrap();

    // Write 3 files with different theorems
    let files: Vec<_> = (0..3)
        .map(|file_idx| {
            let path = tmp.path().join(format!("batch_{file_idx}.parquet"));
            let theorem = format!("thm_{file_idx}");
            let mut writer = TrajectoryWriter::new(path.clone());
            for i in 0..(10 + file_idx * 5) {
                writer.record(make_record(
                    &theorem,
                    i as u64,
                    if i == 0 { None } else { Some(i as u64 - 1) },
                    i as u32,
                    false,
                ));
            }
            writer.finish().unwrap();
            path
        })
        .collect();

    // Read all three
    let all = TrajectoryReader::read_multiple(&files).unwrap();
    // file 0: 10, file 1: 15, file 2: 20
    assert_eq!(all.len(), 10 + 15 + 20);

    // Verify each theorem's records are present
    let counts: HashMap<String, usize> = all.iter().fold(HashMap::new(), |mut acc, r| {
        *acc.entry(r.theorem_name.clone()).or_default() += 1;
        acc
    });
    assert_eq!(counts["thm_0"], 10);
    assert_eq!(counts["thm_1"], 15);
    assert_eq!(counts["thm_2"], 20);
}

/// Multi-theorem trajectory: mix of proved and unproved, write all, read summary.
#[test]
fn test_multi_theorem_pipeline() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("multi.parquet");

    let mut writer = TrajectoryWriter::new(path.clone());

    // Theorem 1: proved (root → 1 → 2 QED, plus dead end 3)
    let result1 = SearchResult {
        theorem_name: "proved_thm".to_string(),
        proved: true,
        proof_tactics: vec!["intro n".into(), "rfl".into()],
        nodes_expanded: 4,
        total_states: 4,
        max_depth_reached: 2,
        wall_time_ms: 200,
        all_records: vec![
            make_record("proved_thm", 0, None, 0, false),
            make_record("proved_thm", 1, Some(0), 1, false),
            make_record("proved_thm", 2, Some(1), 2, true),
            make_record("proved_thm", 3, Some(0), 1, false),
        ],
        stats: SearchStats::default(),
        failure_reason: String::new(),
    };

    // Theorem 2: not proved (5 dead-end nodes)
    let result2 = SearchResult {
        theorem_name: "failed_thm".to_string(),
        proved: false,
        proof_tactics: vec![],
        nodes_expanded: 5,
        total_states: 5,
        max_depth_reached: 3,
        wall_time_ms: 5000,
        all_records: vec![
            make_record("failed_thm", 10, None, 0, false),
            make_record("failed_thm", 11, Some(10), 1, false),
            make_record("failed_thm", 12, Some(11), 2, false),
            make_record("failed_thm", 13, Some(11), 2, false),
            make_record("failed_thm", 14, Some(12), 3, false),
        ],
        stats: SearchStats::default(),
        failure_reason: String::new(),
    };

    // Theorem 3: proved (trivial — root → 1 QED)
    let result3 = SearchResult {
        theorem_name: "trivial_thm".to_string(),
        proved: true,
        proof_tactics: vec!["trivial".into()],
        nodes_expanded: 2,
        total_states: 2,
        max_depth_reached: 1,
        wall_time_ms: 50,
        all_records: vec![
            make_record("trivial_thm", 20, None, 0, false),
            make_record("trivial_thm", 21, Some(20), 1, true),
        ],
        stats: SearchStats::default(),
        failure_reason: String::new(),
    };

    // Label and write all
    writer.record_all(TrajectoryWriter::from_search_result(&result1));
    writer.record_all(TrajectoryWriter::from_search_result(&result2));
    writer.record_all(TrajectoryWriter::from_search_result(&result3));
    writer.finish().unwrap();

    // Read summary
    let summary = TrajectoryReader::read_summary(&path).unwrap();
    assert_eq!(summary.total_records, 11); // 4 + 5 + 2
    assert_eq!(summary.unique_theorems, 3);
    assert_eq!(summary.proved_theorems, 2); // proved_thm and trivial_thm

    // proved_thm: 3 positive (0,1,2) + 1 negative (3) = 3 positive
    // failed_thm: 5 negative
    // trivial_thm: 2 positive (20,21)
    assert_eq!(summary.positive_count, 5); // 3 + 2
    assert_eq!(summary.negative_count, 6); // 1 + 5

    // Filter for failed theorem — all should be negative
    let failed = TrajectoryReader::read_for_theorem(&path, "failed_thm").unwrap();
    assert_eq!(failed.len(), 5);
    assert!(failed.iter().all(|r| r.label == TrajectoryLabel::Negative));
    assert!(failed.iter().all(|r| r.remaining_depth == -1));

    // Filter for proved theorem — check proof path
    let proved = TrajectoryReader::read_for_theorem(&path, "proved_thm").unwrap();
    assert_eq!(proved.len(), 4);
    let positive_count = proved.iter().filter(|r| r.label == TrajectoryLabel::Positive).count();
    assert_eq!(positive_count, 3); // root, 1, 2 on path
}

/// Two theorems with overlapping state_ids — verify independent labeling.
///
/// Realistic scenario: Pantograph restarts state IDs per worker, so different
/// theorems can have the same state_id values (0, 1, 2).
#[test]
fn test_overlapping_state_ids_across_theorems() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("overlap.parquet");

    let mut writer = TrajectoryWriter::new(path.clone());

    // Theorem A: proved (root 0 → 1 → 2 QED)
    let result_a = SearchResult {
        theorem_name: "thm_a".to_string(),
        proved: true,
        proof_tactics: vec!["tactic_1".into(), "tactic_2".into()],
        nodes_expanded: 3,
        total_states: 3,
        max_depth_reached: 2,
        wall_time_ms: 100,
        all_records: vec![
            make_record("thm_a", 0, None, 0, false),
            make_record("thm_a", 1, Some(0), 1, false),
            make_record("thm_a", 2, Some(1), 2, true),
        ],
        stats: SearchStats::default(),
        failure_reason: String::new(),
    };

    // Theorem B: unproved (root 0 → 1 dead end, → 2 dead end)
    // Same state_ids as theorem A!
    let result_b = SearchResult {
        theorem_name: "thm_b".to_string(),
        proved: false,
        proof_tactics: vec![],
        nodes_expanded: 3,
        total_states: 3,
        max_depth_reached: 1,
        wall_time_ms: 200,
        all_records: vec![
            make_record("thm_b", 0, None, 0, false),
            make_record("thm_b", 1, Some(0), 1, false),
            make_record("thm_b", 2, Some(0), 1, false),
        ],
        stats: SearchStats::default(),
        failure_reason: String::new(),
    };

    // Label independently and write
    writer.record_all(TrajectoryWriter::from_search_result(&result_a));
    writer.record_all(TrajectoryWriter::from_search_result(&result_b));
    writer.finish().unwrap();

    // Read back all records
    let all = TrajectoryReader::read_all(&path).unwrap();
    assert_eq!(all.len(), 6); // 3 + 3

    // Build per-theorem maps
    let thm_a_records: Vec<_> = all.iter().filter(|r| r.theorem_name == "thm_a").collect();
    let thm_b_records: Vec<_> = all.iter().filter(|r| r.theorem_name == "thm_b").collect();
    assert_eq!(thm_a_records.len(), 3);
    assert_eq!(thm_b_records.len(), 3);

    // Theorem A (proved): state_id 0 → Positive
    let a_by_id: HashMap<u64, &TrajectoryRecord> =
        thm_a_records.iter().map(|r| (r.state_id, *r)).collect();
    assert_eq!(a_by_id[&0].label, TrajectoryLabel::Positive, "thm_a state_id=0 should be Positive");
    assert_eq!(a_by_id[&1].label, TrajectoryLabel::Positive, "thm_a state_id=1 should be Positive");
    assert_eq!(a_by_id[&2].label, TrajectoryLabel::Positive, "thm_a state_id=2 should be Positive (QED)");

    // Theorem B (unproved): state_id 0 → Negative (independent of thm_a!)
    let b_by_id: HashMap<u64, &TrajectoryRecord> =
        thm_b_records.iter().map(|r| (r.state_id, *r)).collect();
    assert_eq!(b_by_id[&0].label, TrajectoryLabel::Negative, "thm_b state_id=0 should be Negative");
    assert_eq!(b_by_id[&1].label, TrajectoryLabel::Negative, "thm_b state_id=1 should be Negative");
    assert_eq!(b_by_id[&2].label, TrajectoryLabel::Negative, "thm_b state_id=2 should be Negative");
}
