//! Integration tests for the prover-core CLI pipeline.
//!
//! Mock tests run without Lean or an LLM. The `#[ignore]` test requires Pantograph.

use std::sync::Arc;

use lean_repl::TacticResult;
use search::mocks::{make_tactic, MockEnvironment, MockPolicy};
use search::{SearchConfig, SearchEngine};
use trajectory::{TheoremIndex, TrajectoryReader, TrajectoryWriter};

/// Search theorems using mocks, write Parquet, read back and verify.
#[tokio::test]
async fn test_mock_pipeline_search_and_write() {
    let tmp = tempfile::TempDir::new().unwrap();
    let output = tmp.path().join("trajectory.parquet");

    // Mock environment: True proved by "trivial", ∀ n, n = n by "intro n" + "rfl"
    let mut env = MockEnvironment::new();
    env.add_response(0, "trivial", TacticResult::ProofComplete { state_id: 1 });
    env.add_response(
        0,
        "intro n",
        TacticResult::Success {
            state_id: 1,
            goals: vec![lean_repl::Goal::parse(0, "n : Nat\n⊢ n = n")],
        },
    );
    env.add_response(1, "rfl", TacticResult::ProofComplete { state_id: 2 });

    let mut policy = MockPolicy::new();
    policy.add_response("⊢ True", vec![make_tactic("trivial", -0.1)]);
    policy.add_response(
        "⊢ ∀ (n : Nat), n = n",
        vec![make_tactic("intro n", -0.3)],
    );
    policy.add_response("n : Nat\n⊢ n = n", vec![make_tactic("rfl", -0.1)]);

    let engine = SearchEngine::new(SearchConfig::default());
    let mut writer = TrajectoryWriter::new(output.clone());
    let mut proved = 0u32;

    let theorems = vec![
        ("true_trivial", "True"),
        ("nat_refl", "∀ (n : Nat), n = n"),
    ];

    for (name, stmt) in &theorems {
        let result = engine
            .search_one(&env, &policy, None, name, stmt)
            .await
            .unwrap();
        if result.proved {
            proved += 1;
        }
        let labeled = TrajectoryWriter::from_search_result(&result);
        writer.record_all(labeled);
    }
    writer.finish().unwrap();

    assert_eq!(proved, 2);

    // Read back and verify
    let summary = TrajectoryReader::read_summary(&output).unwrap();
    assert_eq!(summary.unique_theorems, 2);
    assert_eq!(summary.proved_theorems, 2);
    assert!(summary.positive_count > 0);
    assert!(summary.total_records >= 4); // at least root + terminal per theorem
}

/// All tactics fail — verify negative labels.
#[tokio::test]
async fn test_mock_pipeline_unproved_theorem() {
    let tmp = tempfile::TempDir::new().unwrap();
    let output = tmp.path().join("unproved.parquet");

    let env = MockEnvironment::new();
    let policy = MockPolicy::with_default(vec![make_tactic("bad_tactic", -5.0)]);

    let config = SearchConfig {
        max_nodes: 5,
        ..SearchConfig::default()
    };
    let engine = SearchEngine::new(config);
    let result = engine
        .search_one(&env, &policy, None, "hard_thm", "∀ x, x = x")
        .await
        .unwrap();

    assert!(!result.proved);

    let labeled = TrajectoryWriter::from_search_result(&result);
    let mut writer = TrajectoryWriter::new(output.clone());
    writer.record_all(labeled);
    writer.finish().unwrap();

    let summary = TrajectoryReader::read_summary(&output).unwrap();
    assert_eq!(summary.proved_theorems, 0);
    assert_eq!(summary.positive_count, 0);
    assert!(summary.negative_count > 0);
}

/// Load test_theorems.json and verify it has the expected theorems.
#[test]
fn test_load_test_theorems() {
    // Find data/test_theorems.json relative to workspace root
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let json_path = workspace_root.join("data").join("test_theorems.json");
    assert!(json_path.exists(), "data/test_theorems.json not found at {}", json_path.display());

    let index = TheoremIndex::from_json(&json_path).unwrap();
    assert!(index.len() >= 10, "Expected at least 10 theorems, got {}", index.len());
    assert_eq!(index.theorems[0].name, "true_trivial");
    assert_eq!(index.theorems[0].statement, "True");
}

/// Manually write a Parquet file and verify read_summary.
#[test]
fn test_eval_reads_parquet() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("eval_test.parquet");

    let mut writer = TrajectoryWriter::new(path.clone());
    for i in 0..5 {
        writer.record(trajectory::TrajectoryRecord {
            theorem_name: "t1".to_string(),
            state_id: i,
            state_pp: format!("⊢ state_{i}"),
            tactic_applied: if i == 0 { String::new() } else { format!("tac_{i}") },
            parent_state_id: if i == 0 { None } else { Some(i - 1) },
            label: trajectory::TrajectoryLabel::Positive,
            depth_from_root: i as u32,
            remaining_depth: (4 - i) as i32,
            llm_log_prob: -0.1 * i as f64,
            ebm_score: 0.0,
            is_proof_complete: i == 4,
            timestamp_ms: 1700000000000 + i,
        });
    }
    for i in 0..3 {
        writer.record(trajectory::TrajectoryRecord {
            theorem_name: "t2".to_string(),
            state_id: i,
            state_pp: format!("⊢ state_{i}"),
            tactic_applied: if i == 0 { String::new() } else { format!("tac_{i}") },
            parent_state_id: if i == 0 { None } else { Some(i - 1) },
            label: trajectory::TrajectoryLabel::Negative,
            depth_from_root: i as u32,
            remaining_depth: -1,
            llm_log_prob: -0.5,
            ebm_score: 0.0,
            is_proof_complete: false,
            timestamp_ms: 1700000000000 + i,
        });
    }
    writer.finish().unwrap();

    let summary = TrajectoryReader::read_summary(&path).unwrap();
    assert_eq!(summary.total_records, 8);
    assert_eq!(summary.positive_count, 5);
    assert_eq!(summary.negative_count, 3);
    assert_eq!(summary.unique_theorems, 2);
    assert_eq!(summary.proved_theorems, 1); // only t1 has is_proof_complete
}

/// Verify that configs/search.toml is valid TOML.
#[test]
fn test_real_search_toml_is_valid() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let toml_path = workspace_root.join("configs").join("search.toml");
    assert!(toml_path.exists(), "configs/search.toml not found");

    let contents = std::fs::read_to_string(&toml_path).unwrap();
    let _: toml::Value = toml::from_str(&contents).unwrap();
}

/// Real Lean pool + mock policy — search 3 theorems.
#[tokio::test]
#[ignore]
async fn test_lean_pipeline_multiple_theorems() {
    let tmp = tempfile::TempDir::new().unwrap();
    let output = tmp.path().join("lean_trajectory.parquet");

    // Real Lean pool with auto-discovered Pantograph
    let lean_config = lean_repl::LeanPoolConfig::with_bundled_pantograph()
        .expect("Pantograph not found");
    let pool = Arc::new(lean_repl::LeanPool::new(lean_config).await.unwrap());

    // MockPolicy with canned tactics matching real Lean responses
    let mut policy = MockPolicy::new();
    // True: trivial
    policy.add_contains_response("True", vec![make_tactic("trivial", -0.1)]);
    // ∀ n, n = n: intro n, then rfl
    policy.add_contains_response("∀", vec![make_tactic("intro n", -0.3)]);
    policy.add_contains_response("n = n", vec![make_tactic("rfl", -0.1)]);
    // False → False: intro h; exact h
    policy.add_contains_response("False →", vec![make_tactic("intro h", -0.2)]);
    policy.add_contains_response("⊢ False", vec![make_tactic("exact h", -0.1)]);

    let engine = SearchEngine::new(SearchConfig {
        max_nodes: 20,
        max_depth: 10,
        timeout_per_theorem: 30,
        ..SearchConfig::default()
    });

    let mut writer = TrajectoryWriter::new(output.clone());
    let mut proved_count = 0u32;

    let theorems = vec![
        ("true_trivial", "True"),
        ("nat_refl", "∀ (n : Nat), n = n"),
        ("false_implies_false", "False → False"),
    ];

    for (name, stmt) in &theorems {
        match engine.search_one(&pool, &policy, None, name, stmt).await {
            Ok(result) => {
                if result.proved {
                    proved_count += 1;
                }
                let labeled = TrajectoryWriter::from_search_result(&result);
                writer.record_all(labeled);
            }
            Err(e) => {
                tracing::warn!(theorem = name, error = %e, "Search failed");
            }
        }
    }

    writer.finish().unwrap();
    pool.shutdown().await;

    // At least 2 of 3 should be provable
    assert!(
        proved_count >= 2,
        "Expected at least 2 proved, got {proved_count}"
    );

    // Verify Parquet output
    let summary = TrajectoryReader::read_summary(&output).unwrap();
    assert!(summary.total_records > 0);
    assert!(summary.proved_theorems >= 2);
}
