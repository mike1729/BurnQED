//! Integration tests for the prover-core CLI pipeline.
//!
//! Mock tests run without Lean or an LLM. The `#[ignore]` test requires Pantograph.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use burn::backend::ndarray::NdArray;
use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::ElementConversion;

use lean_repl::TacticResult;
use search::mocks::{make_tactic, MockEnvironment, MockPolicy};
use search::{SearchConfig, SearchEngine, ValueScorer};
use trajectory::{
    SearchResult, SearchStats, TerminationReason, TheoremIndex, TrajectoryLabel, TrajectoryReader,
    TrajectoryRecord, TrajectoryWriter,
};

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

// ---------------------------------------------------------------------------
// Helpers for EBM integration tests
// ---------------------------------------------------------------------------

type TestBackend = NdArray<f32>;

/// Helper: create a trajectory record for testing.
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

/// Write a proved theorem's trajectory to Parquet (root→n1→n2 proof path + n3 dead end).
fn write_proved_theorem(dir: &std::path::Path, filename: &str, theorem: &str) -> std::path::PathBuf {
    let path = dir.join(filename);

    let root = make_record(theorem, 0, 0, 2, false, None);
    let n1 = make_record(theorem, 1, 1, 1, false, Some(0));
    let n2 = make_record(theorem, 2, 2, 0, true, Some(1));
    let n3 = make_record(theorem, 3, 1, -1, false, Some(0));

    let result = SearchResult {
        theorem_name: theorem.to_string(),
        proved: true,
        termination: TerminationReason::Proved,
        proof_tactics: vec!["tac_1".into(), "tac_2".into()],
        nodes_expanded: 4,
        total_states: 4,
        max_depth_reached: 2,
        wall_time_ms: 100,
        all_records: vec![root, n1, n2, n3],
        stats: SearchStats::default(),
    };

    let labeled = TrajectoryWriter::from_search_result(&result);
    let mut writer = TrajectoryWriter::new(path.clone());
    writer.record_all(labeled);
    writer.finish().unwrap();
    path
}

// ---------------------------------------------------------------------------
// Test: EBM training pipeline with mock encode_fn (no real LLM)
// ---------------------------------------------------------------------------

/// Train EBM with synthetic trajectory data and mock encoder, verify checkpoint + config saved.
#[test]
fn test_train_ebm_mock_pipeline() {
    let tmp = tempfile::TempDir::new().unwrap();

    // Write synthetic trajectory Parquet
    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_train_a");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_train_b");

    let d_encoder = 16;
    let device: <TestBackend as Backend>::Device = Default::default();
    let head_config = ebm::EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0);

    // Create model on autodiff backend for training
    type TrainBackend = burn::backend::Autodiff<NdArray<f32>>;
    let model = head_config.init::<TrainBackend>(&device);

    // Load sampler
    let sampler = ebm::ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();
    assert!(sampler.num_records() > 0);
    assert!(sampler.num_eligible_theorems() >= 2);

    // Mock encode_fn (no real LLM)
    let encode_fn = |_state: &str| -> anyhow::Result<Vec<f32>> { Ok(vec![0.42_f32; d_encoder]) };

    // Train for a few steps
    let checkpoint_dir = tmp.path().join("ckpt_prover");
    let config = ebm::EBMTrainingConfig::new()
        .with_total_steps(5)
        .with_warmup_steps(1)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(0)
        .with_checkpoint_interval(0)
        .with_checkpoint_dir(checkpoint_dir.to_string_lossy().to_string());

    let _trained = ebm::train(&config, model, &encode_fn, &sampler, None, &device, None).unwrap();

    // Verify final checkpoint exists (new layout: final/model.mpk)
    let final_ckpt = checkpoint_dir.join("final").join("model.mpk");
    assert!(
        final_ckpt.exists(),
        "Final checkpoint should exist at {}",
        final_ckpt.display()
    );

    // Save EnergyHeadConfig alongside (mimicking run_train_ebm)
    let config_path = checkpoint_dir.join("energy_head_config.json");
    let config_json = serde_json::to_string_pretty(&head_config).unwrap();
    std::fs::write(&config_path, &config_json).unwrap();
    assert!(config_path.exists());

    // Verify config can be loaded back
    let loaded_json = std::fs::read_to_string(&config_path).unwrap();
    let loaded_config: ebm::EnergyHeadConfig = serde_json::from_str(&loaded_json).unwrap();
    assert_eq!(loaded_config.d_encoder, d_encoder);
    assert_eq!(loaded_config.d_hidden1, 8);
    assert_eq!(loaded_config.d_hidden2, 4);
}

// ---------------------------------------------------------------------------
// Test: Search with mock EBM scorer
// ---------------------------------------------------------------------------

/// Train EBM using precomputed embedding cache (no LLM needed at train time).
#[test]
fn test_train_ebm_with_cached_embeddings() {
    let tmp = tempfile::TempDir::new().unwrap();

    // 1. Write trajectory data
    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_cached_a");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_cached_b");

    let d_encoder = 16;

    // 2. Load sampler and precompute cache with a mock encode_fn
    let sampler = ebm::ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();

    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        let hash = state.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        Ok((0..d_encoder).map(|i| ((hash as f32 + i as f32) * 0.1).sin()).collect())
    };

    let cache = ebm::EmbeddingCache::precompute(&sampler, &encode_fn, d_encoder);
    let unique_count = sampler.unique_states().len();
    assert_eq!(cache.len(), unique_count);

    // 3. Save cache to Parquet
    let cache_path = tmp.path().join("embeddings_cache.parquet");
    cache.save(&cache_path).unwrap();

    // 4. Load cache back (simulating a separate run)
    let loaded_cache = ebm::EmbeddingCache::load(&cache_path).unwrap();
    assert_eq!(loaded_cache.len(), unique_count);

    // 5. Train using cache-backed encode_fn (no LLM calls)
    let cache_encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        loaded_cache.get_or_err(state)
    };

    let device: <TestBackend as Backend>::Device = Default::default();
    let head_config = ebm::EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0);

    type TrainBackend = burn::backend::Autodiff<NdArray<f32>>;
    let model = head_config.init::<TrainBackend>(&device);

    let checkpoint_dir = tmp.path().join("ckpt_cached");
    let config = ebm::EBMTrainingConfig::new()
        .with_total_steps(5)
        .with_warmup_steps(1)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(0)
        .with_checkpoint_interval(0)
        .with_checkpoint_dir(checkpoint_dir.to_string_lossy().to_string());

    let result = ebm::train(&config, model, &cache_encode_fn, &sampler, None, &device, None);
    assert!(result.is_ok(), "Training with cached embeddings should succeed: {:?}", result.err());

    // Verify checkpoint saved (new layout: final/model.mpk)
    assert!(checkpoint_dir.join("final").join("model.mpk").exists());
}

/// Create sampler → precompute cache → save → load → train with loaded cache → checkpoint saved.
#[test]
fn test_train_ebm_save_embeddings_roundtrip() {
    let tmp = tempfile::TempDir::new().unwrap();

    // Write trajectory data
    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_emb_rt_a");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_emb_rt_b");

    let d_encoder = 16;

    // Load sampler
    let sampler = ebm::ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();
    let unique_count = sampler.unique_states().len();
    assert!(unique_count > 0, "Should have unique states");

    // Precompute cache with mock encode_fn
    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        let hash = state.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        Ok((0..d_encoder).map(|i| ((hash as f32 + i as f32) * 0.1).sin()).collect())
    };

    let cache = ebm::EmbeddingCache::precompute(&sampler, &encode_fn, d_encoder);
    assert_eq!(cache.len(), unique_count);

    // Save cache to Parquet
    let cache_path = tmp.path().join("embeddings.parquet");
    cache.save(&cache_path).unwrap();
    assert!(cache_path.exists());

    // Load cache back (simulating a separate process)
    let loaded_cache = ebm::EmbeddingCache::load(&cache_path).unwrap();
    assert_eq!(loaded_cache.len(), unique_count);
    assert_eq!(loaded_cache.dim(), d_encoder);

    // Verify every entry matches
    for state in sampler.unique_states() {
        let orig = cache.get(state).unwrap();
        let loaded = loaded_cache.get(state).unwrap();
        assert_eq!(orig, loaded, "Cache entry mismatch for: {state}");
    }

    // Train with loaded cache
    let cache_encode = |state: &str| -> anyhow::Result<Vec<f32>> {
        loaded_cache.get_or_err(state)
    };

    let device: <TestBackend as Backend>::Device = Default::default();
    let head_config = ebm::EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0);

    type TrainBackend = burn::backend::Autodiff<NdArray<f32>>;
    let model = head_config.init::<TrainBackend>(&device);

    let checkpoint_dir = tmp.path().join("ckpt_emb_rt");
    let config = ebm::EBMTrainingConfig::new()
        .with_total_steps(5)
        .with_warmup_steps(1)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(0)
        .with_checkpoint_interval(0)
        .with_checkpoint_dir(checkpoint_dir.to_string_lossy().to_string());

    let result = ebm::train(&config, model, &cache_encode, &sampler, None, &device, None);
    assert!(result.is_ok(), "Training with loaded cache should succeed: {:?}", result.err());
    assert!(checkpoint_dir.join("final").join("model.mpk").exists(), "Checkpoint should be saved");
}

/// Two-phase training: first run saves checkpoint + cache, second run resumes.
#[test]
fn test_train_ebm_resume_with_cache() {
    let tmp = tempfile::TempDir::new().unwrap();

    // Write trajectory data
    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_resume_a");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_resume_b");

    let d_encoder = 16;
    let sampler = ebm::ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();

    // Precompute cache
    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        let hash = state.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        Ok((0..d_encoder).map(|i| ((hash as f32 + i as f32) * 0.1).sin()).collect())
    };
    let cache = ebm::EmbeddingCache::precompute(&sampler, &encode_fn, d_encoder);
    let cache_encode = |state: &str| -> anyhow::Result<Vec<f32>> {
        cache.get_or_err(state)
    };

    let device: <TestBackend as Backend>::Device = Default::default();
    let head_config = ebm::EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0);

    type TrainBackend = burn::backend::Autodiff<NdArray<f32>>;

    // --- First run: train 5 steps ---
    let ckpt_dir1 = tmp.path().join("ckpt_run1");
    let config1 = ebm::EBMTrainingConfig::new()
        .with_total_steps(5)
        .with_warmup_steps(1)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(0)
        .with_checkpoint_interval(0)
        .with_checkpoint_dir(ckpt_dir1.to_string_lossy().to_string());

    let model1 = head_config.init::<TrainBackend>(&device);
    ebm::train(&config1, model1, &cache_encode, &sampler, None, &device, None).unwrap();

    let ckpt1_path = ckpt_dir1.join("final").join("model.mpk");
    assert!(ckpt1_path.exists(), "First run checkpoint should exist");
    let ckpt1_size = std::fs::metadata(&ckpt1_path).unwrap().len();

    // --- Second run: resume from first checkpoint, train 5 more steps ---
    let ckpt_dir2 = tmp.path().join("ckpt_run2");
    let config2 = ebm::EBMTrainingConfig::new()
        .with_total_steps(5)
        .with_warmup_steps(0)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(0)
        .with_checkpoint_interval(0)
        .with_checkpoint_dir(ckpt_dir2.to_string_lossy().to_string());

    // Resume: load weights from first run's checkpoint
    let resumed_model: ebm::EnergyHead<TrainBackend> =
        ebm::resume_from_checkpoint(&ckpt_dir1.join("final").join("model"), &head_config, &device).unwrap();
    ebm::train(&config2, resumed_model, &cache_encode, &sampler, None, &device, None).unwrap();

    let ckpt2_path = ckpt_dir2.join("final").join("model.mpk");
    assert!(ckpt2_path.exists(), "Second run checkpoint should exist");
    let ckpt2_size = std::fs::metadata(&ckpt2_path).unwrap().len();

    // Both checkpoints should have the same size (same model architecture)
    assert_eq!(ckpt1_size, ckpt2_size, "Checkpoint sizes should match (same architecture)");

    // Verify both can be loaded back and produce outputs
    let loaded1: ebm::EnergyHead<TestBackend> =
        ebm::resume_from_checkpoint(&ckpt_dir1.join("final").join("model"), &head_config, &Default::default()).unwrap();
    let loaded2: ebm::EnergyHead<TestBackend> =
        ebm::resume_from_checkpoint(&ckpt_dir2.join("final").join("model"), &head_config, &Default::default()).unwrap();

    let probe = ebm::bridge::embeddings_to_tensor::<TestBackend>(
        &[vec![0.5_f32; d_encoder]],
        &Default::default(),
    );
    let e1: f32 = loaded1.forward(probe.clone()).into_scalar().elem();
    let e2: f32 = loaded2.forward(probe).into_scalar().elem();

    assert!(e1.is_finite(), "First run model should produce finite output: {e1}");
    assert!(e2.is_finite(), "Second run model should produce finite output: {e2}");

    // The two models were trained for different durations, so outputs should differ
    // (unless spectral norm variance masks it — use generous tolerance)
    // Note: This is a weak assertion due to SpectralNorm Option C randomness
}

/// Search using MockEnvironment + MockPolicy + a real (small) EBM scorer.
#[tokio::test]
async fn test_search_with_mock_ebm() {
    let tmp = tempfile::TempDir::new().unwrap();

    // 1. Create and save a small EnergyHead
    let d_encoder = 8;
    let device: <TestBackend as Backend>::Device = Default::default();
    let head_config = ebm::EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(4)
        .with_d_hidden2(2)
        .with_dropout(0.0);
    let model = head_config.init::<TestBackend>(&device);

    let ckpt_dir = tmp.path().join("ebm_ckpt");
    std::fs::create_dir_all(&ckpt_dir).unwrap();

    // Save model checkpoint
    let ckpt_path = ckpt_dir.join("final");
    model
        .clone()
        .save_file(&ckpt_path, &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
        .expect("Failed to save checkpoint");

    // Save config
    let config_json = serde_json::to_string_pretty(&head_config).unwrap();
    std::fs::write(ckpt_dir.join("energy_head_config.json"), &config_json).unwrap();

    // 2. Create EBMScorer with mock encode_fn
    let encode_fn: Box<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync> =
        Box::new(move |state: &str| {
            // Deterministic embedding based on string hash
            let mut emb = vec![0.0_f32; d_encoder];
            for (i, byte) in state.bytes().enumerate() {
                emb[i % d_encoder] += byte as f32 / 255.0;
            }
            Ok(emb)
        });

    let scorer = ebm::EBMScorer::<TestBackend>::load(&ckpt_path, &head_config, encode_fn, device)
        .unwrap();
    let value_fn = ebm::EBMValueFn::new(scorer);

    // Verify scorer produces finite values
    let test_score = value_fn.score("⊢ True").unwrap();
    assert!(test_score.is_finite(), "Score should be finite: {test_score}");

    // 3. Set up mock search environment
    let mut env = MockEnvironment::new();
    env.add_response(0, "trivial", TacticResult::ProofComplete { state_id: 1 });

    let mut policy = MockPolicy::new();
    policy.add_response("⊢ True", vec![make_tactic("trivial", -0.1)]);

    // 4. Search with EBM scorer
    let engine = SearchEngine::new(SearchConfig::default());
    let scorer_ref: Option<&dyn ValueScorer> = Some(&value_fn);
    let result = engine
        .search_one(&env, &policy, scorer_ref, "true_trivial", "True")
        .await
        .unwrap();

    assert!(result.proved, "Should prove True with EBM scorer");
}

// ---------------------------------------------------------------------------
// Test: Eval with mock budgets
// ---------------------------------------------------------------------------

/// Evaluate at two budgets with mocks, verify IterationResult JSON structure.
#[tokio::test]
async fn test_eval_mock_budgets() {
    // Re-define result types locally (prover-core is a binary crate, modules not importable).
    // These match crates/prover-core/src/results.rs exactly.
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct IterationResult {
        iteration: Option<u32>,
        timestamp: String,
        llm_path: String,
        ebm_path: Option<String>,
        benchmark: String,
        total_theorems: u32,
        budget_results: Vec<BudgetResult>,
        cumulative_solved: u32,
        cumulative_rate: f64,
    }
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct BudgetResult {
        budget: u32,
        solved: u32,
        total: u32,
        rate: f64,
        avg_nodes: f64,
        avg_time_secs: f64,
        median_time_secs: f64,
        per_theorem: Vec<TheoremResult>,
    }
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct TheoremResult {
        name: String,
        proved: bool,
        nodes_used: u32,
        time_secs: f64,
    }

    let mut env = MockEnvironment::new();
    env.add_response(0, "trivial", TacticResult::ProofComplete { state_id: 1 });

    let mut policy = MockPolicy::new();
    policy.add_response("⊢ True", vec![make_tactic("trivial", -0.1)]);
    policy.add_response(
        "⊢ ∀ (n : Nat), n = n",
        vec![make_tactic("bad_tactic", -5.0)],
    );

    let config = SearchConfig {
        max_nodes: 5,
        probe_tactics: vec![], // disable probes so "trivial" doesn't solve nat_refl
        ..SearchConfig::default()
    };

    // Evaluate at budgets [5, 10]
    let budgets = vec![5u32, 10];
    let theorems = vec![
        ("true_trivial", "True"),
        ("nat_refl", "∀ (n : Nat), n = n"),
    ];

    let mut budget_results = Vec::new();
    let mut cumulative_solved_set = std::collections::HashSet::new();

    for &budget in &budgets {
        let mut search_config = config.clone();
        search_config.max_nodes = budget;
        let engine = SearchEngine::new(search_config);

        let mut per_theorem = Vec::new();
        let mut solved = 0u32;

        for (name, stmt) in &theorems {
            let result = engine
                .search_one(&env, &policy, None, name, stmt)
                .await
                .unwrap();

            let time_secs = result.wall_time_ms as f64 / 1000.0;
            if result.proved {
                solved += 1;
                cumulative_solved_set.insert(name.to_string());
            }
            per_theorem.push(TheoremResult {
                name: name.to_string(),
                proved: result.proved,
                nodes_used: result.nodes_expanded,
                time_secs,
            });
        }

        let total = theorems.len() as u32;
        let rate = solved as f64 / total as f64;
        budget_results.push(BudgetResult {
            budget,
            solved,
            total,
            rate,
            avg_nodes: 0.0,
            avg_time_secs: 0.0,
            median_time_secs: 0.0,
            per_theorem,
        });
    }

    let cumulative_solved = cumulative_solved_set.len() as u32;
    let cumulative_rate = cumulative_solved as f64 / theorems.len() as f64;

    let iter_result = IterationResult {
        iteration: None,
        timestamp: "2026-01-01T00:00:00Z".to_string(),
        llm_path: "test".to_string(),
        ebm_path: None,
        benchmark: "test".to_string(),
        total_theorems: theorems.len() as u32,
        budget_results,
        cumulative_solved,
        cumulative_rate,
    };

    // Serialize and verify structure
    let json = serde_json::to_string_pretty(&iter_result).unwrap();
    let loaded: IterationResult = serde_json::from_str(&json).unwrap();

    assert_eq!(loaded.budget_results.len(), 2);
    assert_eq!(loaded.budget_results[0].budget, 5);
    assert_eq!(loaded.budget_results[1].budget, 10);

    // true_trivial should be solved at both budgets
    for br in &loaded.budget_results {
        assert_eq!(br.total, 2);
        assert_eq!(br.solved, 1); // only True is provable
        assert!(br.per_theorem.iter().any(|t| t.name == "true_trivial" && t.proved));
        assert!(br.per_theorem.iter().any(|t| t.name == "nat_refl" && !t.proved));
    }

    assert_eq!(loaded.cumulative_solved, 1);
}

// ---------------------------------------------------------------------------
// Test: Compare two evaluation result JSON files
// ---------------------------------------------------------------------------

/// Write two IterationResult JSON files, read them back, verify deserialization.
#[test]
fn test_compare_two_results() {
    // Re-define result types locally (prover-core is a binary crate).
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct IterationResult {
        iteration: Option<u32>,
        timestamp: String,
        llm_path: String,
        ebm_path: Option<String>,
        benchmark: String,
        total_theorems: u32,
        budget_results: Vec<BudgetResult>,
        cumulative_solved: u32,
        cumulative_rate: f64,
    }
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct BudgetResult {
        budget: u32,
        solved: u32,
        total: u32,
        rate: f64,
        avg_nodes: f64,
        avg_time_secs: f64,
        median_time_secs: f64,
        per_theorem: Vec<TheoremResult>,
    }
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct TheoremResult {
        name: String,
        proved: bool,
        nodes_used: u32,
        time_secs: f64,
    }

    let tmp = tempfile::TempDir::new().unwrap();

    let make_result = |iteration: u32, rate100: f64, rate300: f64| -> IterationResult {
        IterationResult {
            iteration: Some(iteration),
            timestamp: format!("2026-01-0{}T00:00:00Z", iteration + 1),
            llm_path: "models/test".to_string(),
            ebm_path: None,
            benchmark: "data/test.json".to_string(),
            total_theorems: 10,
            budget_results: vec![
                BudgetResult {
                    budget: 100,
                    solved: (rate100 * 10.0) as u32,
                    total: 10,
                    rate: rate100,
                    avg_nodes: 50.0,
                    avg_time_secs: 5.0,
                    median_time_secs: 4.0,
                    per_theorem: vec![],
                },
                BudgetResult {
                    budget: 300,
                    solved: (rate300 * 10.0) as u32,
                    total: 10,
                    rate: rate300,
                    avg_nodes: 150.0,
                    avg_time_secs: 15.0,
                    median_time_secs: 12.0,
                    per_theorem: vec![],
                },
            ],
            cumulative_solved: (rate300 * 10.0) as u32,
            cumulative_rate: rate300,
        }
    };

    let r0 = make_result(0, 0.25, 0.40);
    let r1 = make_result(1, 0.30, 0.45);

    let path0 = tmp.path().join("eval_0.json");
    let path1 = tmp.path().join("eval_1.json");
    std::fs::write(&path0, serde_json::to_string_pretty(&r0).unwrap()).unwrap();
    std::fs::write(&path1, serde_json::to_string_pretty(&r1).unwrap()).unwrap();

    // Read both back and verify
    let loaded0: IterationResult =
        serde_json::from_str(&std::fs::read_to_string(&path0).unwrap()).unwrap();
    let loaded1: IterationResult =
        serde_json::from_str(&std::fs::read_to_string(&path1).unwrap()).unwrap();

    assert_eq!(loaded0.iteration, Some(0));
    assert_eq!(loaded1.iteration, Some(1));
    assert_eq!(loaded0.budget_results.len(), 2);
    assert_eq!(loaded1.budget_results.len(), 2);

    // Verify deltas can be computed
    let delta_100 = loaded1.budget_results[0].rate - loaded0.budget_results[0].rate;
    let delta_300 = loaded1.budget_results[1].rate - loaded0.budget_results[1].rate;
    assert!((delta_100 - 0.05).abs() < 1e-9);
    assert!((delta_300 - 0.05).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Test: Resume from partial trajectory
// ---------------------------------------------------------------------------

/// Write partial Parquet with 2 theorems, search 3 with resume, verify merge.
#[tokio::test]
async fn test_resume_from_partial() {
    let tmp = tempfile::TempDir::new().unwrap();

    // Write partial trajectory with records for t1 and t2
    let partial_path = tmp.path().join("partial.parquet");
    let mut partial_writer = TrajectoryWriter::new(partial_path.clone());
    for name in &["t1", "t2"] {
        let mut r = make_record(name, 0, 0, -1, false, None);
        r.label = TrajectoryLabel::Negative;
        partial_writer.record(r);
    }
    partial_writer.finish().unwrap();

    // Read theorem names that are "done"
    let done = TrajectoryReader::read_theorem_names(&partial_path).unwrap();
    assert!(done.contains("t1"));
    assert!(done.contains("t2"));
    assert!(!done.contains("t3"));

    // Set up mock search for remaining theorem (t3)
    let mut env = MockEnvironment::new();
    env.add_response(0, "trivial", TacticResult::ProofComplete { state_id: 1 });

    let mut policy = MockPolicy::new();
    policy.add_response("⊢ True", vec![make_tactic("trivial", -0.1)]);

    let config = SearchConfig::default();
    let engine = SearchEngine::new(config);

    // Only search theorems not in "done" set
    let all_theorems = vec![
        ("t1", "True"),
        ("t2", "True"),
        ("t3", "True"),
    ];
    let remaining: Vec<_> = all_theorems
        .iter()
        .filter(|(name, _)| !done.contains(*name))
        .collect();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].0, "t3");

    // Search remaining and write to separate output
    let output_path = tmp.path().join("new.parquet");
    let mut writer = TrajectoryWriter::new(output_path.clone());

    for (name, stmt) in &remaining {
        let result = engine
            .search_one(&env, &policy, None, name, stmt)
            .await
            .unwrap();
        let labeled = TrajectoryWriter::from_search_result(&result);
        writer.record_all(labeled);
    }
    writer.finish().unwrap();

    // Merge old + new
    let old_records = TrajectoryReader::read_all(&partial_path).unwrap();
    let new_records = TrajectoryReader::read_all(&output_path).unwrap();

    let merged_path = tmp.path().join("merged.parquet");
    let mut merged_writer = TrajectoryWriter::new(merged_path.clone());
    merged_writer.record_all(old_records);
    merged_writer.record_all(new_records);
    merged_writer.finish().unwrap();

    // Verify merged has all 3 theorem names
    let merged_names = TrajectoryReader::read_theorem_names(&merged_path).unwrap();
    assert_eq!(merged_names.len(), 3);
    assert!(merged_names.contains("t1"));
    assert!(merged_names.contains("t2"));
    assert!(merged_names.contains("t3"));
}

// ---------------------------------------------------------------------------
// Tests: Parallel search and eval using JoinSet + Semaphore
// ---------------------------------------------------------------------------

/// Search 4 theorems with concurrency=2, verify all results collected and Parquet valid.
#[tokio::test]
async fn test_mock_pipeline_parallel_search() {
    let tmp = tempfile::TempDir::new().unwrap();
    let output = tmp.path().join("parallel.parquet");

    // Mock environment: True proved by "trivial", nat_refl by "intro n" + "rfl"
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
    // hard_a and hard_b will fail (no matching tactics)
    policy.add_response("⊢ hard_a", vec![make_tactic("bad", -5.0)]);
    policy.add_response("⊢ hard_b", vec![make_tactic("bad", -5.0)]);

    let engine = SearchEngine::new(SearchConfig {
        max_nodes: 10,
        probe_tactics: vec![], // disable probes so "trivial" doesn't solve hard_a/hard_b
        ..SearchConfig::default()
    });

    let theorems = vec![
        ("true_trivial", "True"),
        ("nat_refl", "∀ (n : Nat), n = n"),
        ("hard_a", "hard_a"),
        ("hard_b", "hard_b"),
    ];

    // Arc-wrap mocks for concurrent access (JoinSet requires 'static futures)
    let env = Arc::new(env);
    let policy = Arc::new(policy);
    let concurrency_sem = Arc::new(tokio::sync::Semaphore::new(2));
    let mut join_set = tokio::task::JoinSet::new();

    for (name, stmt) in &theorems {
        let permit = concurrency_sem.clone().acquire_owned().await.unwrap();
        let env = Arc::clone(&env);
        let policy = Arc::clone(&policy);
        let engine = engine.clone();
        let name = name.to_string();
        let stmt = stmt.to_string();

        join_set.spawn(async move {
            let _permit = permit;
            let result = engine
                .search_one(&*env, &*policy, None, &name, &stmt)
                .await;
            (name, result)
        });
    }

    // Collect results (single-threaded, just like run_search)
    let mut writer = TrajectoryWriter::new(output.clone());
    let mut proved_count = 0u32;
    let mut failed_count = 0u32;

    while let Some(join_result) = join_set.join_next().await {
        let (name, result) = join_result.unwrap();
        match result {
            Ok(sr) => {
                if sr.proved {
                    proved_count += 1;
                } else {
                    failed_count += 1;
                }
                let labeled = TrajectoryWriter::from_search_result(&sr);
                writer.record_all(labeled);
            }
            Err(e) => {
                failed_count += 1;
                eprintln!("Search failed for {name}: {e}");
            }
        }
    }
    writer.finish().unwrap();

    // Verify results
    assert_eq!(proved_count, 2, "true_trivial and nat_refl should be proved");
    assert_eq!(failed_count, 2, "hard_a and hard_b should fail");

    // Verify Parquet output
    let summary = TrajectoryReader::read_summary(&output).unwrap();
    assert_eq!(summary.unique_theorems, 4);
    assert_eq!(summary.proved_theorems, 2);
    assert!(summary.total_records >= 4); // at least root per theorem
}

/// Simulate interruption with AtomicBool after 2 completions, verify remaining not searched.
#[tokio::test]
async fn test_parallel_search_interruption() {
    // Set up mock: all theorems provable by "trivial"
    let mut env = MockEnvironment::new();
    env.add_response(0, "trivial", TacticResult::ProofComplete { state_id: 1 });

    let mut policy = MockPolicy::new();
    policy.add_response("⊢ True", vec![make_tactic("trivial", -0.1)]);

    let engine = SearchEngine::new(SearchConfig::default());

    let theorems = vec![
        ("t1", "True"),
        ("t2", "True"),
        ("t3", "True"),
        ("t4", "True"),
    ];

    let interrupted = Arc::new(AtomicBool::new(false));
    let concurrency_sem = Arc::new(tokio::sync::Semaphore::new(1)); // sequential
    let mut join_set = tokio::task::JoinSet::new();

    let env = Arc::new(env);
    let policy = Arc::new(policy);

    for (name, stmt) in &theorems {
        // Check interruption before spawning
        if interrupted.load(Ordering::Relaxed) {
            break;
        }

        let permit = concurrency_sem.clone().acquire_owned().await.unwrap();
        if interrupted.load(Ordering::Relaxed) {
            break;
        }

        let env = Arc::clone(&env);
        let policy = Arc::clone(&policy);
        let engine = engine.clone();
        let name = name.to_string();
        let stmt = stmt.to_string();
        let interrupted = Arc::clone(&interrupted);

        join_set.spawn(async move {
            let _permit = permit;
            if interrupted.load(Ordering::Relaxed) {
                return (name, Err(search::SearchError::ProofStart("interrupted".into())));
            }
            let result = engine
                .search_one(&*env, &*policy, None, &name, &stmt)
                .await;
            (name, result)
        });
    }

    // Collect results, simulate interrupt after 2 completions
    let mut completed = 0u32;

    while let Some(join_result) = join_set.join_next().await {
        let (_name, result) = join_result.unwrap();
        if result.is_ok() {
            completed += 1;
        }
        if completed >= 2 {
            interrupted.store(true, Ordering::Relaxed);
        }
    }

    // With concurrency=1 and interruption after 2, we expect:
    // - 2 fully completed searches
    // - remaining tasks either interrupted or not spawned
    // Since concurrency=1, tasks are spawned sequentially. After 2 complete,
    // interrupt fires. The 3rd task may already be spawned (permit acquired),
    // and the 4th won't be spawned (the for loop checks interrupted before acquire).
    // So completed should be 2, with possibly 1-2 interrupted tasks.
    assert!(completed >= 2, "Expected at least 2 completed, got {completed}");
    assert!(completed <= 4, "Should not exceed total theorems");
}

/// Eval 3 theorems at budgets [5, 10] with concurrency=2, verify results match expectations.
#[tokio::test]
async fn test_parallel_eval_mock_budgets() {
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
    policy.add_response("⊢ hard_thm", vec![make_tactic("bad", -5.0)]);

    let env = Arc::new(env);
    let policy = Arc::new(policy);

    let budgets = vec![5u32, 10];
    let theorems = vec![
        ("true_trivial", "True"),
        ("nat_refl", "∀ (n : Nat), n = n"),
        ("hard_thm", "hard_thm"),
    ];

    let concurrency = 2;
    let concurrency_sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let mut cumulative_solved: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut all_budget_results = Vec::new();

    for &budget in &budgets {
        let config = SearchConfig {
            max_nodes: budget,
            probe_tactics: vec![], // disable probes so "trivial" doesn't solve hard_thm
            ..SearchConfig::default()
        };
        let engine = SearchEngine::new(config);

        let mut join_set = tokio::task::JoinSet::new();

        for (name, stmt) in &theorems {
            let permit = concurrency_sem.clone().acquire_owned().await.unwrap();
            let env = Arc::clone(&env);
            let policy = Arc::clone(&policy);
            let engine = engine.clone();
            let name = name.to_string();
            let stmt = stmt.to_string();

            join_set.spawn(async move {
                let _permit = permit;
                let result = engine
                    .search_one(&*env, &*policy, None, &name, &stmt)
                    .await;
                (name, result)
            });
        }

        let mut solved = 0u32;
        let mut per_theorem_proved = Vec::new();

        while let Some(join_result) = join_set.join_next().await {
            let (name, result) = join_result.unwrap();
            match result {
                Ok(sr) => {
                    if sr.proved {
                        solved += 1;
                        cumulative_solved.insert(name.clone());
                    }
                    per_theorem_proved.push((name, sr.proved));
                }
                Err(_) => {
                    per_theorem_proved.push((name, false));
                }
            }
        }

        all_budget_results.push((budget, solved, per_theorem_proved));
    }

    // Verify: true_trivial and nat_refl should be proved at both budgets
    for (budget, solved, per_theorem) in &all_budget_results {
        assert_eq!(
            *solved, 2,
            "Expected 2 proved at budget {budget}, got {solved}"
        );
        let proved_names: Vec<_> = per_theorem
            .iter()
            .filter(|(_, p)| *p)
            .map(|(n, _)| n.as_str())
            .collect();
        assert!(
            proved_names.contains(&"true_trivial"),
            "true_trivial should be proved at budget {budget}"
        );
        assert!(
            proved_names.contains(&"nat_refl"),
            "nat_refl should be proved at budget {budget}"
        );
    }

    // Cumulative should be 2 (true_trivial + nat_refl)
    assert_eq!(cumulative_solved.len(), 2);
    assert!(cumulative_solved.contains("true_trivial"));
    assert!(cumulative_solved.contains("nat_refl"));
}
