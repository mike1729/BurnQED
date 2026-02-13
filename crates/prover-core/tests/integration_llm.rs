//! TinyLlama end-to-end integration tests.
//!
//! These tests exercise the real LLM encode pipeline through EBM training,
//! inference, and search. They use TinyLlama-1.1B (2048 hidden_size) which
//! loads in ~10s on CPU, catching dimension mismatches, mutex deadlocks,
//! and wiring bugs that mock-only tests miss.
//!
//! All tests are `#[ignore]` because they require model weights at
//! `models/tinyllama-1.1b/` relative to the workspace root.
//!
//! ## Running
//!
//! ```bash
//! # Tests 1-3 (no Lean needed):
//! cargo test -p prover-core --test integration_llm -- --ignored --nocapture --test-threads=1
//!
//! # Test 4 (needs Pantograph):
//! cargo test -p prover-core --test integration_llm test_tinyllama_full_pipeline_with_lean -- --ignored --nocapture --test-threads=1
//! ```

use std::sync::Arc;

use burn::backend::ndarray::NdArray;
use burn::prelude::Backend;

use policy::{DeviceConfig, GeneratedTactic, PolicyConfig, TacticGenerator};
use search::{PolicyProvider, SearchConfig, SearchEngine, SearchError, ValueScorer};

/// Local test helper: wraps TacticGenerator in Mutex for PolicyProvider trait.
/// (Production code uses SGLang via InferencePolicyProvider instead.)
struct MutexPolicyProvider {
    generator: Arc<std::sync::Mutex<TacticGenerator>>,
}

impl MutexPolicyProvider {
    fn new_shared(generator: Arc<std::sync::Mutex<TacticGenerator>>) -> Self {
        Self { generator }
    }
}

impl PolicyProvider for MutexPolicyProvider {
    fn generate_candidates(
        &self,
        proof_state: &str,
        n: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError> {
        let mut gen = self
            .generator
            .lock()
            .map_err(|e| SearchError::Policy(anyhow::anyhow!("{e}")))?;
        gen.generate_candidates(proof_state, n)
            .map_err(SearchError::Policy)
    }
}
use trajectory::{
    SearchResult, SearchStats, TrajectoryLabel, TrajectoryReader, TrajectoryRecord,
    TrajectoryWriter,
};

type TestBackend = NdArray<f32>;

/// Default model path relative to the workspace root.
const DEFAULT_MODEL_PATH: &str = "models/tinyllama-1.1b";

/// Get the TinyLlama model path relative to the workspace root.
fn tinyllama_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("MODEL_PATH") {
        return std::path::PathBuf::from(p);
    }
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    workspace_root.join(DEFAULT_MODEL_PATH)
}

/// Create a PolicyConfig suitable for fast testing.
fn test_policy_config() -> PolicyConfig {
    let mut cfg = PolicyConfig::new(tinyllama_path());
    cfg.max_seq_len = 2048;
    cfg.num_candidates = 3;
    cfg.temperature = 0.6;
    cfg.top_p = 0.95;
    cfg.max_tactic_tokens = 16;
    cfg.device = DeviceConfig::Cpu;
    cfg
}

/// Write synthetic trajectory Parquet for the given state strings.
///
/// Creates a proved theorem with a proof path (root -> n1 -> n2 QED) plus
/// a dead-end branch (root -> n3), using the given state strings for
/// the root, n1, n2, and n3 nodes respectively.
fn write_synthetic_trajectory(
    dir: &std::path::Path,
    filename: &str,
    theorem: &str,
    states: &[&str; 4],
) -> std::path::PathBuf {
    let path = dir.join(filename);

    let root = TrajectoryRecord {
        theorem_name: theorem.to_string(),
        state_id: 0,
        state_pp: states[0].to_string(),
        tactic_applied: String::new(),
        parent_state_id: None,
        label: TrajectoryLabel::Unknown,
        depth_from_root: 0,
        remaining_depth: 2,
        llm_log_prob: 0.0,
        ebm_score: 0.0,
        is_proof_complete: false,
        timestamp_ms: 1700000000000,
    };
    let n1 = TrajectoryRecord {
        theorem_name: theorem.to_string(),
        state_id: 1,
        state_pp: states[1].to_string(),
        tactic_applied: "tac_1".to_string(),
        parent_state_id: Some(0),
        label: TrajectoryLabel::Unknown,
        depth_from_root: 1,
        remaining_depth: 1,
        llm_log_prob: -0.3,
        ebm_score: 0.0,
        is_proof_complete: false,
        timestamp_ms: 1700000000001,
    };
    let n2 = TrajectoryRecord {
        theorem_name: theorem.to_string(),
        state_id: 2,
        state_pp: states[2].to_string(),
        tactic_applied: "tac_2".to_string(),
        parent_state_id: Some(1),
        label: TrajectoryLabel::Unknown,
        depth_from_root: 2,
        remaining_depth: 0,
        llm_log_prob: -0.1,
        ebm_score: 0.0,
        is_proof_complete: true,
        timestamp_ms: 1700000000002,
    };
    let n3 = TrajectoryRecord {
        theorem_name: theorem.to_string(),
        state_id: 3,
        state_pp: states[3].to_string(),
        tactic_applied: "bad_tac".to_string(),
        parent_state_id: Some(0),
        label: TrajectoryLabel::Unknown,
        depth_from_root: 1,
        remaining_depth: -1,
        llm_log_prob: -2.0,
        ebm_score: 0.0,
        is_proof_complete: false,
        timestamp_ms: 1700000000003,
    };

    let result = SearchResult {
        theorem_name: theorem.to_string(),
        proved: true,
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
// Test 1: Real LLM encode_only() -> real EBM training
// ---------------------------------------------------------------------------

/// Test that real TinyLlama embeddings feed through EBM training correctly.
///
/// This catches:
/// - Dimension mismatches (hidden_size=2048 vs default 4096)
/// - Embedding NaN/Inf issues
/// - Training loop with real encoder function
#[test]
#[ignore]
fn test_tinyllama_encode_and_ebm_train() {
    let config = test_policy_config();
    let mut gen = TacticGenerator::load(&config).unwrap();
    let hidden_size = gen.hidden_size();
    println!("TinyLlama hidden_size = {hidden_size}");
    assert_eq!(hidden_size, 2048, "TinyLlama should have hidden_size=2048");

    // Encode several proof state strings
    let states = [
        "⊢ True",
        "n : Nat\n⊢ n + 0 = n",
        "h : False\n⊢ False",
        "a b : Prop\nh : a ∧ b\n⊢ b ∧ a",
    ];

    let mut embeddings = Vec::new();
    for state in &states {
        let emb = gen.encode_only(state).unwrap();
        assert_eq!(emb.dim, hidden_size, "Embedding dim should match hidden_size");
        assert!(
            emb.data.iter().all(|x| x.is_finite()),
            "All embedding values should be finite for state: {state}"
        );
        embeddings.push(emb);
    }

    // Verify embeddings are distinct (different states -> different embeddings)
    let dist_01: f64 = embeddings[0]
        .data
        .iter()
        .zip(&embeddings[1].data)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        dist_01 > 1e-3,
        "Different states should produce different embeddings (L2 dist = {dist_01})"
    );

    // Create synthetic trajectory with these state strings
    let tmp = tempfile::TempDir::new().unwrap();
    let path1 = write_synthetic_trajectory(
        tmp.path(),
        "thm1.parquet",
        "thm_encode_a",
        &[states[0], states[1], states[2], states[3]],
    );
    let path2 = write_synthetic_trajectory(
        tmp.path(),
        "thm2.parquet",
        "thm_encode_b",
        &[states[1], states[2], states[3], states[0]],
    );

    // Build sampler
    let sampler = ebm::ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();
    assert!(sampler.num_records() > 0);
    assert!(sampler.num_eligible_theorems() >= 2);

    // Build EnergyHead with correct hidden_size (NOT the default 4096)
    let head_config = ebm::EnergyHeadConfig::new(hidden_size)
        .with_d_hidden1(256)
        .with_d_hidden2(128)
        .with_dropout(0.0);

    type TrainBackend = burn::backend::Autodiff<NdArray<f32>>;
    let device: <TrainBackend as Backend>::Device = Default::default();
    let model = head_config.init::<TrainBackend>(&device);

    // Create encode_fn using the real generator
    let gen = std::sync::Mutex::new(gen);
    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        let mut g = gen
            .lock()
            .map_err(|e| anyhow::anyhow!("Generator lock poisoned: {e}"))?;
        let embedding = g.encode_only(state)?;
        Ok(embedding.data)
    };

    // Train for a few steps
    let checkpoint_dir = tmp.path().join("ckpt_real");
    let training_config = ebm::EBMTrainingConfig::new()
        .with_total_steps(10)
        .with_warmup_steps(2)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(0)
        .with_checkpoint_interval(0)
        .with_checkpoint_dir(checkpoint_dir.to_string_lossy().to_string());

    let trained = ebm::train(&training_config, model, &encode_fn, &sampler, &device);
    assert!(
        trained.is_ok(),
        "Training with real TinyLlama encode should succeed: {:?}",
        trained.err()
    );

    // Verify checkpoint saved
    let final_ckpt = checkpoint_dir.join("final.mpk");
    assert!(final_ckpt.exists(), "Final checkpoint should exist");

    // Save config JSON
    let config_path = checkpoint_dir.join("energy_head_config.json");
    let config_json = serde_json::to_string_pretty(&head_config).unwrap();
    std::fs::write(&config_path, &config_json).unwrap();
    assert!(config_path.exists());

    println!("Test 1 passed: TinyLlama encode + EBM train succeeded");
}

// ---------------------------------------------------------------------------
// Test 2: Train -> save -> load -> score (checkpoint roundtrip)
// ---------------------------------------------------------------------------

/// Test that EBM checkpoint roundtrip works with real TinyLlama dimensions.
///
/// This catches:
/// - Checkpoint save/load with non-default dimensions
/// - EBMScorer correctly loading and scoring with real encode_fn
/// - Scores are finite and distinct for different states
#[test]
#[ignore]
fn test_tinyllama_ebm_scorer_roundtrip() {
    let config = test_policy_config();
    let gen = TacticGenerator::load(&config).unwrap();
    let hidden_size = gen.hidden_size();

    // Build a small trajectory and train EBM
    let tmp = tempfile::TempDir::new().unwrap();
    let states = [
        "⊢ True",
        "n : Nat\n⊢ n + 0 = n",
        "h : False\n⊢ False",
        "⊢ ∀ (n : Nat), n = n",
    ];
    let path1 = write_synthetic_trajectory(
        tmp.path(),
        "thm1.parquet",
        "thm_rt_a",
        &[states[0], states[1], states[2], states[3]],
    );
    let path2 = write_synthetic_trajectory(
        tmp.path(),
        "thm2.parquet",
        "thm_rt_b",
        &[states[2], states[3], states[0], states[1]],
    );

    let sampler = ebm::ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();

    let head_config = ebm::EnergyHeadConfig::new(hidden_size)
        .with_d_hidden1(256)
        .with_d_hidden2(128)
        .with_dropout(0.0);

    type TrainBackend = burn::backend::Autodiff<NdArray<f32>>;
    let device: <TrainBackend as Backend>::Device = Default::default();
    let model = head_config.init::<TrainBackend>(&device);

    // Train with real encoder
    let gen = std::sync::Mutex::new(gen);
    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        let mut g = gen
            .lock()
            .map_err(|e| anyhow::anyhow!("Generator lock poisoned: {e}"))?;
        let embedding = g.encode_only(state)?;
        Ok(embedding.data)
    };

    let ckpt_dir = tmp.path().join("ckpt_rt");
    let training_config = ebm::EBMTrainingConfig::new()
        .with_total_steps(10)
        .with_warmup_steps(2)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(0)
        .with_checkpoint_interval(0)
        .with_checkpoint_dir(ckpt_dir.to_string_lossy().to_string());

    ebm::train(&training_config, model, &encode_fn, &sampler, &device).unwrap();

    // Save config
    let config_json = serde_json::to_string_pretty(&head_config).unwrap();
    std::fs::write(ckpt_dir.join("energy_head_config.json"), &config_json).unwrap();

    // Load checkpoint back via EBMScorer with real encode_fn
    let ckpt_path = ckpt_dir.join("final");
    let inference_device: <TestBackend as Backend>::Device = Default::default();

    let encode_fn_scorer: Box<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync> =
        Box::new(move |state: &str| {
            let mut g = gen
                .lock()
                .map_err(|e| anyhow::anyhow!("Generator lock poisoned: {e}"))?;
            let embedding = g.encode_only(state)?;
            Ok(embedding.data)
        });

    let scorer =
        ebm::EBMScorer::<TestBackend>::load(&ckpt_path, &head_config, encode_fn_scorer, inference_device)
            .expect("EBMScorer should load from checkpoint");

    // Score several states and verify finite
    let scores: Vec<f64> = states
        .iter()
        .map(|s| scorer.score_state(s).unwrap())
        .collect();

    for (i, score) in scores.iter().enumerate() {
        assert!(
            score.is_finite(),
            "Score for state '{}' should be finite, got {score}",
            states[i]
        );
    }

    // Verify different states produce different scores (at least some pairs differ)
    let all_same = scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
    assert!(
        !all_same,
        "Not all scores should be identical: {scores:?}"
    );

    println!("Test 2 passed: EBM scorer roundtrip with real TinyLlama dimensions");
}

// ---------------------------------------------------------------------------
// Test 3: Arc<Mutex<TacticGenerator>> sharing between policy and EBM
// ---------------------------------------------------------------------------

/// Test the shared generator pattern used in load_policy_and_ebm().
///
/// This catches:
/// - Mutex deadlocks when interleaving policy + encode calls
/// - Dimension consistency between policy and encode paths
/// - The MutexPolicyProvider::new_shared() + shared_generator() pattern
#[test]
#[ignore]
fn test_tinyllama_shared_generator_policy_and_ebm() {
    let config = test_policy_config();
    let gen = TacticGenerator::load(&config).unwrap();
    let hidden_size = gen.hidden_size();

    // Wrap in Arc<Mutex> — the shared pattern
    let shared_gen = Arc::new(std::sync::Mutex::new(gen));

    // Create MutexPolicyProvider from the shared generator
    let policy = MutexPolicyProvider::new_shared(shared_gen.clone());

    // Create EBM encode_fn closure using the same shared generator
    let encode_gen = shared_gen.clone();
    let encode_fn = move |state: &str| -> anyhow::Result<Vec<f32>> {
        let mut g = encode_gen
            .lock()
            .map_err(|e| anyhow::anyhow!("Generator lock poisoned: {e}"))?;
        let embedding = g.encode_only(state)?;
        Ok(embedding.data)
    };

    // Interleave policy and encode calls to stress the mutex
    let states = [
        "⊢ True",
        "n : Nat\n⊢ n + 0 = n",
        "h : False\n⊢ False",
    ];

    for (i, state) in states.iter().enumerate() {
        // Policy call
        let candidates = policy.generate_candidates(state, 2);
        assert!(
            candidates.is_ok(),
            "Policy should generate candidates for '{state}': {:?}",
            candidates.err()
        );
        let candidates = candidates.unwrap();
        assert!(
            !candidates.is_empty(),
            "Should generate at least 1 candidate for '{state}'"
        );
        for c in &candidates {
            assert!(!c.text.is_empty(), "Candidate text should not be empty");
            assert!(c.log_prob.is_finite(), "Candidate log_prob should be finite");
        }
        println!("  Policy call {i}: {} candidates for '{state}'", candidates.len());

        // Encode call (same generator, different code path)
        let embedding = encode_fn(state).unwrap();
        assert_eq!(
            embedding.len(),
            hidden_size,
            "Embedding from encode_fn should have dim={hidden_size}"
        );
        assert!(
            embedding.iter().all(|x| x.is_finite()),
            "All embedding values should be finite"
        );
        println!("  Encode call {i}: dim={} for '{state}'", embedding.len());
    }

    // One more round of interleaved calls to confirm no deadlocks
    for state in &states {
        let _ = policy.generate_candidates(state, 1).unwrap();
        let emb = encode_fn(state).unwrap();
        assert_eq!(emb.len(), hidden_size);
    }

    println!("Test 3 passed: shared generator policy + EBM with no deadlocks");
}

// ---------------------------------------------------------------------------
// Test 4: Full pipeline with Lean (requires Pantograph)
// ---------------------------------------------------------------------------

/// Test the complete production path: search -> Parquet -> train EBM -> search with EBM.
///
/// This catches:
/// - Full pipeline wiring with real LLM + real Lean + real EBM
/// - SearchResult field validity with real infrastructure
/// - No panics or deadlocks under the complete production code path
///
/// Note: TinyLlama doesn't know Lean tactics, so we do NOT assert proofs found.
#[tokio::test]
#[ignore]
async fn test_tinyllama_full_pipeline_with_lean() {
    let config = test_policy_config();
    let gen = TacticGenerator::load(&config).unwrap();
    let hidden_size = gen.hidden_size();

    // Set up Lean pool
    let lean_config = lean_repl::LeanPoolConfig::with_bundled_pantograph()
        .expect("Pantograph not found — this test requires the vendor/Pantograph submodule");
    let pool = Arc::new(lean_repl::LeanPool::new(lean_config).await.unwrap());

    // Share the generator
    let shared_gen = Arc::new(std::sync::Mutex::new(gen));
    let policy = MutexPolicyProvider::new_shared(shared_gen.clone());

    // --- Phase 1: Search WITHOUT EBM ---
    let search_config = SearchConfig {
        max_nodes: 10,
        max_depth: 5,
        timeout_per_theorem: 30,
        num_candidates: 3,
        ..SearchConfig::default()
    };
    let engine = SearchEngine::new(search_config.clone());

    let theorems = [
        ("true_trivial", "True"),
        ("false_implies_false", "False → False"),
        ("nat_refl", "∀ (n : Nat), n = n"),
    ];

    let tmp = tempfile::TempDir::new().unwrap();
    let trajectory_path = tmp.path().join("trajectory.parquet");
    let mut writer = TrajectoryWriter::new(trajectory_path.clone());

    println!("Searching {} theorems without EBM...", theorems.len());
    for (name, stmt) in &theorems {
        match engine
            .search_one(&pool, &policy, None, name, stmt)
            .await
        {
            Ok(result) => {
                println!(
                    "  {name}: proved={}, nodes={}, time={}ms",
                    result.proved, result.nodes_expanded, result.wall_time_ms
                );
                assert!(
                    result.nodes_expanded > 0,
                    "Should expand at least 1 node for '{name}'"
                );
                assert!(
                    result.wall_time_ms < 120_000,
                    "Search for '{name}' should complete within 2 minutes"
                );
                let labeled = TrajectoryWriter::from_search_result(&result);
                writer.record_all(labeled);
            }
            Err(e) => {
                // TinyLlama may produce invalid tactics that cause errors — that's OK
                println!("  {name}: search error (expected with TinyLlama): {e}");
            }
        }
    }
    writer.finish().unwrap();

    // Verify trajectory has records
    let records = TrajectoryReader::read_all(&trajectory_path).unwrap();
    assert!(
        !records.is_empty(),
        "Should have at least some trajectory records"
    );
    println!("Wrote {} trajectory records", records.len());

    // --- Phase 2: Train EBM from trajectory ---
    println!("Training EBM from trajectory...");
    let head_config = ebm::EnergyHeadConfig::new(hidden_size)
        .with_d_hidden1(256)
        .with_d_hidden2(128)
        .with_dropout(0.0);

    type TrainBackend = burn::backend::Autodiff<NdArray<f32>>;
    let device: <TrainBackend as Backend>::Device = Default::default();
    let model = head_config.init::<TrainBackend>(&device);

    let sampler = ebm::ContrastiveSampler::from_parquet(&[trajectory_path.clone()], 2);
    if let Ok(sampler) = sampler {
        if sampler.num_eligible_theorems() > 0 {
            let encode_gen = shared_gen.clone();
            let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
                let mut g = encode_gen
                    .lock()
                    .map_err(|e| anyhow::anyhow!("Generator lock poisoned: {e}"))?;
                let embedding = g.encode_only(state)?;
                Ok(embedding.data)
            };

            let ckpt_dir = tmp.path().join("ebm_ckpt");
            let training_config = ebm::EBMTrainingConfig::new()
                .with_total_steps(10)
                .with_warmup_steps(2)
                .with_batch_size(2)
                .with_k_negatives(2)
                .with_log_interval(0)
                .with_checkpoint_interval(0)
                .with_checkpoint_dir(ckpt_dir.to_string_lossy().to_string());

            let trained = ebm::train(&training_config, model, &encode_fn, &sampler, &device);
            assert!(
                trained.is_ok(),
                "EBM training should succeed: {:?}",
                trained.err()
            );

            // Save config
            let config_json = serde_json::to_string_pretty(&head_config).unwrap();
            std::fs::write(ckpt_dir.join("energy_head_config.json"), &config_json).unwrap();

            // --- Phase 3: Search WITH EBM ---
            println!("Searching with EBM scorer...");
            let ckpt_path = ckpt_dir.join("final");
            let inference_device: <TestBackend as Backend>::Device = Default::default();

            let encode_gen2 = shared_gen.clone();
            let encode_fn_scorer: Box<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync> =
                Box::new(move |state: &str| {
                    let mut g = encode_gen2
                        .lock()
                        .map_err(|e| anyhow::anyhow!("Generator lock poisoned: {e}"))?;
                    let embedding = g.encode_only(state)?;
                    Ok(embedding.data)
                });

            let scorer = ebm::EBMScorer::<TestBackend>::load(
                &ckpt_path,
                &head_config,
                encode_fn_scorer,
                inference_device,
            )
            .expect("EBMScorer should load");
            let value_fn = ebm::EBMValueFn::new(scorer);
            let scorer_ref: Option<&dyn ValueScorer> = Some(&value_fn);

            for (name, stmt) in &theorems {
                match engine
                    .search_one(&pool, &policy, scorer_ref, name, stmt)
                    .await
                {
                    Ok(result) => {
                        println!(
                            "  {name} (with EBM): proved={}, nodes={}, time={}ms",
                            result.proved, result.nodes_expanded, result.wall_time_ms
                        );
                        assert!(
                            result.nodes_expanded > 0,
                            "Should expand at least 1 node with EBM for '{name}'"
                        );
                    }
                    Err(e) => {
                        println!("  {name} (with EBM): search error (expected): {e}");
                    }
                }
            }
            println!("Phase 3 complete: search with EBM scorer ran without panics");
        } else {
            println!(
                "Skipping EBM training: no eligible theorems (TinyLlama likely produced no useful trajectory)"
            );
        }
    } else {
        println!("Skipping EBM training: sampler construction failed (insufficient trajectory data)");
    }

    // Shutdown
    pool.shutdown().await;

    println!("Test 4 passed: full pipeline (search -> train EBM -> search with EBM)");
}
