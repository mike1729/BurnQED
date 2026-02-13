//! Integration tests for the EBM crate.
//!
//! These tests exercise cross-module interactions: EnergyHead + loss + optimizer,
//! bridge -> model -> extraction pipeline, Parquet -> ContrastiveSampler pipeline,
//! full training step simulations, and end-to-end train/checkpoint/resume flows.
//! All use NdArray backend and synthetic data — no Lean, no LLM model needed.

use std::sync::atomic::{AtomicUsize, Ordering};

use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::{Distribution, TensorData};
use tempfile::TempDir;

use ebm::bridge::{embeddings_to_tensor, tensor_to_vec};
use ebm::model::energy_head::EnergyHeadConfig;

use ebm::training::data::ContrastiveSampler;
use ebm::training::loss::{depth_regression_loss, info_nce_loss};
use ebm::training::metrics::{EBMMetrics, MetricsHistory};
use ebm::training::trainer::{resume_from_checkpoint, train, EBMTrainingConfig};

// Re-export imports (test 18 verifies these work)
use ebm::{
    ContrastiveSampler as ContrastiveSamplerReexport,
    EBMScorer, EBMTrainingConfig as EBMTrainingConfigReexport,
    EBMValueFn, EmbeddingCache, EnergyHead as EnergyHeadReexport,
    EnergyHeadConfig as EnergyHeadConfigReexport,
    EncoderBackend, SpectralNormLinearConfig,
    lr_schedule, MetricsHistory as MetricsHistoryReexport,
};
use trajectory::{
    SearchResult, SearchStats, TrajectoryLabel, TrajectoryRecord, TrajectoryWriter,
};

type TestBackend = NdArray<f32>;
type TestAutodiffBackend = Autodiff<NdArray<f32>>;

/// Helper: create a trajectory record for test data.
fn make_record(
    theorem: &str,
    state_id: u64,
    label: TrajectoryLabel,
    depth: u32,
    remaining: i32,
    is_complete: bool,
    parent: Option<u64>,
) -> TrajectoryRecord {
    TrajectoryRecord {
        theorem_name: theorem.to_string(),
        state_id,
        state_pp: format!("⊢ state_{theorem}_{state_id}"),
        tactic_applied: if state_id == 0 {
            String::new()
        } else {
            format!("tactic_{state_id}")
        },
        parent_state_id: parent,
        label,
        depth_from_root: depth,
        remaining_depth: remaining,
        llm_log_prob: -0.5,
        ebm_score: 0.0,
        is_proof_complete: is_complete,
        timestamp_ms: 1700000000000 + state_id,
    }
}

/// Helper: write a proved theorem's trajectory to Parquet.
/// Tree: root(0) -> n1(1) -> n2(2, QED) + n3(3, dead end)
fn write_proved_theorem(dir: &std::path::Path, filename: &str, theorem: &str) -> std::path::PathBuf {
    let path = dir.join(filename);

    // Create search result with proof tree
    let root = make_record(theorem, 0, TrajectoryLabel::Unknown, 0, -1, false, None);
    let n1 = make_record(theorem, 1, TrajectoryLabel::Unknown, 1, -1, false, Some(0));
    let n2 = make_record(theorem, 2, TrajectoryLabel::Unknown, 2, -1, true, Some(1));
    let n3 = make_record(theorem, 3, TrajectoryLabel::Unknown, 1, -1, false, Some(0));

    let result = SearchResult {
        theorem_name: theorem.to_string(),
        proved: true,
        proof_tactics: vec!["tactic_1".into(), "tactic_2".into()],
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

/// Helper: write an unproved theorem's trajectory to Parquet.
fn write_unproved_theorem(
    dir: &std::path::Path,
    filename: &str,
    theorem: &str,
) -> std::path::PathBuf {
    let path = dir.join(filename);

    let root = make_record(theorem, 0, TrajectoryLabel::Unknown, 0, -1, false, None);
    let n1 = make_record(theorem, 1, TrajectoryLabel::Unknown, 1, -1, false, Some(0));
    let n2 = make_record(theorem, 2, TrajectoryLabel::Unknown, 1, -1, false, Some(0));

    let result = SearchResult {
        theorem_name: theorem.to_string(),
        proved: false,
        proof_tactics: vec![],
        nodes_expanded: 3,
        total_states: 3,
        max_depth_reached: 1,
        wall_time_ms: 200,
        all_records: vec![root, n1, n2],
        stats: SearchStats::default(),
    };

    let labeled = TrajectoryWriter::from_search_result(&result);
    let mut writer = TrajectoryWriter::new(path.clone());
    writer.record_all(labeled);
    writer.finish().unwrap();
    path
}

// ---------------------------------------------------------------------------
// Test 1: EnergyHead end-to-end forward + backward (training step simulation)
// ---------------------------------------------------------------------------

#[test]
fn test_energy_head_forward_backward_training_step() {
    let device = Default::default();
    let model = EnergyHeadConfig::new(32)
        .with_d_hidden1(16)
        .with_d_hidden2(8)
        .with_dropout(0.0)
        .init::<TestAutodiffBackend>(&device);

    let mut optim = AdamConfig::new().init();

    // Create synthetic positive (low mean) and negative (high mean) embeddings
    let pos_emb = Tensor::<TestAutodiffBackend, 2>::random(
        [4, 32],
        Distribution::Normal(-2.0, 0.5),
        &device,
    );
    let neg_emb = Tensor::<TestAutodiffBackend, 2>::random(
        [4, 32],
        Distribution::Normal(2.0, 0.5),
        &device,
    );

    // Forward pass
    let pos_energy = model.forward(pos_emb.clone());
    // For negatives, we need (batch, K) shape — K=1 here
    let neg_energy_flat = model.forward(neg_emb.clone());
    let neg_energies = neg_energy_flat.unsqueeze_dim::<2>(1); // (4, 1)

    // Compute loss
    let loss = info_nce_loss(pos_energy, neg_energies);
    let loss_val_before: f32 = loss.clone().into_scalar().elem();
    assert!(loss_val_before.is_finite(), "Initial loss should be finite");

    // Backward + optimizer step
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let model = optim.step(0.001.into(), model, grads);

    // Forward again — loss should decrease (or at least be finite)
    let pos_energy2 = model.forward(pos_emb);
    let neg_energy_flat2 = model.forward(neg_emb);
    let neg_energies2 = neg_energy_flat2.unsqueeze_dim::<2>(1);
    let loss2 = info_nce_loss(pos_energy2, neg_energies2);
    let loss_val_after: f32 = loss2.into_scalar().elem();

    assert!(
        loss_val_after.is_finite(),
        "Loss after step should be finite"
    );
    // With well-separated data and Adam, one step should reduce loss
    assert!(
        loss_val_after < loss_val_before + 0.1,
        "Loss should not increase substantially: before={loss_val_before}, after={loss_val_after}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: EnergyHead with realistic batch sizes (d_encoder=4096)
// ---------------------------------------------------------------------------

#[test]
fn test_energy_head_realistic_dimensions() {
    let device = Default::default();
    let model = EnergyHeadConfig::new(4096).init::<TestBackend>(&device);

    let input = Tensor::<TestBackend, 2>::random(
        [64, 4096],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let output = model.forward(input);

    assert_eq!(output.dims(), [64], "Output should be (batch=64,)");

    // Verify all outputs are finite
    let data: Vec<f32> = output.into_data().to_vec().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "Output[{i}] is not finite: {v}");
    }
}

// ---------------------------------------------------------------------------
// Test 3: Bridge -> EnergyHead pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_bridge_to_energy_head_pipeline() {
    let device = Default::default();
    let model = EnergyHeadConfig::new(32)
        .with_d_hidden1(16)
        .with_d_hidden2(8)
        .with_dropout(0.0)
        .init::<TestBackend>(&device);

    // Simulate LLM encoder output: Vec<Vec<f32>>
    let embeddings: Vec<Vec<f32>> = (0..8)
        .map(|i| {
            (0..32)
                .map(|j| (i as f32 * 0.1 + j as f32 * 0.01).sin())
                .collect()
        })
        .collect();

    // Bridge: Vec<Vec<f32>> -> burn Tensor
    let tensor = embeddings_to_tensor::<TestBackend>(&embeddings, &device);
    assert_eq!(tensor.dims(), [8, 32]);

    // Forward through EnergyHead
    let energies = model.forward(tensor);
    assert_eq!(energies.dims(), [8]);

    // Extract: burn Tensor -> Vec<f64>
    let energy_vec = tensor_to_vec::<TestBackend>(energies);
    assert_eq!(energy_vec.len(), 8);

    // Verify all values are finite and distinct
    for (i, &v) in energy_vec.iter().enumerate() {
        assert!(v.is_finite(), "Energy[{i}] is not finite: {v}");
    }
    // At least some distinct values (different inputs should produce different energies)
    let unique: std::collections::HashSet<u64> = energy_vec
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert!(
        unique.len() > 1,
        "Expected distinct energies for different inputs"
    );
}

// ---------------------------------------------------------------------------
// Test 4: ContrastiveSampler::from_parquet() roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_contrastive_sampler_from_parquet() {
    let tmp = TempDir::new().unwrap();

    // Write a proved theorem (has both positive and negative records)
    let path = write_proved_theorem(tmp.path(), "proved.parquet", "thm_proved");

    // Load via ContrastiveSampler
    let sampler = ContrastiveSampler::from_parquet(&[path], 2).unwrap();

    // Proved theorem: root(0)+n1(1)+n2(2) on proof path = 3 positive, n3 = 1 negative
    // Total records after filtering "unknown" labels: 4 (3 positive + 1 negative)
    assert_eq!(sampler.num_records(), 4, "Should have 4 records (3 pos + 1 neg)");
    assert_eq!(
        sampler.num_eligible_theorems(),
        1,
        "Should have 1 eligible theorem"
    );

    // Sample and verify structure
    let mut rng = rand::thread_rng();
    let sample = sampler.sample(&mut rng);
    assert_eq!(sample.negatives.len(), 2, "Should have exactly 2 negatives");
    assert_eq!(sample.positive.label, "positive");
}

// ---------------------------------------------------------------------------
// Test 5: Full training step from Parquet data
// ---------------------------------------------------------------------------

#[test]
fn test_full_training_step_from_parquet() {
    let tmp = TempDir::new().unwrap();

    // Write proved + unproved theorems to get both positive and negative records
    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_alpha");
    let path2 = write_unproved_theorem(tmp.path(), "thm2.parquet", "thm_beta");

    // Load data
    let sampler = ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();
    assert!(
        sampler.num_eligible_theorems() >= 1,
        "Need at least 1 eligible theorem"
    );

    // Sample a batch
    let mut rng = rand::thread_rng();
    let batch = sampler.sample_batch(4, &mut rng);
    assert_eq!(batch.len(), 4);

    let device = Default::default();
    let model = EnergyHeadConfig::new(16)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0)
        .init::<TestAutodiffBackend>(&device);
    let mut optim = AdamConfig::new().init();

    // Create mock embeddings for each sample (in real code, encoder.encode_only())
    let batch_size = batch.len();
    let k = 2;

    // Positive embeddings: (batch_size, d_encoder=16)
    let pos_emb = Tensor::<TestAutodiffBackend, 2>::random(
        [batch_size, 16],
        Distribution::Normal(-1.0, 0.5),
        &device,
    );
    // Negative embeddings: (batch_size * K, d_encoder=16)
    let neg_emb = Tensor::<TestAutodiffBackend, 2>::random(
        [batch_size * k, 16],
        Distribution::Normal(1.0, 0.5),
        &device,
    );

    // Forward
    let pos_energy = model.forward(pos_emb);
    let neg_energy_flat = model.forward(neg_emb);
    let neg_energies = neg_energy_flat.reshape([batch_size, k]); // (batch, K)

    // Remaining depths from batch samples
    let remaining: Vec<f32> = batch
        .iter()
        .map(|s| s.remaining_depth as f32)
        .collect();
    let remaining_tensor = Tensor::<TestAutodiffBackend, 1>::from_data(
        TensorData::new(remaining, [batch_size]),
        &device,
    );

    // Compute losses
    let contrastive = info_nce_loss(pos_energy.clone(), neg_energies);
    let depth = depth_regression_loss(pos_energy, remaining_tensor);
    let total_loss = contrastive + depth;

    let loss_val: f32 = total_loss.clone().into_scalar().elem();
    assert!(loss_val.is_finite(), "Total loss should be finite: {loss_val}");

    // Backward + optimizer step
    let grads = GradientsParams::from_grads(total_loss.backward(), &model);
    let _model = optim.step(0.001.into(), model, grads);

    // If we got here without panicking, the full pipeline works
}

// ---------------------------------------------------------------------------
// Test 6: EBMMetrics over simulated training
// ---------------------------------------------------------------------------

#[test]
fn test_metrics_history_simulated_training() {
    let device = Default::default();
    let mut history = MetricsHistory::new();

    // Simulate 10 training steps with improving separation
    for step in 0..10 {
        let gap = step as f32 * 0.5; // increasing gap
        let pos_val = -(gap / 2.0);
        let neg_val = gap / 2.0;

        let pos = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([pos_val, pos_val, pos_val, pos_val]),
            &device,
        );
        let neg = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [neg_val, neg_val],
                [neg_val, neg_val],
                [neg_val, neg_val],
                [neg_val, neg_val],
            ]),
            &device,
        );

        let contrastive_loss = 1.0 / (1.0 + step as f64);
        let metrics = EBMMetrics::compute(
            &pos,
            &neg,
            contrastive_loss,
            0.01,
            contrastive_loss + 0.01,
        );

        history.push(step, metrics);
    }

    assert_eq!(history.len(), 10);

    // Gap should be improving over the last 5 entries
    assert!(
        history.is_improving(5),
        "Energy gap should be trending upward"
    );

    // Last entry should have no health check warnings (good separation)
    let last = history.last().unwrap();
    let warnings = last.health_check();
    assert!(
        warnings.is_empty(),
        "Well-separated data should produce no warnings, got: {:?}",
        warnings
    );

    // Verify the gap is positive and rank_accuracy is 1.0
    assert!(last.energy_gap > 0.0, "Gap should be positive");
    assert!(
        (last.rank_accuracy - 1.0).abs() < 1e-6,
        "Perfect separation should give rank_accuracy=1.0, got {}",
        last.rank_accuracy
    );
}

// ---------------------------------------------------------------------------
// Test 7: Multi-file Parquet loading into ContrastiveSampler
// ---------------------------------------------------------------------------

#[test]
fn test_multi_file_parquet_loading() {
    let tmp = TempDir::new().unwrap();

    // Write 3 separate Parquet files with different theorems
    let path1 = write_proved_theorem(tmp.path(), "file1.parquet", "thm_one");
    let path2 = write_proved_theorem(tmp.path(), "file2.parquet", "thm_two");
    let path3 = write_proved_theorem(tmp.path(), "file3.parquet", "thm_three");

    // Load all via from_parquet
    let sampler = ContrastiveSampler::from_parquet(&[path1, path2, path3], 3).unwrap();

    // Each proved theorem contributes 4 records (3 pos + 1 neg), 3 files = 12 total
    assert_eq!(sampler.num_records(), 12, "Should have 12 records from 3 files");
    assert_eq!(
        sampler.num_eligible_theorems(),
        3,
        "Should have 3 eligible theorems"
    );

    // Sample many times and verify negatives can come from different theorems
    let mut rng = rand::thread_rng();
    let mut seen_theorems = std::collections::HashSet::new();
    for _ in 0..100 {
        let sample = sampler.sample(&mut rng);
        seen_theorems.insert(sample.positive.theorem_name.clone());
        for neg in &sample.negatives {
            seen_theorems.insert(neg.theorem_name.clone());
        }
    }
    assert!(
        seen_theorems.len() == 3,
        "Should sample from all 3 theorems, saw: {:?}",
        seen_theorems
    );
}

// ---------------------------------------------------------------------------
// Test 8: Spectral norm Lipschitz constraint preservation during training
// ---------------------------------------------------------------------------

#[test]
fn test_spectral_norm_lipschitz_during_training() {
    let device = Default::default();

    // Create a small EnergyHead (which stacks 3 SpectralNormLinear layers)
    let model = EnergyHeadConfig::new(32)
        .with_d_hidden1(16)
        .with_d_hidden2(8)
        .with_dropout(0.0)
        .init::<TestAutodiffBackend>(&device);
    let mut optim = AdamConfig::new().init();
    let mut current_model = model;

    // Run 5 training steps
    for step in 0..5 {
        let input = Tensor::<TestAutodiffBackend, 2>::random(
            [16, 32],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = current_model.forward(input.clone());
        let loss = output.powf_scalar(2.0).mean();

        let grads = GradientsParams::from_grads(loss.backward(), &current_model);
        current_model = optim.step(0.01.into(), current_model, grads);

        // After each step, verify spectral norm bound still holds.
        // Probe with Autodiff backend (forward-only, no backward needed).
        let probe = Tensor::<TestAutodiffBackend, 2>::random(
            [100, 32],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let probe_output = current_model.forward(probe.clone());

        // Check input/output norm ratio.
        // Input norms: (100, 1) after sum_dim(1).
        let input_norms = probe.powf_scalar(2.0).sum_dim(1).sqrt(); // (100, 1)
        // Output is (100,) 1D — unsqueeze to (100, 1) for element-wise division.
        let output_abs = probe_output.abs().unsqueeze_dim::<2>(1); // (100, 1)
        let ratios = output_abs / input_norms;
        let max_ratio: f32 = ratios.max().into_scalar().elem();

        // With 3 stacked spectral-normed layers (each ≤ 1.0) plus SiLU,
        // the overall Lipschitz constant should be bounded.
        // SiLU has Lipschitz constant ~1.1, so 3 layers: ~1.1^2 * 1.0^3 ≈ 1.21 + margin
        assert!(
            max_ratio < 3.0,
            "Step {step}: max amplification ratio {max_ratio} too high — spectral norm violated"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 9: Loss gradient direction verification
// ---------------------------------------------------------------------------

#[test]
fn test_loss_gradient_direction() {
    let device = Default::default();
    let model = EnergyHeadConfig::new(16)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0)
        .init::<TestAutodiffBackend>(&device);
    let mut optim = AdamConfig::new().init();

    // Fixed embeddings for positive and negative states.
    // We'll measure how energies change after training.
    let pos_data = TensorData::new(vec![1.0_f32; 64], [4, 16]);
    let neg_data = TensorData::new(vec![-1.0_f32; 64], [4, 16]);

    // Measure energies BEFORE training (forward-only, no backward)
    let pos_before: Vec<f32> = {
        let out = model.forward(
            Tensor::<TestAutodiffBackend, 2>::from_data(pos_data.clone(), &device),
        );
        out.into_data().to_vec().unwrap()
    };
    let neg_before: Vec<f32> = {
        let out = model.forward(
            Tensor::<TestAutodiffBackend, 2>::from_data(neg_data.clone(), &device),
        );
        out.into_data().to_vec().unwrap()
    };

    // Take a few gradient steps with InfoNCE loss
    let mut current_model = model;
    for _ in 0..5 {
        let pos_emb = Tensor::<TestAutodiffBackend, 2>::from_data(pos_data.clone(), &device);
        let neg_emb = Tensor::<TestAutodiffBackend, 2>::from_data(neg_data.clone(), &device);

        let pe = current_model.forward(pos_emb);
        let ne = current_model.forward(neg_emb);
        let ne_2d = ne.unsqueeze_dim::<2>(1); // (4, 1)
        let loss = info_nce_loss(pe, ne_2d);

        let grads = GradientsParams::from_grads(loss.backward(), &current_model);
        current_model = optim.step(0.01.into(), current_model, grads);
    }

    // Measure energies AFTER training
    let pos_after: Vec<f32> = {
        let out = current_model.forward(
            Tensor::<TestAutodiffBackend, 2>::from_data(pos_data, &device),
        );
        out.into_data().to_vec().unwrap()
    };
    let neg_after: Vec<f32> = {
        let out = current_model.forward(
            Tensor::<TestAutodiffBackend, 2>::from_data(neg_data, &device),
        );
        out.into_data().to_vec().unwrap()
    };

    // After training, the gap between neg and pos should be larger
    let gap_before: f32 =
        neg_before.iter().sum::<f32>() / 4.0 - pos_before.iter().sum::<f32>() / 4.0;
    let gap_after: f32 =
        neg_after.iter().sum::<f32>() / 4.0 - pos_after.iter().sum::<f32>() / 4.0;

    assert!(
        gap_after > gap_before,
        "After training, energy gap should increase: before={gap_before:.4}, after={gap_after:.4}"
    );
}

// ---------------------------------------------------------------------------
// Test 10: Full train() loop with mock encoder (happy path)
// ---------------------------------------------------------------------------

#[test]
fn test_train_small_loop() {
    let tmp = TempDir::new().unwrap();

    // Write trajectory data: 2 proved theorems for pos+neg records
    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_x");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_y");
    let sampler = ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();

    let device = Default::default();
    let d_encoder = 16;
    let model = EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0)
        .init::<TestAutodiffBackend>(&device);

    // Mock encode_fn: deterministic embedding based on string hash
    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        let hash = state.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        let base = (hash as f32) / 1000.0;
        Ok((0..d_encoder).map(|i| (base + i as f32 * 0.1).sin()).collect())
    };

    let checkpoint_dir = tmp.path().join("ckpt");
    let config = EBMTrainingConfig::new()
        .with_total_steps(10)
        .with_warmup_steps(2)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(5)
        .with_checkpoint_interval(0) // disable mid-training checkpoints
        .with_checkpoint_dir(checkpoint_dir.to_string_lossy().to_string());

    let trained = train(&config, model, &encode_fn, &sampler, &device);
    assert!(trained.is_ok(), "train() should succeed: {:?}", trained.err());

    let trained_model = trained.unwrap();

    // Verify trained model still produces finite outputs
    let probe = embeddings_to_tensor::<TestAutodiffBackend>(
        &[vec![0.5_f32; d_encoder], vec![-0.5_f32; d_encoder]],
        &device,
    );
    let energies = trained_model.forward(probe);
    let vals: Vec<f32> = energies.into_data().to_vec().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "Energy[{i}] is not finite after training: {v}");
    }

    // Verify final checkpoint was saved
    let final_path = checkpoint_dir.join("final.mpk");
    assert!(
        final_path.exists(),
        "Final checkpoint should exist at {final_path:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 11: Checkpoint save then resume_from_checkpoint() roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_checkpoint_save_and_resume() {
    let tmp = TempDir::new().unwrap();

    // Write trajectory data
    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_ckpt");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_ckpt2");
    let sampler = ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();

    let device = Default::default();
    let d_encoder = 16;
    let head_config = EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0);
    let model = head_config.init::<TestAutodiffBackend>(&device);

    let encode_fn = |_state: &str| -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.42_f32; d_encoder])
    };

    let checkpoint_dir = tmp.path().join("ckpt_roundtrip");
    let config = EBMTrainingConfig::new()
        .with_total_steps(5)
        .with_warmup_steps(1)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(0)  // disable logging
        .with_checkpoint_interval(0) // disable mid-training checkpoints
        .with_checkpoint_dir(checkpoint_dir.to_string_lossy().to_string());

    // Train and save
    let trained = train(&config, model, &encode_fn, &sampler, &device).unwrap();

    // Get output from trained model on a fixed input
    let probe_data = vec![0.1_f32; d_encoder];
    let probe_tensor = embeddings_to_tensor::<TestAutodiffBackend>(&[probe_data.clone()], &device);
    let energy_before: f32 = trained.forward(probe_tensor).into_scalar().elem();

    // Resume from the saved final checkpoint
    let final_ckpt = checkpoint_dir.join("final");
    let loaded: ebm::model::energy_head::EnergyHead<TestBackend> =
        resume_from_checkpoint(&final_ckpt, &head_config, &device).unwrap();

    // Get output from loaded model on the same input
    let probe_tensor2 = embeddings_to_tensor::<TestBackend>(&[probe_data], &device);
    let energy_after: f32 = loaded.forward(probe_tensor2).into_scalar().elem();

    // Outputs should be close. Not exact because SpectralNormLinear uses random
    // u/v vectors (Option C reinit) on each forward call, so different backend
    // instantiations may have slightly different spectral norm estimates.
    assert!(
        (energy_before - energy_after).abs() < 0.1,
        "Loaded model should produce similar energy: trained={energy_before}, loaded={energy_after}"
    );
}

// ---------------------------------------------------------------------------
// Test 12: train() recovers from encode_fn failures
// ---------------------------------------------------------------------------

#[test]
fn test_train_encode_failure_recovery() {
    let tmp = TempDir::new().unwrap();

    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_fail");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_fail2");
    let sampler = ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();

    let device = Default::default();
    let d_encoder = 16;
    let model = EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0)
        .init::<TestAutodiffBackend>(&device);

    // encode_fn that fails on odd-numbered calls
    let call_count = AtomicUsize::new(0);
    let encode_fn = |_state: &str| -> anyhow::Result<Vec<f32>> {
        let n = call_count.fetch_add(1, Ordering::SeqCst);
        if n % 3 == 1 {
            anyhow::bail!("Simulated encoding failure at call {n}")
        }
        Ok(vec![0.5_f32; d_encoder])
    };

    let checkpoint_dir = tmp.path().join("ckpt_fail");
    let config = EBMTrainingConfig::new()
        .with_total_steps(10)
        .with_warmup_steps(0)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(0)
        .with_checkpoint_interval(0)
        .with_checkpoint_dir(checkpoint_dir.to_string_lossy().to_string());

    // Should NOT panic — skips failed steps and continues
    let result = train(&config, model, &encode_fn, &sampler, &device);
    assert!(
        result.is_ok(),
        "train() should handle encode failures gracefully: {:?}",
        result.err()
    );

    // Verify encode_fn was actually called (some calls succeeded, some failed)
    let total_calls = call_count.load(Ordering::SeqCst);
    assert!(
        total_calls > 0,
        "encode_fn should have been called at least once"
    );
}

// ---------------------------------------------------------------------------
// Test 13: depth_loss_weight=0 vs depth_loss_weight=1 produces different models
// ---------------------------------------------------------------------------

#[test]
fn test_train_depth_loss_weight_effect() {
    let tmp = TempDir::new().unwrap();

    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_dw");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_dw2");
    let sampler = ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();

    let device = Default::default();
    let d_encoder = 16;

    // Deterministic encode_fn for reproducibility
    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        let hash = state.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        let base = (hash as f32) / 500.0;
        Ok((0..d_encoder).map(|i| (base + i as f32 * 0.05).sin()).collect())
    };

    // Train with depth_loss_weight = 0 (contrastive only)
    let ckpt_dir_0 = tmp.path().join("ckpt_dw0");
    let config_no_depth = EBMTrainingConfig::new()
        .with_total_steps(20)
        .with_warmup_steps(2)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_depth_loss_weight(0.0)
        .with_log_interval(0)
        .with_checkpoint_interval(0)
        .with_checkpoint_dir(ckpt_dir_0.to_string_lossy().to_string());

    let model_0 = EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0)
        .init::<TestAutodiffBackend>(&device);
    let trained_0 = train(&config_no_depth, model_0, &encode_fn, &sampler, &device).unwrap();

    // Train with depth_loss_weight = 5.0 (heavy depth regression)
    let ckpt_dir_5 = tmp.path().join("ckpt_dw5");
    let config_heavy_depth = EBMTrainingConfig::new()
        .with_total_steps(20)
        .with_warmup_steps(2)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_depth_loss_weight(5.0)
        .with_log_interval(0)
        .with_checkpoint_interval(0)
        .with_checkpoint_dir(ckpt_dir_5.to_string_lossy().to_string());

    let model_5 = EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0)
        .init::<TestAutodiffBackend>(&device);
    let trained_5 = train(&config_heavy_depth, model_5, &encode_fn, &sampler, &device).unwrap();

    // Compare outputs on same probe input — different weight configs should diverge
    let probe = embeddings_to_tensor::<TestAutodiffBackend>(
        &[vec![0.3_f32; d_encoder], vec![-0.3_f32; d_encoder]],
        &device,
    );

    let e0: Vec<f32> = trained_0.forward(probe.clone()).into_data().to_vec().unwrap();
    let e5: Vec<f32> = trained_5.forward(probe).into_data().to_vec().unwrap();

    // At least one energy value should differ between the two models
    let max_diff: f32 = e0.iter().zip(e5.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(
        max_diff > 1e-6,
        "Models trained with different depth_loss_weight should produce different energies, max_diff={max_diff}"
    );
}

// ---------------------------------------------------------------------------
// Test 14: EBMScorer::load() from checkpoint end-to-end
// ---------------------------------------------------------------------------

#[test]
fn test_ebm_scorer_load_from_checkpoint() {
    let tmp = TempDir::new().unwrap();

    // Train a small model and save
    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_scorer");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_scorer2");
    let sampler = ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();

    let device = Default::default();
    let d_encoder = 16;
    let head_config = EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0);
    let model = head_config.init::<TestAutodiffBackend>(&device);

    let make_encode_fn = || {
        let dim = 16;
        move |state: &str| -> anyhow::Result<Vec<f32>> {
            let hash = state.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
            Ok((0..dim).map(|i| ((hash as f32 + i as f32) * 0.1).sin()).collect())
        }
    };
    let encode_fn = make_encode_fn();

    let checkpoint_dir = tmp.path().join("ckpt_scorer");
    let config = EBMTrainingConfig::new()
        .with_total_steps(5)
        .with_warmup_steps(1)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(0)
        .with_checkpoint_interval(0)
        .with_checkpoint_dir(checkpoint_dir.to_string_lossy().to_string());

    train(&config, model, &encode_fn, &sampler, &device).unwrap();

    // Load via EBMScorer::load (inference backend, not autodiff)
    let final_ckpt = checkpoint_dir.join("final");
    let scorer: EBMScorer<TestBackend> = EBMScorer::load(
        &final_ckpt,
        &head_config,
        Box::new(make_encode_fn()),
        Default::default(),
    )
    .unwrap();

    // score_state should return a finite value
    let score = scorer.score_state("⊢ True").unwrap();
    assert!(score.is_finite(), "Score should be finite, got {score}");

    // score_states batch should match length
    let states = ["⊢ True", "n : Nat\n⊢ n = n", "⊢ False → False"];
    let scores = scorer.score_states(&states).unwrap();
    assert_eq!(scores.len(), 3);
    for (i, s) in scores.iter().enumerate() {
        assert!(s.is_finite(), "Batch score[{i}] should be finite, got {s}");
    }

    // Empty batch
    let empty: Vec<f64> = scorer.score_states(&[]).unwrap();
    assert!(empty.is_empty());
}

// ---------------------------------------------------------------------------
// Test 15: EBMValueFn backend-erased wrapper
// ---------------------------------------------------------------------------

#[test]
fn test_ebm_value_fn_backend_erased() {
    let device: <TestBackend as Backend>::Device = Default::default();
    let head = EnergyHeadConfig::new(8)
        .with_d_hidden1(4)
        .with_d_hidden2(2)
        .with_dropout(0.0)
        .init::<TestBackend>(&device);

    let encode_fn: Box<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync> =
        Box::new(|state: &str| {
            let mut emb = vec![0.0_f32; 8];
            for (i, byte) in state.bytes().enumerate() {
                emb[i % 8] += byte as f32 / 255.0;
            }
            Ok(emb)
        });

    let scorer = EBMScorer::new(head, encode_fn, device);
    let value_fn = EBMValueFn::new(scorer);

    // Score via backend-erased interface
    let s1 = value_fn.score("⊢ True").unwrap();
    let s2 = value_fn.score("n : Nat\nhyp : n > 0\n⊢ n + 1 > 1").unwrap();

    assert!(s1.is_finite(), "Score 1 should be finite");
    assert!(s2.is_finite(), "Score 2 should be finite");

    // Different inputs should produce different scores
    assert!(
        (s1 - s2).abs() > 1e-6,
        "Different states should produce different scores: {s1} vs {s2}"
    );

    // Calling score multiple times on the same input should work (mutex doesn't deadlock)
    for _ in 0..10 {
        let s = value_fn.score("⊢ True").unwrap();
        assert!(s.is_finite());
    }
}

// ---------------------------------------------------------------------------
// Test 16: EncoderBackend hidden_dim sizes EnergyHead correctly
// ---------------------------------------------------------------------------

#[test]
fn test_encoder_backend_sizes_energy_head() {
    let device = Default::default();

    // Shared backend with default dim
    let shared = EncoderBackend::default();
    assert_eq!(shared.hidden_dim(), 4096);

    // Use hidden_dim to construct a matching EnergyHead
    let small_dim = 32;
    let backend = EncoderBackend::Shared { hidden_dim: small_dim };
    let head = EnergyHeadConfig::new(backend.hidden_dim())
        .with_d_hidden1(16)
        .with_d_hidden2(8)
        .with_dropout(0.0)
        .init::<TestBackend>(&device);

    // Forward pass with matching-dimension input should succeed
    let input = Tensor::<TestBackend, 2>::random(
        [4, small_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let output = head.forward(input);
    assert_eq!(output.dims(), [4]);

    // Dedicated variant
    let dedicated = EncoderBackend::Dedicated { hidden_dim: 1024 };
    let head2 = EnergyHeadConfig::new(dedicated.hidden_dim())
        .with_d_hidden1(64)
        .with_d_hidden2(32)
        .with_dropout(0.0)
        .init::<TestBackend>(&device);

    let input2 = Tensor::<TestBackend, 2>::random(
        [2, 1024],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let output2 = head2.forward(input2);
    assert_eq!(output2.dims(), [2]);

    // TOML roundtrip → correct dim
    let toml_str = r#"
type = "Shared"
hidden_dim = 64
"#;
    let parsed: EncoderBackend = toml::from_str(toml_str).unwrap();
    assert_eq!(parsed.hidden_dim(), 64);
    let head3 = EnergyHeadConfig::new(parsed.hidden_dim())
        .with_d_hidden1(16)
        .with_d_hidden2(8)
        .with_dropout(0.0)
        .init::<TestBackend>(&device);
    let input3 = Tensor::<TestBackend, 2>::random(
        [1, 64],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let output3 = head3.forward(input3);
    assert_eq!(output3.dims(), [1]);
}

// ---------------------------------------------------------------------------
// Test 17: Mid-training checkpoint roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_mid_training_checkpoint_roundtrip() {
    let tmp = TempDir::new().unwrap();

    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_midckpt");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_midckpt2");
    let sampler = ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();

    let device = Default::default();
    let d_encoder = 16;
    let head_config = EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0);
    let model = head_config.init::<TestAutodiffBackend>(&device);

    let encode_fn = |_state: &str| -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.3_f32; d_encoder])
    };

    let checkpoint_dir = tmp.path().join("ckpt_mid");
    let config = EBMTrainingConfig::new()
        .with_total_steps(12)
        .with_warmup_steps(1)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(0)
        .with_checkpoint_interval(5) // save at step 5 and 10
        .with_checkpoint_dir(checkpoint_dir.to_string_lossy().to_string());

    train(&config, model, &encode_fn, &sampler, &device).unwrap();

    // Verify mid-training checkpoints exist
    let step5_path = checkpoint_dir.join("step_5.mpk");
    let step10_path = checkpoint_dir.join("step_10.mpk");
    let final_path = checkpoint_dir.join("final.mpk");

    assert!(step5_path.exists(), "Step 5 checkpoint should exist at {step5_path:?}");
    assert!(step10_path.exists(), "Step 10 checkpoint should exist at {step10_path:?}");
    assert!(final_path.exists(), "Final checkpoint should exist at {final_path:?}");

    // Load step-5 checkpoint and verify it produces valid output
    let step5_ckpt = checkpoint_dir.join("step_5");
    let loaded: ebm::EnergyHead<TestBackend> =
        resume_from_checkpoint(&step5_ckpt, &head_config, &Default::default()).unwrap();

    let probe = embeddings_to_tensor::<TestBackend>(&[vec![0.5_f32; d_encoder]], &Default::default());
    let energy: f32 = loaded.forward(probe).into_scalar().elem();
    assert!(energy.is_finite(), "Mid-checkpoint energy should be finite, got {energy}");
}

// ---------------------------------------------------------------------------
// Test 18: Re-exports are accessible via ebm::*
// ---------------------------------------------------------------------------

#[test]
fn test_re_exports_accessible() {
    // This test verifies that all key types are accessible through the
    // top-level ebm:: re-exports. If any re-export is broken, this test
    // fails at compile time (the imports at the top of this file).
    // Here we just verify a few runtime properties to ensure the types
    // are the same as the deep-path versions.

    let device: <TestBackend as Backend>::Device = Default::default();

    // EnergyHeadConfig re-export
    let config: EnergyHeadConfigReexport = EnergyHeadConfigReexport::new(16);
    assert_eq!(config.d_encoder, 16);

    // EnergyHead re-export
    let _head: EnergyHeadReexport<TestBackend> = config
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0)
        .init(&device);

    // EncoderBackend re-export
    let backend = EncoderBackend::default();
    assert_eq!(backend.hidden_dim(), 4096);

    // EBMTrainingConfig re-export
    let train_config = EBMTrainingConfigReexport::new();
    assert_eq!(train_config.total_steps, 50_000);

    // lr_schedule re-export
    let lr = lr_schedule(1e-4, 100, 1000, 50);
    assert!(lr > 0.0 && lr < 1e-4);

    // MetricsHistory re-export
    let history = MetricsHistoryReexport::new();
    assert!(history.is_empty());

    // SpectralNormLinearConfig re-export
    let _snl_config = SpectralNormLinearConfig::new(8, 4);

    // Verify types are identical by using deep-path and re-export interchangeably:
    // If these were different types, the assignment would fail.
    let deep_config = ebm::model::energy_head::EnergyHeadConfig::new(32);
    let _reexport_config: EnergyHeadConfigReexport = deep_config;
}

// ---------------------------------------------------------------------------
// Test 19: All-unproved data produces sampler error
// ---------------------------------------------------------------------------

#[test]
fn test_all_unproved_sampler_error() {
    let tmp = TempDir::new().unwrap();

    // Write only unproved theorems — these produce only negative labels
    let path1 = write_unproved_theorem(tmp.path(), "unproved1.parquet", "thm_fail_a");
    let path2 = write_unproved_theorem(tmp.path(), "unproved2.parquet", "thm_fail_b");

    // ContrastiveSampler requires theorems with BOTH positive and negative records.
    // All-unproved data has only negative records → no eligible theorems → error.
    let result = ContrastiveSampler::from_parquet(&[path1, path2], 2);
    match result {
        Ok(_) => panic!("Sampler should error when no eligible theorems exist (all unproved)"),
        Err(e) => {
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("No eligible theorems"),
                "Error should mention missing eligible theorems, got: {err_msg}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 20: EmbeddingCache reduces encode calls
// ---------------------------------------------------------------------------

#[test]
fn test_cache_reduces_encode_calls() {
    let tmp = TempDir::new().unwrap();

    // Write 5 proved theorems = 5*4=20 records, but many share state_pp patterns
    // Each proved theorem has 4 records with unique state_pp per theorem:
    //   "⊢ state_{thm}_{0..3}"
    // So 5 theorems × 4 states = 20 unique states
    let mut paths = Vec::new();
    for i in 0..5 {
        paths.push(write_proved_theorem(
            tmp.path(),
            &format!("thm{i}.parquet"),
            &format!("thm_cache_{i}"),
        ));
    }

    let sampler = ContrastiveSampler::from_parquet(&paths, 2).unwrap();

    // Now add duplicate records by writing same theorem names again
    // Actually, let's just check the raw unique count
    let unique = sampler.unique_states();
    assert_eq!(unique.len(), 20, "5 theorems × 4 states = 20 unique");

    let call_count = AtomicUsize::new(0);
    let encode_fn = |_state: &str| -> anyhow::Result<Vec<f32>> {
        call_count.fetch_add(1, Ordering::SeqCst);
        Ok(vec![0.5_f32; 16])
    };

    let cache = EmbeddingCache::precompute(&sampler, &encode_fn, 16);

    // Verify: encode called exactly once per unique state
    assert_eq!(call_count.load(Ordering::SeqCst), 20);
    assert_eq!(cache.len(), 20);

    // All states should be retrievable
    for state in &unique {
        assert!(
            cache.get(state).is_some(),
            "Cache should contain state: {state}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 21: EmbeddingCache Parquet roundtrip with real sampler data
// ---------------------------------------------------------------------------

#[test]
fn test_cache_parquet_roundtrip_with_sampler() {
    let tmp = TempDir::new().unwrap();

    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_rt");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_rt2");
    let sampler = ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();

    let d = 8;
    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        let hash = state.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        Ok((0..d).map(|i| ((hash as f32 + i as f32) * 0.1).sin()).collect())
    };

    let cache = EmbeddingCache::precompute(&sampler, &encode_fn, d);
    let cache_path = tmp.path().join("embeddings.parquet");
    cache.save(&cache_path).unwrap();

    let loaded = EmbeddingCache::load(&cache_path).unwrap();
    assert_eq!(loaded.len(), cache.len());
    assert_eq!(loaded.dim(), d);

    // Verify every embedding matches
    for state in sampler.unique_states() {
        let original = cache.get(state).unwrap();
        let restored = loaded.get(state).unwrap();
        assert_eq!(original, restored, "Embedding mismatch for state: {state}");
    }
}

// ---------------------------------------------------------------------------
// Test 22: Train with cached embeddings end-to-end
// ---------------------------------------------------------------------------

#[test]
fn test_train_with_cached_embeddings_end_to_end() {
    let tmp = TempDir::new().unwrap();

    // Write trajectory data
    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_ce_a");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_ce_b");
    let sampler = ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();

    let d_encoder = 16;

    // Precompute cache with a mock encode_fn
    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        let hash = state.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        Ok((0..d_encoder).map(|i| ((hash as f32 + i as f32) * 0.1).sin()).collect())
    };

    let cache = EmbeddingCache::precompute(&sampler, &encode_fn, d_encoder);
    assert_eq!(cache.len(), sampler.unique_states().len());

    // Use cache-backed encode_fn for training (no LLM calls)
    let cache_encode = |state: &str| -> anyhow::Result<Vec<f32>> {
        cache.get_or_err(state)
    };

    let device = Default::default();
    let model = EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0)
        .init::<TestAutodiffBackend>(&device);

    let checkpoint_dir = tmp.path().join("ckpt_ce");
    let config = EBMTrainingConfig::new()
        .with_total_steps(10)
        .with_warmup_steps(2)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(0)
        .with_checkpoint_interval(0)
        .with_checkpoint_dir(checkpoint_dir.to_string_lossy().to_string());

    let result = train(&config, model, &cache_encode, &sampler, &device);
    assert!(result.is_ok(), "Training with cached embeddings should succeed: {:?}", result.err());

    // Verify checkpoint saved
    let final_ckpt = checkpoint_dir.join("final.mpk");
    assert!(final_ckpt.exists(), "Final checkpoint should exist");

    // Verify trained model produces finite outputs
    let trained = result.unwrap();
    let probe = embeddings_to_tensor::<TestAutodiffBackend>(
        &[vec![0.5_f32; d_encoder]],
        &device,
    );
    let energy: f32 = trained.forward(probe).into_scalar().elem();
    assert!(energy.is_finite(), "Trained model output should be finite: {energy}");
}

// ---------------------------------------------------------------------------
// Test 23: Cache → scorer inference chain
// ---------------------------------------------------------------------------

#[test]
fn test_cache_scorer_inference_chain() {
    let tmp = TempDir::new().unwrap();

    // Write trajectory data + precompute cache
    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_cs_a");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_cs_b");
    let sampler = ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();

    let d_encoder = 16;
    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        let hash = state.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        Ok((0..d_encoder).map(|i| ((hash as f32 + i as f32) * 0.1).sin()).collect())
    };

    let cache = EmbeddingCache::precompute(&sampler, &encode_fn, d_encoder);

    // Save and reload cache from Parquet
    let cache_path = tmp.path().join("cache_scorer.parquet");
    cache.save(&cache_path).unwrap();
    let loaded_cache = EmbeddingCache::load(&cache_path).unwrap();
    assert_eq!(loaded_cache.len(), cache.len());

    // Create EBMScorer with cache-backed encode_fn
    let device: <TestBackend as burn::prelude::Backend>::Device = Default::default();
    let head = EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0)
        .init::<TestBackend>(&device);

    let cache_encode_fn: Box<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync> =
        Box::new(move |state: &str| loaded_cache.get_or_err(state));

    let scorer = EBMScorer::new(head, cache_encode_fn, device);

    // Score 5 states from the sampler (all should be in cache)
    let unique_states: Vec<String> = sampler.unique_states().into_iter().map(|s| s.to_string()).collect();
    let test_states: Vec<&str> = unique_states.iter().take(5).map(|s| s.as_str()).collect();

    let scores = scorer.score_states(&test_states).unwrap();
    assert_eq!(scores.len(), test_states.len());

    // All scores should be finite
    for (i, &score) in scores.iter().enumerate() {
        assert!(score.is_finite(), "Score[{i}] should be finite, got {score}");
    }

    // Determinism: same state scored twice should return the same value
    // (cache is deterministic, unlike live encoder with SpectralNorm variance)
    // Note: SpectralNorm Option C still re-randomizes u/v, so scores may differ slightly.
    // We use a generous tolerance.
    for state in test_states.iter().take(3) {
        let s1 = scorer.score_state(state).unwrap();
        let s2 = scorer.score_state(state).unwrap();
        assert!(
            (s1 - s2).abs() < 0.5,
            "Same state scored twice should be close: {s1} vs {s2}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 24: Cache partial encode failure
// ---------------------------------------------------------------------------

#[test]
fn test_cache_partial_encode_failure() {
    let tmp = TempDir::new().unwrap();

    let path1 = write_proved_theorem(tmp.path(), "thm1.parquet", "thm_pf_a");
    let path2 = write_proved_theorem(tmp.path(), "thm2.parquet", "thm_pf_b");
    let sampler = ContrastiveSampler::from_parquet(&[path1, path2], 2).unwrap();

    let d_encoder = 16;
    let total_unique = sampler.unique_states().len();

    // encode_fn that fails for states containing "thm_pf_a_3" (the dead-end node of first theorem)
    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        if state.contains("thm_pf_a_3") {
            anyhow::bail!("Simulated failure for state: {state}")
        }
        let hash = state.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        Ok((0..d_encoder).map(|i| ((hash as f32 + i as f32) * 0.1).sin()).collect())
    };

    let cache = EmbeddingCache::precompute(&sampler, &encode_fn, d_encoder);

    // Cache should have fewer entries than total unique states
    assert!(
        cache.len() < total_unique,
        "Cache should have fewer entries ({}) than total unique states ({total_unique}) due to failures",
        cache.len()
    );
    assert!(cache.len() > 0, "Cache should have at least some entries");

    // Training with this partial cache should still succeed (train skips missing states)
    let cache_encode = |state: &str| -> anyhow::Result<Vec<f32>> {
        cache.get_or_err(state)
    };

    let device = Default::default();
    let model = EnergyHeadConfig::new(d_encoder)
        .with_d_hidden1(8)
        .with_d_hidden2(4)
        .with_dropout(0.0)
        .init::<TestAutodiffBackend>(&device);

    let checkpoint_dir = tmp.path().join("ckpt_pf");
    let config = EBMTrainingConfig::new()
        .with_total_steps(10)
        .with_warmup_steps(0)
        .with_batch_size(2)
        .with_k_negatives(2)
        .with_log_interval(0)
        .with_checkpoint_interval(0)
        .with_checkpoint_dir(checkpoint_dir.to_string_lossy().to_string());

    let result = train(&config, model, &cache_encode, &sampler, &device);
    assert!(result.is_ok(), "train() should handle partial cache gracefully: {:?}", result.err());
}

// ---------------------------------------------------------------------------
// Test 25: Empty cache roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_cache_empty_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("empty_cache.parquet");

    let cache = EmbeddingCache::new(0);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.dim(), 0);

    cache.save(&path).unwrap();
    assert!(path.exists());

    let loaded = EmbeddingCache::load(&path).unwrap();
    assert_eq!(loaded.len(), 0);
    assert!(loaded.is_empty());
    assert_eq!(loaded.dim(), 0);
}

// ---------------------------------------------------------------------------
// Test 26: EmbeddingCache re-export accessible
// ---------------------------------------------------------------------------

#[test]
fn test_cache_re_export_accessible() {
    // Verify EmbeddingCache is accessible via ebm::EmbeddingCache re-export.
    // The import at the top of this file (line ~28) is the real test — if the
    // re-export is broken, this file won't compile. We add a runtime check too.
    let cache = EmbeddingCache::new(16);
    assert_eq!(cache.dim(), 16);
    assert!(cache.is_empty());

    // Verify ContrastiveSampler re-export works at runtime too
    let _ = std::any::type_name::<ContrastiveSamplerReexport>();
}
