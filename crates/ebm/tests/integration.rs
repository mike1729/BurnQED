//! Integration tests for the EBM crate.
//!
//! These tests exercise cross-module interactions: EnergyHead + loss + optimizer,
//! bridge -> model -> extraction pipeline, Parquet -> ContrastiveSampler pipeline,
//! and full training step simulations. All use NdArray backend and synthetic data —
//! no Lean, no LLM model needed.

use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::{Distribution, TensorData};
use tempfile::TempDir;

use ebm::model::bridge::{embeddings_to_tensor, tensor_to_vec};
use ebm::model::energy_head::EnergyHeadConfig;

use ebm::training::data::ContrastiveSampler;
use ebm::training::loss::{depth_regression_loss, info_nce_loss};
use ebm::training::metrics::{EBMMetrics, MetricsHistory};
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

    // Simulate candle encoder output: Vec<Vec<f32>>
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
