//! EBM training loop with single optimizer on energy head.
//!
//! Ties together the contrastive sampler, encoder bridge, loss functions,
//! and metrics into a training loop using AdamW with warmup + cosine LR schedule.

use std::path::Path;
use std::time::Instant;

use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;
use rand::SeedableRng;

use crate::model::bridge::embeddings_to_tensor;
use crate::model::energy_head::{EnergyHead, EnergyHeadConfig};
use crate::training::data::ContrastiveSampler;
use crate::training::loss::{depth_regression_loss, info_nce_loss};
use crate::training::metrics::{EBMMetrics, MetricsHistory};

/// Configuration for EBM training.
#[derive(Config, Debug)]
pub struct EBMTrainingConfig {
    /// Base learning rate for AdamW.
    #[config(default = 1e-4)]
    pub lr: f64,
    /// Weight decay for AdamW.
    #[config(default = 0.01)]
    pub weight_decay: f64,
    /// Maximum gradient norm for clipping.
    #[config(default = 1.0)]
    pub max_grad_norm: f64,
    /// Total number of training steps.
    #[config(default = 50_000)]
    pub total_steps: usize,
    /// Number of linear warmup steps.
    #[config(default = 1_000)]
    pub warmup_steps: usize,
    /// Weight for the depth regression loss component.
    #[config(default = 0.3)]
    pub depth_loss_weight: f64,
    /// Steps between metric logging.
    #[config(default = 50)]
    pub log_interval: usize,
    /// Steps between checkpoint saves.
    #[config(default = 5_000)]
    pub checkpoint_interval: usize,
    /// Number of negative samples per positive.
    #[config(default = 7)]
    pub k_negatives: usize,
    /// Batch size (number of contrastive samples per step).
    #[config(default = 32)]
    pub batch_size: usize,
    /// Directory for saving checkpoints.
    #[config(default = "String::from(\"checkpoints/ebm\")")]
    pub checkpoint_dir: String,
}

/// Compute the learning rate at a given step using warmup + cosine decay.
///
/// - Warmup phase (`step < warmup_steps`): linearly ramps from 0 to `base_lr`.
/// - Cosine phase: decays from `base_lr` to 0 following a cosine schedule.
pub fn lr_schedule(base_lr: f64, warmup_steps: usize, total_steps: usize, step: usize) -> f64 {
    if warmup_steps > 0 && step < warmup_steps {
        // Linear warmup
        base_lr * (step + 1) as f64 / warmup_steps as f64
    } else {
        // Cosine decay
        let decay_steps = total_steps.saturating_sub(warmup_steps).max(1);
        let progress = (step.saturating_sub(warmup_steps)) as f64 / decay_steps as f64;
        let progress = progress.min(1.0);
        base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

/// Evaluate a batch of contrastive samples and return metrics (no gradients).
///
/// Used for both train metrics (after backward) and validation.
fn eval_batch<B: Backend>(
    model: &EnergyHead<B>,
    encode_fn: &dyn Fn(&str) -> anyhow::Result<Vec<f32>>,
    sampler: &ContrastiveSampler,
    batch_size: usize,
    k_negatives: usize,
    depth_loss_weight: f64,
    rng: &mut impl rand::Rng,
    device: &B::Device,
) -> Option<EBMMetrics> {
    let samples = sampler.sample_batch(batch_size, rng);

    // Encode positive states
    let mut pos_embeddings = Vec::with_capacity(batch_size);
    let mut remaining_depths = Vec::with_capacity(batch_size);

    for sample in &samples {
        match encode_fn(&sample.positive.state_pp) {
            Ok(emb) => {
                pos_embeddings.push(emb);
                remaining_depths.push(sample.remaining_depth as f32);
            }
            Err(_) => return None,
        }
    }
    if pos_embeddings.is_empty() {
        return None;
    }

    // Encode negative states
    let mut neg_embeddings = Vec::with_capacity(batch_size * k_negatives);
    for sample in &samples {
        for neg in &sample.negatives {
            match encode_fn(&neg.state_pp) {
                Ok(emb) => neg_embeddings.push(emb),
                Err(_) => return None,
            }
        }
    }

    let actual_batch = pos_embeddings.len();
    let k = k_negatives;

    let pos_tensor = embeddings_to_tensor::<B>(&pos_embeddings, device);
    let neg_tensor = embeddings_to_tensor::<B>(&neg_embeddings, device);
    let remaining_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(remaining_depths, [actual_batch]),
        device,
    );

    let pos_energy = model.forward(pos_tensor);
    let neg_energy_flat = model.forward(neg_tensor);
    let neg_energies = neg_energy_flat.reshape([actual_batch, k]);

    let contrastive_loss = info_nce_loss(pos_energy.clone(), neg_energies.clone());
    let depth_loss = depth_regression_loss(pos_energy.clone(), remaining_tensor);

    let contrastive_val: f64 = contrastive_loss.clone().into_scalar().elem();
    let depth_val: f64 = depth_loss.clone().into_scalar().elem();
    let total_val = contrastive_val + depth_val * depth_loss_weight;

    Some(EBMMetrics::compute(
        &pos_energy,
        &neg_energies,
        contrastive_val,
        depth_val,
        total_val,
    ))
}

/// Run the EBM training loop.
///
/// Trains the energy head using contrastive (InfoNCE) + depth regression loss.
/// The `encode_fn` closure maps proof state strings to embedding vectors,
/// decoupling this module from the LLM encoder.
///
/// # Arguments
/// - `config`: training hyperparameters
/// - `model`: initialized EnergyHead (will be consumed and returned updated)
/// - `encode_fn`: encodes a proof state string into a `Vec<f32>` embedding
/// - `sampler`: provides contrastive training batches from trajectory data
/// - `val_sampler`: optional validation sampler for overfitting detection
/// - `device`: burn device for tensor operations
///
/// # Returns
/// The trained EnergyHead model.
pub fn train<B: AutodiffBackend>(
    config: &EBMTrainingConfig,
    mut model: EnergyHead<B>,
    encode_fn: &dyn Fn(&str) -> anyhow::Result<Vec<f32>>,
    sampler: &ContrastiveSampler,
    val_sampler: Option<&ContrastiveSampler>,
    device: &B::Device,
) -> anyhow::Result<EnergyHead<B>> {
    // Create checkpoint directory
    std::fs::create_dir_all(&config.checkpoint_dir)?;

    // Initialize optimizer
    let optim_config = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(
            config.max_grad_norm as f32,
        )));
    let mut optimizer = optim_config.init();

    let mut rng = rand::rngs::StdRng::from_entropy();
    let mut val_rng = rand::rngs::StdRng::from_entropy();
    let mut _history = MetricsHistory::new();
    let train_start = Instant::now();

    for step in 0..config.total_steps {
        let lr = lr_schedule(config.lr, config.warmup_steps, config.total_steps, step);

        // Sample contrastive batch
        let samples = sampler.sample_batch(config.batch_size, &mut rng);

        // Encode positive states
        let mut pos_embeddings = Vec::with_capacity(config.batch_size);
        let mut remaining_depths = Vec::with_capacity(config.batch_size);
        let mut encode_failed = false;

        for sample in &samples {
            match encode_fn(&sample.positive.state_pp) {
                Ok(emb) => {
                    pos_embeddings.push(emb);
                    remaining_depths.push(sample.remaining_depth as f32);
                }
                Err(e) => {
                    tracing::debug!("Failed to encode positive state: {e}");
                    encode_failed = true;
                    break;
                }
            }
        }
        if encode_failed || pos_embeddings.is_empty() {
            continue;
        }

        // Encode negative states
        let mut neg_embeddings = Vec::with_capacity(config.batch_size * config.k_negatives);
        for sample in &samples {
            for neg in &sample.negatives {
                match encode_fn(&neg.state_pp) {
                    Ok(emb) => neg_embeddings.push(emb),
                    Err(e) => {
                        tracing::debug!("Failed to encode negative state: {e}");
                        encode_failed = true;
                        break;
                    }
                }
                if encode_failed {
                    break;
                }
            }
            if encode_failed {
                break;
            }
        }
        if encode_failed {
            continue;
        }

        let batch_size = pos_embeddings.len();
        let k = config.k_negatives;

        // Convert to burn tensors
        let pos_tensor = embeddings_to_tensor::<B>(&pos_embeddings, device);
        let neg_tensor = embeddings_to_tensor::<B>(&neg_embeddings, device);
        let remaining_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(remaining_depths, [batch_size]),
            device,
        );

        // Forward pass
        let pos_energy = model.forward(pos_tensor);
        let neg_energy_flat = model.forward(neg_tensor);
        let neg_energies = neg_energy_flat.reshape([batch_size, k]);

        // Clone tensors for metrics (before backward consumes them)
        let pos_energy_metrics = pos_energy.clone();
        let neg_energies_metrics = neg_energies.clone();

        // Compute losses
        let contrastive_loss = info_nce_loss(pos_energy.clone(), neg_energies);
        let depth_loss = depth_regression_loss(pos_energy, remaining_tensor);

        // Extract scalar values before backward
        let contrastive_val: f64 = contrastive_loss.clone().into_scalar().elem();
        let depth_val: f64 = depth_loss.clone().into_scalar().elem();

        let total_loss = contrastive_loss + depth_loss * config.depth_loss_weight;
        let total_val: f64 = total_loss.clone().into_scalar().elem();

        // Backward + optimizer step
        let grads = GradientsParams::from_grads(total_loss.backward(), &model);
        model = optimizer.step(lr.into(), model, grads);

        // Log metrics at intervals
        if config.log_interval > 0 && step % config.log_interval == 0 {
            let metrics = EBMMetrics::compute(
                &pos_energy_metrics,
                &neg_energies_metrics,
                contrastive_val,
                depth_val,
                total_val,
            );

            let warnings = metrics.health_check();
            if !warnings.is_empty() {
                tracing::warn!(step, "Health check warnings: {:?}", warnings);
            }

            let elapsed = train_start.elapsed().as_secs_f64();
            let remaining = if step > 0 {
                elapsed * (config.total_steps - step) as f64 / step as f64
            } else {
                0.0
            };
            let eta = if remaining < 60.0 {
                format!("{:.0}s", remaining)
            } else if remaining < 3600.0 {
                format!("{:.0}m", remaining / 60.0)
            } else {
                format!("{:.1}h", remaining / 3600.0)
            };

            // Compute validation metrics if val sampler is available
            let val_str = if let Some(val_s) = val_sampler {
                let inner_device = device.clone();
                let val_model = model.valid();
                match eval_batch(
                    &val_model,
                    encode_fn,
                    val_s,
                    config.batch_size,
                    config.k_negatives,
                    config.depth_loss_weight,
                    &mut val_rng,
                    &inner_device,
                ) {
                    Some(vm) => format!(
                        " | val: loss={:.4} gap={:.2} rank={:.2}",
                        vm.loss, vm.energy_gap, vm.rank_accuracy
                    ),
                    None => String::new(),
                }
            } else {
                String::new()
            };

            tracing::info!(step, lr, eta, "{}{}", metrics.display(), val_str);
            _history.push(step, metrics);
        }

        // Save checkpoint at intervals
        if config.checkpoint_interval > 0 && step > 0 && step % config.checkpoint_interval == 0 {
            let path = format!("{}/step_{step}", config.checkpoint_dir);
            model
                .clone()
                .save_file(path, &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
                .map_err(|e| anyhow::anyhow!("Failed to save checkpoint at step {step}: {e}"))?;
            tracing::info!(step, "Checkpoint saved");
        }
    }

    // Save final model
    let final_path = format!("{}/final", config.checkpoint_dir);
    model
        .clone()
        .save_file(
            final_path,
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to save final model: {e}"))?;
    tracing::info!("Training complete. Final model saved.");

    Ok(model)
}

/// Load an EnergyHead from a checkpoint file.
///
/// Creates a fresh model from config, then loads saved weights on top.
pub fn resume_from_checkpoint<B: Backend>(
    path: &Path,
    config: &EnergyHeadConfig,
    device: &B::Device,
) -> anyhow::Result<EnergyHead<B>> {
    let model = config
        .init::<B>(device)
        .load_file(
            path,
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            device,
        )
        .map_err(|e| anyhow::anyhow!("Failed to load checkpoint from {}: {e}", path.display()))?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr_schedule() {
        let base_lr = 1e-4;
        let warmup = 100;
        let total = 1000;

        // Warmup phase: step 0 → lr = base_lr * 1/100
        let lr0 = lr_schedule(base_lr, warmup, total, 0);
        assert!(
            (lr0 - base_lr / 100.0).abs() < 1e-10,
            "Step 0: expected {}, got {lr0}",
            base_lr / 100.0
        );

        // Warmup midpoint: step 49 → lr = base_lr * 50/100
        let lr49 = lr_schedule(base_lr, warmup, total, 49);
        assert!(
            (lr49 - base_lr * 0.5).abs() < 1e-10,
            "Step 49: expected {}, got {lr49}",
            base_lr * 0.5
        );

        // Peak at warmup boundary: step 99 → lr = base_lr * 100/100 = base_lr
        let lr99 = lr_schedule(base_lr, warmup, total, 99);
        assert!(
            (lr99 - base_lr).abs() < 1e-10,
            "Step 99: expected {base_lr}, got {lr99}"
        );

        // Start of cosine: step 100 → lr = base_lr * 0.5 * (1 + cos(0)) = base_lr
        let lr100 = lr_schedule(base_lr, warmup, total, 100);
        assert!(
            (lr100 - base_lr).abs() < 1e-10,
            "Step 100: expected {base_lr}, got {lr100}"
        );

        // Cosine midpoint: step 550 → progress = 450/900 = 0.5
        // lr = base_lr * 0.5 * (1 + cos(PI * 0.5)) = base_lr * 0.5
        let lr550 = lr_schedule(base_lr, warmup, total, 550);
        assert!(
            (lr550 - base_lr * 0.5).abs() < 1e-10,
            "Step 550: expected {}, got {lr550}",
            base_lr * 0.5
        );

        // Near end: step 999 → progress ≈ 1.0, lr ≈ 0
        let lr999 = lr_schedule(base_lr, warmup, total, 999);
        assert!(
            lr999 < base_lr * 0.01,
            "Step 999: expected near-zero, got {lr999}"
        );

        // Edge case: warmup_steps = 0 (no warmup, straight cosine)
        let lr_no_warmup = lr_schedule(base_lr, 0, 1000, 0);
        assert!(
            (lr_no_warmup - base_lr).abs() < 1e-10,
            "No warmup step 0: expected {base_lr}, got {lr_no_warmup}"
        );

        // Edge case: total_steps == warmup_steps (all warmup, no cosine)
        let lr_all_warmup = lr_schedule(base_lr, 100, 100, 50);
        assert!(
            (lr_all_warmup - base_lr * 51.0 / 100.0).abs() < 1e-10,
            "All-warmup step 50: expected {}, got {lr_all_warmup}",
            base_lr * 51.0 / 100.0
        );
    }
}
