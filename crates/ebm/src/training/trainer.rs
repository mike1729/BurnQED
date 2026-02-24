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
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;
use rand::SeedableRng;

use crate::model::bridge::embeddings_to_tensor;
use crate::model::energy_head::{EnergyHead, EnergyHeadConfig};
use crate::training::data::{ContrastiveSample, ContrastiveSampler};
use crate::training::loss::{contrastive_loss, depth_regression_loss, ContrastiveLossType};
use crate::training::metrics::{EBMMetrics, MetricsHistory};

/// Metadata saved alongside each checkpoint for resuming training.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct CheckpointMeta {
    pub step: usize,
    pub trained_steps: u64,
    pub skipped_steps: u64,
}

/// Configuration for EBM training.
#[derive(Config, Debug)]
pub struct EBMTrainingConfig {
    /// Base learning rate for AdamW.
    #[config(default = 3e-5)]
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
    #[config(default = 2000)]
    pub log_interval: usize,
    /// Steps between checkpoint saves.
    #[config(default = 5_000)]
    pub checkpoint_interval: usize,
    /// Number of negative samples per positive.
    #[config(default = 7)]
    pub k_negatives: usize,
    /// Batch size (number of contrastive samples per step).
    #[config(default = 256)]
    pub batch_size: usize,
    /// Directory for saving checkpoints.
    #[config(default = "String::from(\"checkpoints/ebm\")")]
    pub checkpoint_dir: String,
    /// Contrastive loss type: InfoNCE (softmax) or MarginRanking (hinge).
    #[config(default = "ContrastiveLossType::InfoNCE")]
    pub loss_type: ContrastiveLossType,
    /// Margin for margin ranking loss. Ignored when using InfoNCE.
    #[config(default = 1.0)]
    pub margin: f64,
    /// Number of validation batches to evaluate at each log interval.
    #[config(default = 40)]
    pub val_batches: usize,
    /// Hard ratio for final full-validation evaluation (natural search distribution).
    #[config(default = 0.1)]
    pub final_val_hard_ratio: f64,
    /// Medium ratio for final full-validation evaluation.
    #[config(default = 0.3)]
    pub final_val_medium_ratio: f64,
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

/// Evaluate pre-sampled contrastive examples and return metrics (no gradients).
///
/// Core evaluation logic shared by training-time validation and full final eval.
fn eval_samples<B: Backend>(
    model: &EnergyHead<B>,
    encode_fn: &dyn Fn(&str) -> anyhow::Result<Vec<f32>>,
    samples: &[ContrastiveSample],
    k_negatives: usize,
    depth_loss_weight: f64,
    loss_type: ContrastiveLossType,
    margin: f64,
    device: &B::Device,
) -> Option<EBMMetrics> {
    // Encode positive states
    let mut pos_embeddings = Vec::with_capacity(samples.len());
    let mut remaining_depths = Vec::with_capacity(samples.len());

    for sample in samples {
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
    let mut neg_embeddings = Vec::with_capacity(samples.len() * k_negatives);
    for sample in samples {
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

    let pos_energy_raw = model.forward(pos_tensor);
    let neg_energy_flat_raw = model.forward(neg_tensor);

    // Temperature scaling only for InfoNCE (softmax needs it).
    // Margin ranking must use raw energies — temperature would let the
    // optimizer collapse τ→0 to trivially satisfy the margin.
    let (pos_energy, neg_energy_flat) = if loss_type == ContrastiveLossType::InfoNCE {
        (
            model.temperature_scale(pos_energy_raw.clone()),
            model.temperature_scale(neg_energy_flat_raw.clone()),
        )
    } else {
        (pos_energy_raw.clone(), neg_energy_flat_raw.clone())
    };
    let neg_energies = neg_energy_flat.reshape([actual_batch, k]);

    let cl = contrastive_loss(loss_type, pos_energy.clone(), neg_energies.clone(), margin);
    // Depth regression uses raw (unscaled) energy — it's an absolute target.
    let depth_loss = depth_regression_loss(pos_energy_raw, remaining_tensor);

    let contrastive_val: f64 = cl.clone().into_scalar().elem();
    let depth_val: f64 = depth_loss.clone().into_scalar().elem();
    let total_val = contrastive_val + depth_val * depth_loss_weight;

    Some(EBMMetrics::compute(
        &pos_energy,
        &neg_energies,
        contrastive_val,
        depth_val,
        total_val,
        margin,
    ))
}

/// Evaluate a batch of contrastive samples and return metrics (no gradients).
///
/// Samples from the given sampler and delegates to [`eval_samples`].
/// Used for training-time periodic validation.
fn eval_batch<B: Backend>(
    model: &EnergyHead<B>,
    encode_fn: &dyn Fn(&str) -> anyhow::Result<Vec<f32>>,
    sampler: &ContrastiveSampler,
    batch_size: usize,
    k_negatives: usize,
    depth_loss_weight: f64,
    loss_type: ContrastiveLossType,
    margin: f64,
    rng: &mut impl rand::Rng,
    device: &B::Device,
) -> Option<EBMMetrics> {
    let samples = sampler.sample_batch(batch_size, rng);
    eval_samples(
        model,
        encode_fn,
        &samples,
        k_negatives,
        depth_loss_weight,
        loss_type,
        margin,
        device,
    )
}

/// Running average accumulator for training metrics over a logging interval.
struct RunningAvg {
    loss: f64,
    gap: f64,
    rank: f64,
    pairwise: f64,
    active: f64,
    pos_e: f64,
    neg_e: f64,
    std: f64,
    count: usize,
}

impl RunningAvg {
    fn new() -> Self {
        Self { loss: 0.0, gap: 0.0, rank: 0.0, pairwise: 0.0, active: 0.0, pos_e: 0.0, neg_e: 0.0, std: 0.0, count: 0 }
    }

    fn update(&mut self, m: &EBMMetrics) {
        self.loss += m.loss;
        self.gap += m.energy_gap;
        self.rank += m.rank_accuracy;
        self.pairwise += m.pairwise_acc;
        self.active += m.active_fraction;
        self.pos_e += m.pos_energy_mean;
        self.neg_e += m.neg_energy_mean;
        self.std += m.energy_std;
        self.count += 1;
    }

    fn display(&self) -> String {
        if self.count == 0 {
            return "no data".to_string();
        }
        let n = self.count as f64;
        format!(
            "loss={:.4} gap={:.2} rank={:.2} pair={:.2} active={:.2} pos_e={:.2} neg_e={:.2} std={:.2}",
            self.loss / n, self.gap / n, self.rank / n,
            self.pairwise / n, self.active / n,
            self.pos_e / n, self.neg_e / n, self.std / n,
        )
    }

    fn avg_metrics(&self) -> Option<EBMMetrics> {
        if self.count == 0 {
            return None;
        }
        let n = self.count as f64;
        Some(EBMMetrics {
            loss: self.loss / n,
            contrastive_loss: 0.0,
            depth_loss: 0.0,
            energy_gap: self.gap / n,
            rank_accuracy: self.rank / n,
            pos_energy_mean: self.pos_e / n,
            neg_energy_mean: self.neg_e / n,
            energy_std: self.std / n,
            pairwise_acc: self.pairwise / n,
            active_fraction: self.active / n,
        })
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
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
/// - `resume_step`: if `Some(step)`, load optimizer state and metadata from
///   `{checkpoint_dir}/step_{step}/` and continue training from that step
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
    resume_step: Option<usize>,
) -> anyhow::Result<EnergyHead<B>> {
    // Create checkpoint directory
    std::fs::create_dir_all(&config.checkpoint_dir)?;

    tracing::info!(
        loss_type = %config.loss_type,
        margin = config.margin,
        "Contrastive loss: {}{}",
        config.loss_type,
        if config.loss_type == ContrastiveLossType::MarginRanking {
            format!(" (margin={})", config.margin)
        } else {
            String::new()
        }
    );

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
    let mut running_avg = RunningAvg::new();
    let train_start = Instant::now();
    let mut trained_steps: u64 = 0;
    let mut skipped_steps: u64 = 0;
    let start_step: usize;

    // Resume from checkpoint if requested
    if let Some(step) = resume_step {
        let step_dir = format!("{}/step_{step}", config.checkpoint_dir);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        // Load optimizer state
        let optim_path = format!("{step_dir}/optimizer");
        let optim_record = recorder
            .load(optim_path.into(), device)
            .map_err(|e| anyhow::anyhow!("Failed to load optimizer from {step_dir}: {e}"))?;
        optimizer = optimizer.load_record(optim_record);
        tracing::info!(step, "Restored optimizer state");

        // Load metadata
        let meta_path = format!("{step_dir}/meta.json");
        let meta: CheckpointMeta = serde_json::from_reader(
            std::fs::File::open(&meta_path)
                .map_err(|e| anyhow::anyhow!("Failed to open {meta_path}: {e}"))?,
        )
        .map_err(|e| anyhow::anyhow!("Failed to parse {meta_path}: {e}"))?;
        trained_steps = meta.trained_steps;
        skipped_steps = meta.skipped_steps;
        start_step = step;

        tracing::info!(
            start_step,
            trained_steps,
            skipped_steps,
            "Resumed training from checkpoint"
        );
    } else {
        start_step = 0;
    }

    for step in start_step..config.total_steps {
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
                    tracing::debug!(step, "Failed to encode positive state: {e}");
                    encode_failed = true;
                    break;
                }
            }
        }
        if encode_failed || pos_embeddings.is_empty() {
            skipped_steps += 1;
            if config.log_interval > 0 && step % config.log_interval == 0 {
                let skip_rate = skipped_steps as f64 / (step + 1) as f64 * 100.0;
                tracing::warn!(step, skipped_steps, skip_rate = format!("{skip_rate:.1}%"), "Step skipped (encode failure)");
            }
            continue;
        }

        // Encode negative states
        let mut neg_embeddings = Vec::with_capacity(config.batch_size * config.k_negatives);
        for sample in &samples {
            for neg in &sample.negatives {
                match encode_fn(&neg.state_pp) {
                    Ok(emb) => neg_embeddings.push(emb),
                    Err(e) => {
                        tracing::debug!(step, "Failed to encode negative state: {e}");
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
            skipped_steps += 1;
            if config.log_interval > 0 && step % config.log_interval == 0 {
                let skip_rate = skipped_steps as f64 / (step + 1) as f64 * 100.0;
                tracing::warn!(step, skipped_steps, skip_rate = format!("{skip_rate:.1}%"), "Step skipped (encode failure)");
            }
            continue;
        }

        trained_steps += 1;

        let batch_size = pos_embeddings.len();
        let k = config.k_negatives;

        // Convert to burn tensors
        let pos_tensor = embeddings_to_tensor::<B>(&pos_embeddings, device);
        let neg_tensor = embeddings_to_tensor::<B>(&neg_embeddings, device);
        let remaining_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(remaining_depths, [batch_size]),
            device,
        );

        // Forward pass — raw (unscaled) energy
        let pos_energy_raw = model.forward(pos_tensor);
        let neg_energy_flat_raw = model.forward(neg_tensor);

        // Temperature scaling only for InfoNCE (softmax needs it).
        // Margin ranking must use raw energies — temperature would let the
        // optimizer collapse τ→0 to trivially satisfy the margin.
        let (pos_energy, neg_energy_flat) = if config.loss_type == ContrastiveLossType::InfoNCE {
            (
                model.temperature_scale(pos_energy_raw.clone()),
                model.temperature_scale(neg_energy_flat_raw.clone()),
            )
        } else {
            (pos_energy_raw.clone(), neg_energy_flat_raw.clone())
        };
        let neg_energies = neg_energy_flat.reshape([batch_size, k]);

        // Clone tensors for metrics (before backward consumes them)
        let pos_energy_metrics = pos_energy.clone();
        let neg_energies_metrics = neg_energies.clone();

        // Compute losses
        let cl = contrastive_loss(config.loss_type, pos_energy.clone(), neg_energies, config.margin);
        // Depth regression uses raw (unscaled) energy — it's an absolute target.
        let depth_loss = depth_regression_loss(pos_energy_raw, remaining_tensor);

        // Extract scalar values before backward
        let contrastive_val: f64 = cl.clone().into_scalar().elem();
        let depth_val: f64 = depth_loss.clone().into_scalar().elem();

        let total_loss = cl + depth_loss * config.depth_loss_weight;
        let total_val: f64 = total_loss.clone().into_scalar().elem();

        // NaN guard: skip optimizer step if loss is NaN/Inf to prevent
        // permanently corrupting model weights with NaN gradients.
        if total_val.is_nan() || total_val.is_infinite() {
            skipped_steps += 1;
            if step % 1000 == 0 {
                tracing::warn!(step, loss = total_val, "Skipping step: loss is NaN/Inf");
            }
            continue;
        }

        // Backward + optimizer step
        let grads = GradientsParams::from_grads(total_loss.backward(), &model);
        model = optimizer.step(lr.into(), model, grads);

        // Accumulate running average metrics every step
        {
            let step_metrics = EBMMetrics::compute(
                &pos_energy_metrics,
                &neg_energies_metrics,
                contrastive_val,
                depth_val,
                total_val,
                config.margin,
            );
            running_avg.update(&step_metrics);
        }

        // Log metrics at intervals
        if config.log_interval > 0 && step % config.log_interval == 0 {
            let avg_metrics = running_avg.avg_metrics();
            let warnings = avg_metrics.as_ref().map(|m| m.health_check()).unwrap_or_default();
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
            // Average over multiple batches for stable estimates
            let val_str = if let Some(val_s) = val_sampler {
                let inner_device = device.clone();
                let val_model = model.valid();
                let mut val_avg = RunningAvg::new();
                for _ in 0..config.val_batches {
                    if let Some(vm) = eval_batch(
                        &val_model,
                        encode_fn,
                        val_s,
                        config.batch_size,
                        config.k_negatives,
                        config.depth_loss_weight,
                        config.loss_type,
                        config.margin,
                        &mut val_rng,
                        &inner_device,
                    ) {
                        val_avg.update(&vm);
                    }
                }
                match val_avg.avg_metrics() {
                    Some(vm) => format!(
                        " | val({}): loss={:.4} gap={:.2} rank={:.2}",
                        val_avg.count, vm.loss, vm.energy_gap, vm.rank_accuracy
                    ),
                    None => String::new(),
                }
            } else {
                String::new()
            };

            let avg_display = running_avg.display();
            let lr_str = format!("{:.2e}", lr);
            tracing::info!(step, lr = %lr_str, eta, "avg({}) {}{}", running_avg.count, avg_display, val_str);
            if let Some(m) = avg_metrics {
                _history.push(step, m);
            }
            running_avg.reset();
        }

        // Save checkpoint at intervals
        if config.checkpoint_interval > 0 && step > 0 && step % config.checkpoint_interval == 0 {
            let step_dir = format!("{}/step_{step}", config.checkpoint_dir);
            std::fs::create_dir_all(&step_dir)?;
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

            // Model weights
            let model_path = format!("{step_dir}/model");
            model
                .clone()
                .save_file(&model_path, &recorder)
                .map_err(|e| anyhow::anyhow!("Failed to save model at step {step}: {e}"))?;

            // Optimizer state
            let optim_path = format!("{step_dir}/optimizer");
            recorder
                .record(optimizer.to_record(), optim_path.into())
                .map_err(|e| anyhow::anyhow!("Failed to save optimizer at step {step}: {e}"))?;

            // Metadata
            let meta_path = format!("{step_dir}/meta.json");
            serde_json::to_writer(
                std::fs::File::create(&meta_path)?,
                &CheckpointMeta { step, trained_steps, skipped_steps },
            )?;

            tracing::info!(step, "Checkpoint saved (model + optimizer + meta)");
        }
    }

    // Training summary
    let total_time = train_start.elapsed();
    let skip_rate = if config.total_steps > 0 {
        skipped_steps as f64 / config.total_steps as f64 * 100.0
    } else {
        0.0
    };
    tracing::info!(
        trained_steps,
        skipped_steps,
        total_steps = config.total_steps,
        skip_rate = format!("{skip_rate:.1}%"),
        elapsed_secs = format!("{:.1}", total_time.as_secs_f64()),
        "Training loop finished"
    );
    if skip_rate > 50.0 {
        tracing::warn!(
            skip_rate = format!("{skip_rate:.1}%"),
            "Over 50% of training steps were skipped due to encode failures — \
             check embedding cache coverage"
        );
    }

    // Save final checkpoint (model + optimizer + meta)
    let final_dir = format!("{}/final", config.checkpoint_dir);
    std::fs::create_dir_all(&final_dir)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let final_model_path = format!("{final_dir}/model");
    model
        .clone()
        .save_file(
            &final_model_path,
            &recorder,
        )
        .map_err(|e| anyhow::anyhow!("Failed to save final model: {e}"))?;

    let final_optim_path = format!("{final_dir}/optimizer");
    recorder
        .record(optimizer.to_record(), final_optim_path.into())
        .map_err(|e| anyhow::anyhow!("Failed to save final optimizer: {e}"))?;

    let final_meta_path = format!("{final_dir}/meta.json");
    serde_json::to_writer(
        std::fs::File::create(&final_meta_path)?,
        &CheckpointMeta {
            step: config.total_steps,
            trained_steps,
            skipped_steps,
        },
    )?;

    tracing::info!("Training complete. Final checkpoint saved (model + optimizer + meta).");

    // Full validation evaluation with natural search distribution
    if let Some(val_s) = val_sampler {
        tracing::info!(
            hard_ratio = config.final_val_hard_ratio,
            medium_ratio = config.final_val_medium_ratio,
            "Running full validation evaluation over entire val set..."
        );
        let val_model = model.valid();
        let num_batches = (val_s.num_records() / config.batch_size).max(1);
        let mut val_avg = RunningAvg::new();
        let mut val_rng = rand::rngs::StdRng::from_entropy();
        let eval_start = Instant::now();

        for batch_idx in 0..num_batches {
            let samples = val_s.sample_batch_with_ratios(
                config.batch_size,
                &mut val_rng,
                config.final_val_hard_ratio,
                config.final_val_medium_ratio,
            );
            if let Some(m) = eval_samples(
                &val_model,
                encode_fn,
                &samples,
                config.k_negatives,
                config.depth_loss_weight,
                config.loss_type,
                config.margin,
                device,
            ) {
                val_avg.update(&m);
            }
            if batch_idx % 50 == 0 && batch_idx > 0 {
                tracing::info!(
                    batch = batch_idx,
                    total = num_batches,
                    "Full validation progress: {}/{}",
                    batch_idx,
                    num_batches,
                );
            }
        }

        let eval_elapsed = eval_start.elapsed();
        if let Some(final_metrics) = val_avg.avg_metrics() {
            tracing::info!(
                "\n==================================================\n\
                 FINAL UNBIASED VALIDATION EVALUATION\n\
                 ==================================================\n\
                 Total Samples Evaluated : {}\n\
                 Hard Negative Ratio     : {:.2}\n\
                 Medium Negative Ratio   : {:.2}\n\
                 Evaluation Time         : {:.1}s\n\
                 --------------------------------------------------\n\
                 Final Validation Rank   : {:.4}\n\
                 Final Validation Gap    : {:.4}\n\
                 Final Validation Loss   : {:.4}\n\
                 Final Pairwise Accuracy : {:.4}\n\
                 Average Pos Energy      : {:.4}\n\
                 Average Neg Energy      : {:.4}\n\
                 Energy Std              : {:.4}\n\
                 ==================================================",
                val_avg.count * config.batch_size,
                config.final_val_hard_ratio,
                config.final_val_medium_ratio,
                eval_elapsed.as_secs_f64(),
                final_metrics.rank_accuracy,
                final_metrics.energy_gap,
                final_metrics.loss,
                final_metrics.pairwise_acc,
                final_metrics.pos_energy_mean,
                final_metrics.neg_energy_mean,
                final_metrics.energy_std,
            );
            let warnings = final_metrics.health_check();
            if !warnings.is_empty() {
                tracing::warn!("Final validation health warnings: {:?}", warnings);
            }
        } else {
            tracing::warn!("Full validation evaluation produced no metrics (all batches failed encoding)");
        }
    }

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
