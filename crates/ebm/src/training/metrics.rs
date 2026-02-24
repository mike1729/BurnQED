//! EBM-specific training metrics with health checks.
//!
//! Computes energy gap, rank accuracy, and various diagnostic metrics
//! from burn tensors. Includes a health check system for detecting
//! training pathologies like mode collapse and polarity inversion.

use burn::prelude::*;
use std::fmt;

/// Comprehensive training metrics for the EBM.
#[derive(Debug, Clone)]
pub struct EBMMetrics {
    /// Total loss (contrastive + depth weighted).
    pub loss: f64,
    /// Contrastive (InfoNCE) loss component.
    pub contrastive_loss: f64,
    /// Depth regression loss component.
    pub depth_loss: f64,
    /// Mean(neg_energy) - Mean(pos_energy). Should be > 0.
    pub energy_gap: f64,
    /// Mean energy of positive (on-path) states.
    pub pos_energy_mean: f64,
    /// Mean energy of negative (dead-end) states.
    pub neg_energy_mean: f64,
    /// Fraction of samples where positive has lowest energy.
    pub rank_accuracy: f64,
    /// Standard deviation across all energies. Watch for collapse → 0.
    pub energy_std: f64,
    /// Fraction of (pos, neg) pairs where E_pos < E_neg.
    pub pairwise_acc: f64,
    /// Fraction of (pos, neg) pairs where margin hinge > 0 (margin violated).
    pub active_fraction: f64,
}

impl EBMMetrics {
    /// Compute metrics from energy tensors and loss values.
    ///
    /// # Arguments
    /// - `pos_energy`: shape `(batch,)` — energy of positive states
    /// - `neg_energies`: shape `(batch, K)` — energies of K negative states per sample
    /// - `contrastive_loss`: scalar contrastive loss value
    /// - `depth_loss`: scalar depth regression loss value
    /// - `total_loss`: scalar total loss value
    pub fn compute<B: Backend>(
        pos_energy: &Tensor<B, 1>,
        neg_energies: &Tensor<B, 2>,
        contrastive_loss: f64,
        depth_loss: f64,
        total_loss: f64,
        margin: f64,
    ) -> Self {
        let [batch_size, _k] = neg_energies.dims();

        // Mean energies
        let pos_energy_mean: f64 = pos_energy.clone().mean().into_scalar().elem();
        let neg_energy_mean: f64 = neg_energies.clone().mean().into_scalar().elem();
        let energy_gap = neg_energy_mean - pos_energy_mean;

        // Rank accuracy: fraction where pos_energy[i] < min(neg_energies[i, :])
        let neg_mins = neg_energies.clone().min_dim(1).squeeze::<1>(1); // (batch,)
        let correct = pos_energy.clone().lower(neg_mins); // bool (batch,)
        let correct_count: f64 = correct.int().sum().into_scalar().elem();
        let rank_accuracy = correct_count / batch_size as f64;

        // Pairwise accuracy: fraction of (pos, neg) pairs where E_pos < E_neg
        let pos_expanded = pos_energy
            .clone()
            .unsqueeze_dim::<2>(1)
            .expand([batch_size, _k]); // (batch, K)
        let pairs_correct = pos_expanded.clone().lower(neg_energies.clone()); // bool (batch, K)
        let pairwise_correct_count: f64 = pairs_correct.int().sum().into_scalar().elem();
        let pairwise_acc = pairwise_correct_count / (batch_size * _k) as f64;

        // Active fraction: % of pairs where margin hinge > 0
        // hinge = max(0, margin + E_pos - E_neg)
        let diff = pos_expanded - neg_energies.clone() + margin;
        let active_count: f64 = diff
            .clamp_min(0.0)
            .greater_elem(0.0)
            .int()
            .sum()
            .into_scalar()
            .elem();
        let active_fraction = active_count / (batch_size * _k) as f64;

        // Energy standard deviation across all energies
        // Cat pos (batch,) and neg (batch*K,) into a single 1D tensor
        let neg_flat = neg_energies.clone().reshape([
            neg_energies.dims()[0] * neg_energies.dims()[1],
        ]); // (batch*K,)
        let all_energies = Tensor::cat(vec![pos_energy.clone(), neg_flat], 0);
        let n: f64 = all_energies.dims()[0] as f64;
        let mean: f64 = all_energies.clone().mean().into_scalar().elem();
        let mean_tensor = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::from([mean as f32]),
            &all_energies.device(),
        );
        let diff = all_energies - mean_tensor;
        let variance: f64 = diff.powf_scalar(2.0).sum().into_scalar().elem();
        let energy_std = (variance / n).sqrt();

        EBMMetrics {
            loss: total_loss,
            contrastive_loss,
            depth_loss,
            energy_gap,
            pos_energy_mean,
            neg_energy_mean,
            rank_accuracy,
            energy_std,
            pairwise_acc,
            active_fraction,
        }
    }

    /// Check for training pathologies. Returns a list of warning messages.
    ///
    /// Warnings:
    /// - Mode collapse: `energy_std < 0.1`
    /// - Negative scoring inversion: `energy_gap < 0.0`
    /// - Too-easy negatives: `rank_accuracy > 0.95 && contrastive_loss > 0.5`
    /// - Energy polarity inverted: `pos_energy_mean > 0 && neg_energy_mean < 0`
    /// - Training diverged: `loss` is NaN or infinite
    pub fn health_check(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.energy_std < 0.1 {
            warnings.push("possible mode collapse: energy_std < 0.1".to_string());
        }
        if self.energy_gap < 0.0 {
            warnings.push(
                "model scoring negatives lower than positives: energy_gap < 0"
                    .to_string(),
            );
        }
        if self.rank_accuracy > 0.95 && self.contrastive_loss > 0.5 {
            warnings.push("negatives may be too easy: high rank_accuracy but high loss".to_string());
        }
        if self.pos_energy_mean > 0.0 && self.neg_energy_mean < 0.0 {
            warnings.push("energy polarity inverted: pos > 0, neg < 0".to_string());
        }
        if self.loss.is_nan() || self.loss.is_infinite() {
            warnings.push("training diverged: loss is NaN or infinite".to_string());
        }
        if self.active_fraction < 0.01 {
            warnings.push("gradient dead: no pairs violate margin".to_string());
        }

        warnings
    }

    /// Single-line display string for logging.
    pub fn display(&self) -> String {
        format!(
            "loss={:.4} gap={:.2} rank={:.2} pair={:.2} active={:.2} pos_e={:.2} neg_e={:.2} std={:.2}",
            self.loss,
            self.energy_gap,
            self.rank_accuracy,
            self.pairwise_acc,
            self.active_fraction,
            self.pos_energy_mean,
            self.neg_energy_mean,
            self.energy_std,
        )
    }
}

impl fmt::Display for EBMMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display())
    }
}

/// Tracks metrics history across training steps.
pub struct MetricsHistory {
    history: Vec<(usize, EBMMetrics)>,
}

impl MetricsHistory {
    /// Create a new empty history.
    pub fn new() -> Self {
        MetricsHistory {
            history: Vec::new(),
        }
    }

    /// Record metrics at a given training step.
    ///
    /// # Arguments
    /// - `step`: the training step number (used for logging/plotting).
    /// - `metrics`: the [`EBMMetrics`] snapshot for this step.
    pub fn push(&mut self, step: usize, metrics: EBMMetrics) {
        self.history.push((step, metrics));
    }

    /// Returns the most recent [`EBMMetrics`] entry, or `None` if empty.
    pub fn last(&self) -> Option<&EBMMetrics> {
        self.history.last().map(|(_, m)| m)
    }

    /// Check if training is improving over the last `window` entries.
    ///
    /// Compares the energy gap of the oldest and newest entries within the
    /// window. Returns `true` if the gap increased (model is learning to
    /// separate positive from negative states), `false` otherwise or if
    /// fewer than 2 entries exist.
    ///
    /// # Arguments
    /// - `window`: number of recent entries to consider. Must be >= 2.
    pub fn is_improving(&self, window: usize) -> bool {
        if self.history.len() < 2 || window < 2 {
            return false;
        }

        let start = self.history.len().saturating_sub(window);
        let entries = &self.history[start..];
        if entries.len() < 2 {
            return false;
        }

        // Simple check: is the last entry's gap > first entry's gap?
        let first_gap = entries[0].1.energy_gap;
        let last_gap = entries[entries.len() - 1].1.energy_gap;
        last_gap > first_gap
    }

    /// Returns the number of recorded entries.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Returns `true` if no metrics have been recorded.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

impl Default for MetricsHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::tensor::TensorData;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_compute_perfect() {
        let device = Default::default();
        // Positive energies all -5 (low = good), negative energies all +5 (high = bad)
        let pos = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([-5.0_f32, -5.0, -5.0, -5.0]),
            &device,
        );
        let neg = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [5.0_f32, 5.0],
                [5.0, 5.0],
                [5.0, 5.0],
                [5.0, 5.0],
            ]),
            &device,
        );

        let metrics = EBMMetrics::compute(&pos, &neg, 0.01, 0.0, 0.01, 1.0);

        assert!(
            (metrics.rank_accuracy - 1.0).abs() < 1e-6,
            "Perfect separation should give rank_accuracy=1.0, got {}",
            metrics.rank_accuracy
        );
        assert!(
            (metrics.energy_gap - 10.0).abs() < 0.1,
            "Gap should be ≈10.0, got {}",
            metrics.energy_gap
        );
        assert!(
            (metrics.pos_energy_mean - (-5.0)).abs() < 0.1,
            "pos_energy_mean should be ≈-5.0, got {}",
            metrics.pos_energy_mean
        );
        assert!(
            (metrics.neg_energy_mean - 5.0).abs() < 0.1,
            "neg_energy_mean should be ≈5.0, got {}",
            metrics.neg_energy_mean
        );
        // With gap=10 and margin=1, all pairs satisfy margin → pairwise_acc=1.0
        assert!(
            (metrics.pairwise_acc - 1.0).abs() < 1e-6,
            "Perfect separation should give pairwise_acc=1.0, got {}",
            metrics.pairwise_acc
        );
        // margin + E_pos - E_neg = 1 + (-5) - 5 = -9 → hinge = 0, no active pairs
        assert!(
            metrics.active_fraction < 1e-6,
            "Perfect separation (gap >> margin) should give active_fraction≈0.0, got {}",
            metrics.active_fraction
        );
    }

    #[test]
    fn test_compute_random() {
        let device = Default::default();
        // All energies equal → positive rarely beats negative on strict less-than
        let pos = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([0.0_f32, 0.0, 0.0, 0.0]),
            &device,
        );
        let neg = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.0_f32, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]),
            &device,
        );

        let metrics = EBMMetrics::compute(&pos, &neg, 1.0, 0.0, 1.0, 1.0);

        // With equal energies, pos < neg_min is false → rank_accuracy = 0
        assert!(
            metrics.rank_accuracy < 0.01,
            "Equal energies: rank_accuracy should be ~0, got {}",
            metrics.rank_accuracy
        );
        assert!(
            metrics.energy_gap.abs() < 0.01,
            "Equal energies: gap should be ~0, got {}",
            metrics.energy_gap
        );
        // Equal energies: strict less-than → pairwise_acc = 0
        assert!(
            metrics.pairwise_acc < 0.01,
            "Equal energies: pairwise_acc should be ~0, got {}",
            metrics.pairwise_acc
        );
        // margin + 0 - 0 = 1.0 > 0 → all pairs active
        assert!(
            (metrics.active_fraction - 1.0).abs() < 1e-6,
            "Equal energies with margin=1: active_fraction should be 1.0, got {}",
            metrics.active_fraction
        );
    }

    #[test]
    fn test_health_check_mode_collapse() {
        let metrics = EBMMetrics {
            loss: 0.5,
            contrastive_loss: 0.5,
            depth_loss: 0.0,
            energy_gap: 0.01,
            pos_energy_mean: 0.0,
            neg_energy_mean: 0.01,
            rank_accuracy: 0.5,
            energy_std: 0.01, // very low → mode collapse
            pairwise_acc: 0.5,
            active_fraction: 0.5,
        };

        let warnings = metrics.health_check();
        assert!(
            warnings.iter().any(|w| w.contains("mode collapse")),
            "Should warn about mode collapse, got: {:?}",
            warnings
        );
    }

    #[test]
    fn test_health_check_healthy() {
        let metrics = EBMMetrics {
            loss: 0.3,
            contrastive_loss: 0.3,
            depth_loss: 0.0,
            energy_gap: 2.5,
            pos_energy_mean: -1.0,
            neg_energy_mean: 1.5,
            rank_accuracy: 0.8,
            energy_std: 1.2,
            pairwise_acc: 0.9,
            active_fraction: 0.3,
        };

        let warnings = metrics.health_check();
        assert!(
            warnings.is_empty(),
            "Healthy metrics should produce no warnings, got: {:?}",
            warnings
        );
    }
}
