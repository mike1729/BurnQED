//! Contrastive + depth regression losses for EBM training.
//!
//! Loss functions are generic over `B: Backend` and operate on burn tensors.
//! Convention: lower energy = more provable state.
//!
//! Two contrastive loss variants:
//! - **InfoNCE**: softmax-based, maximizes probability of positive among K+1 candidates
//! - **Margin ranking**: pairwise hinge loss, enforces fixed energy gap between pos/neg

use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;

/// Which contrastive loss to use during EBM training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ContrastiveLossType {
    /// Softmax-based InfoNCE loss. Good at learning relative rankings but can
    /// cause energy score drift (unbounded separation over long training).
    InfoNCE,
    /// Max-margin ranking loss: `L = mean(max(0, margin + E(x+) - E(x-)))`.
    /// Produces bounded, stable energy scores since loss becomes 0 once the
    /// margin is satisfied. Focuses gradient on hard negatives only.
    MarginRanking,
}

impl Default for ContrastiveLossType {
    fn default() -> Self {
        Self::InfoNCE
    }
}

impl std::fmt::Display for ContrastiveLossType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InfoNCE => write!(f, "info_nce"),
            Self::MarginRanking => write!(f, "margin_ranking"),
        }
    }
}

impl std::str::FromStr for ContrastiveLossType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "info_nce" | "infonce" | "InfoNCE" => Ok(Self::InfoNCE),
            "margin_ranking" | "margin" | "MarginRanking" => Ok(Self::MarginRanking),
            _ => Err(format!("unknown loss type: {s:?} (expected info_nce or margin_ranking)")),
        }
    }
}

/// InfoNCE contrastive loss.
///
/// Treats each positive energy as the "correct class" and K negative energies
/// as distractors. Energies are negated to form logits (lower energy → higher logit).
///
/// # Arguments
/// - `pos_energy`: shape `(batch,)` — energy of positive (on-path) states
/// - `neg_energies`: shape `(batch, K)` — energies of K negative states per sample
///
/// # Returns
/// Scalar loss tensor of shape `(1,)`.
pub fn info_nce_loss<B: Backend>(
    pos_energy: Tensor<B, 1>,
    neg_energies: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let device = pos_energy.device();
    let [batch_size, _k] = neg_energies.dims();

    // Negate energies to get logits (lower energy = higher logit = "more likely correct")
    let pos_logits = pos_energy.neg().unsqueeze_dim::<2>(1); // (batch, 1)
    let neg_logits = neg_energies.neg(); // (batch, K)

    // Concatenate: positive at index 0, then K negatives
    let logits = Tensor::cat(vec![pos_logits, neg_logits], 1); // (batch, K+1)

    // Labels: class 0 (positive) is correct for every sample
    let labels =
        Tensor::<B, 1, Int>::zeros([batch_size], &device);

    CrossEntropyLossConfig::new()
        .init(&device)
        .forward(logits, labels)
}

/// Max-margin ranking loss for contrastive EBM training.
///
/// For each (positive, negative) pair, computes `max(0, margin + E(x+) - E(x-))`.
/// Loss is zero once the negative energy exceeds the positive by at least `margin`,
/// producing stable, bounded energy scores. Gradients focus exclusively on hard
/// negatives that violate the margin.
///
/// Only pairs that violate the margin (hinge > 0) contribute to the loss average.
/// This prevents easy negatives (already past the margin) from diluting the gradient
/// signal on the hard negatives that actually need pushing.
///
/// # Arguments
/// - `pos_energy`: shape `(batch,)` — energy of positive (on-path) states
/// - `neg_energies`: shape `(batch, K)` — energies of K negative states per sample
/// - `margin`: the required minimum gap E(x-) - E(x+) >= margin
///
/// # Returns
/// Scalar loss tensor of shape `(1,)`.
pub fn margin_ranking_loss<B: Backend>(
    pos_energy: Tensor<B, 1>,
    neg_energies: Tensor<B, 2>,
    margin: f64,
) -> Tensor<B, 1> {
    let [batch_size, k] = neg_energies.dims();

    // Broadcast pos_energy to (batch, K)
    let pos_expanded = pos_energy.unsqueeze_dim::<2>(1).expand([batch_size, k]);

    // hinge = max(0, margin + E(x+) - E(x-))
    let diff = pos_expanded - neg_energies + margin;
    let hinge = diff.clamp_min(0.0);

    // Mean over only active (margin-violating) pairs to avoid dilution
    let active = hinge.clone().greater_elem(0.0).float();
    let num_active = active.clone().sum();
    hinge.mul(active).sum() / (num_active + 1e-8)
}

/// Compute the contrastive loss based on the selected loss type.
///
/// Dispatches to either `info_nce_loss` or `margin_ranking_loss`.
pub fn contrastive_loss<B: Backend>(
    loss_type: ContrastiveLossType,
    pos_energy: Tensor<B, 1>,
    neg_energies: Tensor<B, 2>,
    margin: f64,
) -> Tensor<B, 1> {
    match loss_type {
        ContrastiveLossType::InfoNCE => info_nce_loss(pos_energy, neg_energies),
        ContrastiveLossType::MarginRanking => margin_ranking_loss(pos_energy, neg_energies, margin),
    }
}

/// Depth regression loss: MSE between energy and normalized remaining depth.
///
/// Only states with known remaining depth (>= 0) contribute. States with
/// `remaining_depth == -1` are masked out.
///
/// # Arguments
/// - `energy`: shape `(batch,)` — predicted energy values
/// - `remaining_depth`: shape `(batch,)` — ground-truth remaining depth (-1 for unknown)
///
/// # Returns
/// Scalar loss tensor of shape `(1,)`. Returns 0 if no valid depths exist.
pub fn depth_regression_loss<B: Backend>(
    energy: Tensor<B, 1>,
    remaining_depth: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let device = energy.device();

    // Create mask for valid entries (remaining_depth >= 0)
    let zeros = Tensor::<B, 1>::zeros(remaining_depth.dims(), &device);
    let mask = remaining_depth.clone().greater_equal(zeros); // bool tensor (batch,)
    let mask_float = Tensor::<B, 1>::from_data(
        mask.int().into_data(),
        &device,
    ); // (batch,) as float

    // Count valid entries, clamp to avoid div-by-zero
    let valid_count = mask_float.clone().sum();
    let valid_count_clamped = valid_count.clone().clamp_min(1.0);

    // Normalize remaining_depth to [0, 1] within batch
    let depth_max = remaining_depth.clone().max().clamp_min(1.0);
    let normalized_depth = remaining_depth / depth_max;

    // MSE between energy and normalized depth, masked
    let diff = energy - normalized_depth;
    let squared = diff.powf_scalar(2.0);
    let masked_squared = squared * mask_float;
    let mean_loss = masked_squared.sum() / valid_count_clamped;

    // If no valid entries, return 0
    let has_valid = valid_count.greater_elem(0.0);
    let has_valid_float = Tensor::<B, 1>::from_data(
        has_valid.int().into_data(),
        &device,
    );
    mean_loss * has_valid_float
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;
    use burn::tensor::TensorData;

    type TestBackend = NdArray<f32>;
    type TestAutodiffBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_info_nce_perfect_separation() {
        let device = Default::default();
        // Positive energies very low (good), negative energies very high (bad)
        let pos = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([-10.0_f32, -10.0]),
            &device,
        );
        let neg = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[10.0_f32, 10.0], [10.0, 10.0]]),
            &device,
        );

        let loss: f32 = info_nce_loss(pos, neg).into_scalar().elem();
        assert!(
            loss < 0.01,
            "Perfect separation should give near-zero loss, got {loss}"
        );
    }

    #[test]
    fn test_info_nce_no_separation() {
        let device = Default::default();
        // All energies equal → loss = ln(K+1) = ln(3) ≈ 1.099
        let pos = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([0.0_f32, 0.0]),
            &device,
        );
        let neg = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0_f32, 0.0], [0.0, 0.0]]),
            &device,
        );

        let loss: f32 = info_nce_loss(pos, neg).into_scalar().elem();
        let expected = (3.0_f32).ln(); // ln(3) ≈ 1.099
        assert!(
            (loss - expected).abs() < 0.05,
            "No separation: expected ≈{expected}, got {loss}"
        );
    }

    #[test]
    fn test_info_nce_wrong_separation() {
        let device = Default::default();
        // Positive energy high (bad), negative energy low (good) → high loss
        let pos = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([10.0_f32, 10.0]),
            &device,
        );
        let neg = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[-10.0_f32, -10.0], [-10.0, -10.0]]),
            &device,
        );

        let loss: f32 = info_nce_loss(pos, neg).into_scalar().elem();
        assert!(
            loss > 2.0,
            "Wrong separation should give high loss, got {loss}"
        );
    }

    #[test]
    fn test_info_nce_gradient_direction() {
        let device = Default::default();
        // Positive energy at 0, negatives at 0 → gradient should push pos down, neg up
        let pos = Tensor::<TestAutodiffBackend, 1>::from_data(
            TensorData::from([0.0_f32, 0.0]),
            &device,
        )
        .require_grad();
        let neg = Tensor::<TestAutodiffBackend, 2>::from_data(
            TensorData::from([[0.0_f32, 0.0], [0.0, 0.0]]),
            &device,
        )
        .require_grad();

        let loss = info_nce_loss(pos.clone(), neg.clone());
        let grads = loss.backward();

        // dL/d(pos_energy): logit = -energy, so dL/d(energy) = -dL/d(logit).
        // dL/d(logit_pos) = softmax(0) - 1 = 1/3 - 1 = -2/3
        // dL/d(pos_energy) = -(-2/3) = +2/3 per sample, averaged = 1/3 ≈ 0.333
        // Positive gradient → SGD (subtract lr*grad) decreases energy → correct!
        let pos_grad = pos.grad(&grads).unwrap();
        let pos_grad_data: Vec<f32> = pos_grad.into_data().to_vec().unwrap();
        for &g in &pos_grad_data {
            assert!(
                g > 0.0,
                "Positive energy gradient should be positive (SGD will decrease energy), got {g}"
            );
        }

        // Negative energy gradient should be negative → SGD will increase energy
        let neg_grad = neg.grad(&grads).unwrap();
        let neg_grad_data: Vec<f32> = neg_grad.into_data().to_vec().unwrap();
        for &g in &neg_grad_data {
            assert!(
                g < 0.0,
                "Negative energy gradient should be negative (SGD will increase energy), got {g}"
            );
        }
    }

    #[test]
    fn test_margin_ranking_satisfied() {
        let device = Default::default();
        // Positive energy -5, negatives +5 → gap = 10, margin = 1.0 → loss = 0
        let pos = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([-5.0_f32, -5.0]),
            &device,
        );
        let neg = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[5.0_f32, 5.0], [5.0, 5.0]]),
            &device,
        );

        let loss: f32 = margin_ranking_loss(pos, neg, 1.0).into_scalar().elem();
        assert!(
            loss.abs() < 1e-6,
            "Margin satisfied (gap=10, margin=1) should give zero loss, got {loss}"
        );
    }

    #[test]
    fn test_margin_ranking_violated() {
        let device = Default::default();
        // Positive energy 0, negatives 0 → hinge = max(0, 1.0 + 0 - 0) = 1.0
        let pos = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([0.0_f32, 0.0]),
            &device,
        );
        let neg = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0_f32, 0.0], [0.0, 0.0]]),
            &device,
        );

        let loss: f32 = margin_ranking_loss(pos, neg, 1.0).into_scalar().elem();
        assert!(
            (loss - 1.0).abs() < 0.01,
            "Equal energies with margin=1.0 should give loss≈1.0, got {loss}"
        );
    }

    #[test]
    fn test_margin_ranking_wrong_order() {
        let device = Default::default();
        // Positive energy +10, negatives -10 → hinge = max(0, 1.0 + 10 - (-10)) = 21.0
        let pos = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([10.0_f32]),
            &device,
        );
        let neg = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[-10.0_f32, -10.0]]),
            &device,
        );

        let loss: f32 = margin_ranking_loss(pos, neg, 1.0).into_scalar().elem();
        assert!(
            (loss - 21.0).abs() < 0.1,
            "Wrong order should give high loss≈21.0, got {loss}"
        );
    }

    #[test]
    fn test_margin_ranking_gradient_direction() {
        let device = Default::default();
        let pos = Tensor::<TestAutodiffBackend, 1>::from_data(
            TensorData::from([0.0_f32, 0.0]),
            &device,
        )
        .require_grad();
        let neg = Tensor::<TestAutodiffBackend, 2>::from_data(
            TensorData::from([[0.0_f32, 0.0], [0.0, 0.0]]),
            &device,
        )
        .require_grad();

        let loss = margin_ranking_loss(pos.clone(), neg.clone(), 1.0);
        let grads = loss.backward();

        // Gradient on pos_energy should be positive (SGD decreases it → good)
        let pos_grad: Vec<f32> = pos.grad(&grads).unwrap().into_data().to_vec().unwrap();
        for &g in &pos_grad {
            assert!(
                g > 0.0,
                "Positive energy gradient should be positive, got {g}"
            );
        }

        // Gradient on neg_energies should be negative (SGD increases it → good)
        let neg_grad: Vec<f32> = neg.grad(&grads).unwrap().into_data().to_vec().unwrap();
        for &g in &neg_grad {
            assert!(
                g < 0.0,
                "Negative energy gradient should be negative, got {g}"
            );
        }
    }

    #[test]
    fn test_contrastive_loss_dispatch() {
        let device = Default::default();
        let pos = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([-5.0_f32, -5.0]),
            &device,
        );
        let neg = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[5.0_f32, 5.0], [5.0, 5.0]]),
            &device,
        );

        // InfoNCE dispatch
        let loss_nce: f32 = contrastive_loss(
            ContrastiveLossType::InfoNCE, pos.clone(), neg.clone(), 1.0,
        ).into_scalar().elem();
        let loss_nce_direct: f32 = info_nce_loss(pos.clone(), neg.clone()).into_scalar().elem();
        assert!((loss_nce - loss_nce_direct).abs() < 1e-6, "Dispatch should match direct call");

        // Margin dispatch
        let loss_margin: f32 = contrastive_loss(
            ContrastiveLossType::MarginRanking, pos.clone(), neg.clone(), 1.0,
        ).into_scalar().elem();
        let loss_margin_direct: f32 = margin_ranking_loss(pos, neg, 1.0).into_scalar().elem();
        assert!((loss_margin - loss_margin_direct).abs() < 1e-6, "Dispatch should match direct call");
    }

    #[test]
    fn test_depth_loss_perfect() {
        let device = Default::default();
        // Energy matches normalized depth exactly → loss ≈ 0
        // Depths: [4, 2, 0], max=4, normalized: [1.0, 0.5, 0.0]
        let energy = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([1.0_f32, 0.5, 0.0]),
            &device,
        );
        let depth = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([4.0_f32, 2.0, 0.0]),
            &device,
        );

        let loss: f32 = depth_regression_loss(energy, depth).into_scalar().elem();
        assert!(
            loss < 0.01,
            "Perfect match should give near-zero loss, got {loss}"
        );
    }

    #[test]
    fn test_depth_loss_with_unknowns() {
        let device = Default::default();
        // Mix of known and unknown depths — unknowns should be masked
        let energy = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([1.0_f32, 0.5, 999.0, 0.0]),
            &device,
        );
        let depth = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([4.0_f32, 2.0, -1.0, 0.0]),
            &device,
        );

        let loss: f32 = depth_regression_loss(energy, depth).into_scalar().elem();
        // The 999.0 energy at index 2 should be masked out (depth=-1)
        // Without masking, loss would be enormous
        assert!(
            loss < 10.0,
            "Unknown depths should be masked, got loss={loss}"
        );
    }

    #[test]
    fn test_depth_loss_all_unknown() {
        let device = Default::default();
        // All depths unknown → loss should be 0
        let energy = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([5.0_f32, 3.0, 1.0]),
            &device,
        );
        let depth = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([-1.0_f32, -1.0, -1.0]),
            &device,
        );

        let loss: f32 = depth_regression_loss(energy, depth).into_scalar().elem();
        assert!(
            loss.abs() < 1e-6,
            "All unknown depths should give zero loss, got {loss}"
        );
    }
}
