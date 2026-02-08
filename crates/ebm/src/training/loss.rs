//! InfoNCE + depth regression losses for EBM training.
//!
//! Both loss functions are generic over `B: Backend` and operate on burn tensors.
//! Convention: lower energy = more provable state.

use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;

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
