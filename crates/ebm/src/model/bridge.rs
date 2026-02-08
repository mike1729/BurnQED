//! Tensor bridge: utilities to convert between candle encoder output
//! (`Vec<f32>`) and burn tensors.
//!
//! This module is the boundary between the candle (encoder) and burn (EBM)
//! frameworks. The candle encoder produces `Vec<f32>` embeddings; burn
//! needs `Tensor<B, 2>` inputs for the EnergyHead.

use burn::prelude::*;
use burn::tensor::TensorData;

/// Convert a batch of f32 embeddings to a burn 2D tensor.
///
/// # Arguments
/// - `embeddings`: slice of vectors, each of dimension `dim`
/// - `device`: burn device to place the tensor on
///
/// # Panics
/// Panics if embeddings is empty or if vectors have inconsistent lengths.
pub fn embeddings_to_tensor<B: Backend>(
    embeddings: &[Vec<f32>],
    device: &B::Device,
) -> Tensor<B, 2> {
    assert!(!embeddings.is_empty(), "embeddings must not be empty");
    let dim = embeddings[0].len();
    assert!(dim > 0, "embedding dimension must be > 0");
    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(
            emb.len(),
            dim,
            "embedding {i} has length {}, expected {dim}",
            emb.len()
        );
    }

    let batch = embeddings.len();
    let flat: Vec<f32> = embeddings.iter().flat_map(|v| v.iter().copied()).collect();
    Tensor::from_data(TensorData::new(flat, [batch, dim]), device)
}

/// Convert a single f32 embedding to a burn 2D tensor of shape `(1, dim)`.
pub fn embedding_to_tensor<B: Backend>(
    embedding: &[f32],
    device: &B::Device,
) -> Tensor<B, 2> {
    let dim = embedding.len();
    assert!(dim > 0, "embedding dimension must be > 0");
    Tensor::from_data(TensorData::new(embedding.to_vec(), [1, dim]), device)
}

/// Extract f64 values from a burn 1D tensor.
pub fn tensor_to_vec<B: Backend>(tensor: Tensor<B, 1>) -> Vec<f64> {
    let data = tensor.into_data();
    data.to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(|v| v as f64)
        .collect()
}

/// Extract a single f64 scalar from a burn 1D tensor.
///
/// # Panics
/// Panics if the tensor does not contain exactly one element.
pub fn tensor_to_f64<B: Backend>(tensor: Tensor<B, 1>) -> f64 {
    let val: f32 = tensor.into_scalar().elem();
    val as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_embeddings_round_trip() {
        let device = Default::default();
        let embeddings = vec![
            vec![1.0_f32, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];

        let tensor = embeddings_to_tensor::<TestBackend>(&embeddings, &device);
        assert_eq!(tensor.dims(), [2, 4]);

        // Extract back and verify values
        let row0: Vec<f32> = tensor
            .clone()
            .slice([0..1, 0..4])
            .reshape([4])
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(row0, vec![1.0, 2.0, 3.0, 4.0]);

        let row1: Vec<f32> = tensor
            .slice([1..2, 0..4])
            .reshape([4])
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(row1, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_embedding_shape() {
        let device = Default::default();
        let embeddings: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0_f32; 64]).collect();

        let tensor = embeddings_to_tensor::<TestBackend>(&embeddings, &device);
        assert_eq!(tensor.dims(), [4, 64]);
    }

    #[test]
    fn test_single_embedding() {
        let device = Default::default();
        let embedding = vec![0.5_f32; 128];

        let tensor = embedding_to_tensor::<TestBackend>(&embedding, &device);
        assert_eq!(tensor.dims(), [1, 128]);

        // Verify first value
        let val: f32 = tensor.slice([0..1, 0..1]).into_scalar().elem();
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_to_vec() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([1.0_f32, 2.5, -3.0]),
            &device,
        );

        let values = tensor_to_vec::<TestBackend>(tensor);
        assert_eq!(values.len(), 3);
        assert!((values[0] - 1.0).abs() < 1e-5);
        assert!((values[1] - 2.5).abs() < 1e-5);
        assert!((values[2] - (-3.0)).abs() < 1e-5);

        // Also test scalar extraction
        let scalar_tensor = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([42.0_f32]),
            &device,
        );
        let scalar = tensor_to_f64::<TestBackend>(scalar_tensor);
        assert!((scalar - 42.0).abs() < 1e-5);
    }
}
