//! Batch scoring of proof states for search integration.
//!
//! Provides `EBMScorer<B>` (generic over burn backend) for scoring proof states
//! using the energy head, and `EBMValueFn` (backend-erased) for use as a trait
//! object in the search engine.

use std::path::Path;

use burn::prelude::*;

use crate::model::bridge::{embedding_to_tensor, embeddings_to_tensor, tensor_to_f64, tensor_to_vec};
use crate::model::energy_head::{EnergyHead, EnergyHeadConfig};
use crate::training::trainer::resume_from_checkpoint;

/// Generic EBM scorer parameterized by burn backend.
///
/// Wraps an `EnergyHead` and an encoder function to score proof states.
/// The `encode_fn` maps proof state strings to `Vec<f32>` embeddings,
/// decoupling this module from the candle encoder.
///
/// Convention: **higher score = more provable** (negated energy).
pub struct EBMScorer<B: Backend> {
    energy_head: EnergyHead<B>,
    encode_fn: Box<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync>,
    device: B::Device,
}

impl<B: Backend> EBMScorer<B> {
    /// Create a new scorer from an existing energy head and encoder function.
    pub fn new(
        energy_head: EnergyHead<B>,
        encode_fn: Box<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync>,
        device: B::Device,
    ) -> Self {
        Self {
            energy_head,
            encode_fn,
            device,
        }
    }

    /// Load a scorer from a checkpoint file.
    ///
    /// Creates a fresh `EnergyHead` from config, loads saved weights,
    /// and pairs it with the given encoder function.
    pub fn load(
        path: &Path,
        config: &EnergyHeadConfig,
        encode_fn: Box<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync>,
        device: B::Device,
    ) -> anyhow::Result<Self> {
        let energy_head = resume_from_checkpoint::<B>(path, config, &device)?;
        Ok(Self {
            energy_head,
            encode_fn,
            device,
        })
    }

    /// Score a single proof state.
    ///
    /// Returns negated energy: higher = more provable.
    pub fn score_state(&self, proof_state: &str) -> anyhow::Result<f64> {
        let embedding = (self.encode_fn)(proof_state)?;
        let tensor = embedding_to_tensor::<B>(&embedding, &self.device);
        let energy = self.energy_head.forward(tensor);
        let energy_val = tensor_to_f64::<B>(energy);
        // Negate: energy head convention is lower = more provable,
        // but ValueScorer convention is higher = more provable.
        Ok(-energy_val)
    }

    /// Score multiple proof states in a batch.
    ///
    /// Returns negated energies: higher = more provable.
    pub fn score_states(&self, states: &[&str]) -> anyhow::Result<Vec<f64>> {
        if states.is_empty() {
            return Ok(vec![]);
        }

        let mut embeddings = Vec::with_capacity(states.len());
        for state in states {
            let emb = (self.encode_fn)(state)?;
            embeddings.push(emb);
        }

        let tensor = embeddings_to_tensor::<B>(&embeddings, &self.device);
        let energies = self.energy_head.forward(tensor);
        let energy_vals = tensor_to_vec::<B>(energies);

        // Negate all: higher = more provable
        Ok(energy_vals.into_iter().map(|e| -e).collect())
    }
}

/// Backend-erased wrapper for EBM scoring.
///
/// Wraps an `EBMScorer<B>` in a `Mutex` behind a closure, erasing the burn
/// backend type parameter. This allows the search engine to use the scorer
/// as a plain trait object without knowing the backend.
///
/// A `Mutex` is needed because burn's `Param` types contain `OnceCell` which
/// is not `Sync`. The `Mutex` provides `Sync` for the `Send`-only scorer.
pub struct EBMValueFn {
    score_fn: Box<dyn Fn(&str) -> anyhow::Result<f64> + Send + Sync>,
}

impl EBMValueFn {
    /// Create a new backend-erased value function from a typed scorer.
    ///
    /// The scorer is wrapped in `Mutex` so the closure is `Send + Sync`.
    pub fn new<B: Backend + 'static>(scorer: EBMScorer<B>) -> Self
    where
        B::Device: Send,
    {
        let scorer = std::sync::Mutex::new(scorer);
        Self {
            score_fn: Box::new(move |state| {
                let scorer = scorer
                    .lock()
                    .map_err(|e| anyhow::anyhow!("EBM scorer lock poisoned: {e}"))?;
                scorer.score_state(state)
            }),
        }
    }

    /// Score a single proof state.
    ///
    /// Returns negated energy: higher = more provable.
    pub fn score(&self, proof_state: &str) -> anyhow::Result<f64> {
        (self.score_fn)(proof_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    /// Create a small energy head for testing.
    fn small_head(device: &<TestBackend as Backend>::Device) -> EnergyHead<TestBackend> {
        crate::model::energy_head::EnergyHeadConfig::new(8)
            .with_d_hidden1(4)
            .with_d_hidden2(2)
            .with_dropout(0.0)
            .init::<TestBackend>(device)
    }

    /// Deterministic mock encode function: produces an 8-dim embedding
    /// based on the hash of the input string.
    fn mock_encode_fn() -> Box<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync> {
        Box::new(|state: &str| {
            let mut emb = vec![0.0_f32; 8];
            for (i, byte) in state.bytes().enumerate() {
                emb[i % 8] += byte as f32 / 255.0;
            }
            Ok(emb)
        })
    }

    #[test]
    fn test_score_state_returns_finite() {
        let device = Default::default();
        let head = small_head(&device);
        let scorer = EBMScorer::new(head, mock_encode_fn(), device);

        let score = scorer.score_state("⊢ True").unwrap();
        assert!(score.is_finite(), "Score should be finite, got {score}");
    }

    #[test]
    fn test_different_states_different_scores() {
        let device = Default::default();
        let head = small_head(&device);
        let scorer = EBMScorer::new(head, mock_encode_fn(), device);

        let score1 = scorer.score_state("⊢ True").unwrap();
        let score2 = scorer
            .score_state("n : Nat\nhyp : n > 0\n⊢ n + 1 > 1")
            .unwrap();

        // Different inputs through the deterministic encode_fn should produce
        // different embeddings and thus different scores.
        assert!(
            (score1 - score2).abs() > 1e-6,
            "Different states should produce different scores: {score1} vs {score2}"
        );
    }

    #[test]
    fn test_score_states_matches_individual() {
        // Note: SpectralNorm Option C re-initializes u/v vectors randomly on
        // each forward() call, so batch vs individual scores won't be identical.
        // We use a generous tolerance (0.5) to account for this variance.
        let device = Default::default();
        let head = small_head(&device);
        let scorer = EBMScorer::new(head, mock_encode_fn(), device);

        let states = ["⊢ True", "⊢ False → False", "n : Nat\n⊢ n = n"];

        let batch_scores = scorer.score_states(&states).unwrap();
        assert_eq!(batch_scores.len(), 3);

        for (i, state) in states.iter().enumerate() {
            let individual = scorer.score_state(state).unwrap();
            assert!(
                (batch_scores[i] - individual).abs() < 0.5,
                "Batch score {i} ({}) should roughly match individual ({})",
                batch_scores[i],
                individual,
            );
        }
    }
}
