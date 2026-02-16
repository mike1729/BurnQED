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
/// decoupling this module from the LLM encoder.
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

    /// Score multiple proof states using a separate batch encode function.
    ///
    /// Unlike [`score_states`], which calls `encode_fn` once per state,
    /// this method accepts a batch encode function that can encode all
    /// states in a single HTTP call (e.g. via SGLang's `/encode` batch endpoint).
    ///
    /// Returns negated energies: higher = more provable.
    pub fn score_states_batch(
        &self,
        states: &[&str],
        batch_encode: &dyn Fn(&[&str]) -> anyhow::Result<Vec<Vec<f32>>>,
    ) -> anyhow::Result<Vec<f64>> {
        if states.is_empty() {
            return Ok(vec![]);
        }

        let embeddings = batch_encode(states)?;
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
///
/// **Important**: When calling `score()` from an async context (e.g. tokio),
/// callers MUST wrap the call in `block_in_place` to avoid thread starvation.
/// Without this, concurrent tasks blocking on `mutex.lock()` starve the tokio
/// thread pool, preventing HTTP futures from completing — a classic deadlock.
/// See `search::adapters::ValueScorer for EBMValueFn` for the correct pattern.
pub struct EBMValueFn {
    score_fn: Box<dyn Fn(&str) -> anyhow::Result<f64> + Send + Sync>,
    score_batch_fn: Box<dyn Fn(&[&str]) -> anyhow::Result<Vec<f64>> + Send + Sync>,
}

impl EBMValueFn {
    /// Create a new backend-erased value function from a typed scorer.
    ///
    /// The scorer is wrapped in `Arc<Mutex>` so both closures can share it.
    pub fn new<B: Backend + 'static>(scorer: EBMScorer<B>) -> Self
    where
        B::Device: Send,
    {
        let scorer = std::sync::Arc::new(std::sync::Mutex::new(scorer));
        let scorer2 = scorer.clone();
        Self {
            score_fn: Box::new(move |state| {
                let scorer = scorer
                    .lock()
                    .map_err(|e| anyhow::anyhow!("EBM scorer lock poisoned: {e}"))?;
                scorer.score_state(state)
            }),
            score_batch_fn: Box::new(move |states| {
                let scorer = scorer2
                    .lock()
                    .map_err(|e| anyhow::anyhow!("EBM scorer lock poisoned: {e}"))?;
                scorer.score_states(states)
            }),
        }
    }

    /// Create a new backend-erased value function with a custom batch encode function.
    ///
    /// The `batch_encode_fn` is used by `score_batch()` to encode all states
    /// in a single call (e.g. via SGLang's batch encode endpoint), while
    /// `score()` still uses the per-state `encode_fn` inside the scorer.
    pub fn with_batch_encode<B: Backend + 'static>(
        scorer: EBMScorer<B>,
        batch_encode_fn: Box<dyn Fn(&[&str]) -> anyhow::Result<Vec<Vec<f32>>> + Send + Sync>,
    ) -> Self
    where
        B::Device: Send,
    {
        let scorer = std::sync::Arc::new(std::sync::Mutex::new(scorer));
        let scorer2 = scorer.clone();
        Self {
            score_fn: Box::new(move |state| {
                let scorer = scorer
                    .lock()
                    .map_err(|e| anyhow::anyhow!("EBM scorer lock poisoned: {e}"))?;
                scorer.score_state(state)
            }),
            score_batch_fn: Box::new(move |states| {
                let scorer = scorer2
                    .lock()
                    .map_err(|e| anyhow::anyhow!("EBM scorer lock poisoned: {e}"))?;
                scorer.score_states_batch(states, &*batch_encode_fn)
            }),
        }
    }

    /// Score a single proof state.
    ///
    /// Returns negated energy: higher = more provable.
    pub fn score(&self, proof_state: &str) -> anyhow::Result<f64> {
        (self.score_fn)(proof_state)
    }

    /// Score multiple proof states in one call.
    ///
    /// When constructed with [`with_batch_encode`], this uses a single
    /// batch HTTP call for all embeddings. Otherwise falls back to
    /// sequential per-state scoring via the scorer's `encode_fn`.
    pub fn score_batch(&self, proof_states: &[&str]) -> anyhow::Result<Vec<f64>> {
        (self.score_batch_fn)(proof_states)
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

    #[test]
    fn test_score_states_batch_with_batch_encode() {
        let device = Default::default();
        let head = small_head(&device);
        let scorer = EBMScorer::new(head, mock_encode_fn(), device);

        let batch_encode = |states: &[&str]| -> anyhow::Result<Vec<Vec<f32>>> {
            states
                .iter()
                .map(|state| {
                    let mut emb = vec![0.0_f32; 8];
                    for (i, byte) in state.bytes().enumerate() {
                        emb[i % 8] += byte as f32 / 255.0;
                    }
                    Ok(emb)
                })
                .collect()
        };

        let states = ["⊢ True", "⊢ False"];
        let scores = scorer.score_states_batch(&states, &batch_encode).unwrap();
        assert_eq!(scores.len(), 2);
        for s in &scores {
            assert!(s.is_finite(), "Score should be finite, got {s}");
        }
    }

    #[test]
    fn test_score_states_batch_empty() {
        let device = Default::default();
        let head = small_head(&device);
        let scorer = EBMScorer::new(head, mock_encode_fn(), device);

        let batch_encode = |_: &[&str]| -> anyhow::Result<Vec<Vec<f32>>> {
            panic!("Should not be called for empty input");
        };

        let scores = scorer.score_states_batch(&[], &batch_encode).unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    fn test_ebm_value_fn_score_batch() {
        let device = Default::default();
        let head = small_head(&device);
        let scorer = EBMScorer::new(head, mock_encode_fn(), device);
        let value_fn = EBMValueFn::new(scorer);

        let states = ["⊢ True", "⊢ False"];
        let scores = value_fn.score_batch(&states).unwrap();
        assert_eq!(scores.len(), 2);
        for s in &scores {
            assert!(s.is_finite(), "Score should be finite, got {s}");
        }
    }

    #[test]
    fn test_ebm_value_fn_with_batch_encode() {
        let device = Default::default();
        let head = small_head(&device);
        let scorer = EBMScorer::new(head, mock_encode_fn(), device);

        let batch_encode: Box<dyn Fn(&[&str]) -> anyhow::Result<Vec<Vec<f32>>> + Send + Sync> =
            Box::new(|states: &[&str]| {
                states
                    .iter()
                    .map(|state| {
                        let mut emb = vec![0.0_f32; 8];
                        for (i, byte) in state.bytes().enumerate() {
                            emb[i % 8] += byte as f32 / 255.0;
                        }
                        Ok(emb)
                    })
                    .collect()
            });

        let value_fn = EBMValueFn::with_batch_encode(scorer, batch_encode);

        // Single score still works
        let s = value_fn.score("⊢ True").unwrap();
        assert!(s.is_finite());

        // Batch score works
        let scores = value_fn.score_batch(&["⊢ True", "⊢ False"]).unwrap();
        assert_eq!(scores.len(), 2);
        for s in &scores {
            assert!(s.is_finite());
        }
    }
}
