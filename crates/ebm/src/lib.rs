//! Energy-Based Model (EBM) for proof state scoring.
//!
//! Provides a trainable spectral-normed MLP energy head that maps encoder
//! embeddings to scalar energy values (lower = more provable). Trained via
//! contrastive (InfoNCE) + depth regression loss on search trajectory data.
//!
//! # Architecture
//!
//! The energy head is a small MLP (`4096 → 512 → 256 → 1`) with spectral
//! normalization and learnable log-temperature. It receives embeddings from
//! an external encoder (injected as a closure), keeping the `ebm` crate
//! independent of the `policy` crate.
//!
//! # Key types
//!
//! - [`EnergyHead`] / [`EnergyHeadConfig`] — the MLP model
//! - [`EBMScorer`] — generic scorer parameterized by burn backend, supports
//!   both per-state (`score_state`) and batch (`score_states_batch`) scoring
//! - [`EBMValueFn`] — backend-erased wrapper for the search engine; supports
//!   `score()` and `score_batch()`. Use `with_batch_encode()` constructor to
//!   supply a batch encode closure for efficient HTTP batching via SGLang.
//! - [`EBMTrainingConfig`] / [`train`] — training loop configuration and entry point
//! - [`ContrastiveSampler`] — loads trajectory data and produces contrastive batches
//! - [`EncoderBackend`] — configuration enum for embedding dimension

pub mod inference;
pub mod model;
pub mod training;

// -- Model re-exports --
pub use model::bridge;
pub use model::encoder::EncoderBackend;
pub use model::energy_head::{EnergyHead, EnergyHeadConfig};
pub use model::spectral_norm::{SpectralNormLinear, SpectralNormLinearConfig};

// -- Inference re-exports --
pub use inference::{EBMScorer, EBMValueFn};

// -- Training re-exports --
pub use training::cache::EmbeddingCache;
pub use training::data::{
    load_records_from_parquet, load_tactic_pairs, load_tactic_pairs_grouped, ContrastiveSample,
    ContrastiveSampler, ProofStateRecord, TacticStep,
};
pub use training::loss::{depth_regression_loss, info_nce_loss};
pub use training::metrics::{EBMMetrics, MetricsHistory};
pub use training::trainer::{
    lr_schedule, resume_from_checkpoint, train, CheckpointMeta, EBMTrainingConfig,
};
