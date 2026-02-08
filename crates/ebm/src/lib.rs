//! Energy-Based Model (EBM) for proof state scoring.
//!
//! Provides a trainable spectral-normed MLP energy head that maps encoder
//! embeddings to scalar energy values (lower = more provable). Trained via
//! contrastive (InfoNCE) + depth regression loss on search trajectory data.

pub mod inference;
pub mod model;
pub mod training;
