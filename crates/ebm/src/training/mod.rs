//! EBM training pipeline: contrastive data sampling, loss functions,
//! metrics with health checks, and the AdamW training loop.

pub mod cache;
pub mod data;
pub mod loss;
pub mod metrics;
pub mod trainer;
