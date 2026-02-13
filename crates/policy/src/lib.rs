//! LLM-based tactic generation and hidden-state extraction for proof search.
//!
//! This crate wraps DeepSeek-Prover-V2-7B (a Llama architecture model) loaded via
//! candle for both autoregressive tactic generation and mean-pooled embedding extraction.
//!
//! # Key types
//!
//! - [`TacticGenerator`] — the main entry point for inference
//! - [`LeanTokenizer`] — HuggingFace tokenizer wrapper
//! - [`PolicyConfig`] — configuration for model loading and generation
//! - [`GeneratedTactic`] — a generated tactic with log-probability
//! - [`Embedding`] — a mean-pooled hidden-state vector

pub mod llama;
pub mod model;
pub mod service;
pub mod tokenizer;
pub mod types;

pub use model::TacticGenerator;
pub use service::{spawn_generation_service, GenerationServiceHandle};
pub use tokenizer::LeanTokenizer;
pub use types::{DeviceConfig, Embedding, GeneratedTactic, PolicyConfig};
