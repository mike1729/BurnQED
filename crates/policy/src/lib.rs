//! LLM-based tactic generation and hidden-state extraction for proof search.
//!
//! This crate provides two inference backends:
//!
//! - **Local candle**: In-process inference via [`TacticGenerator`] (for EBM training)
//! - **SGLang HTTP**: Remote server inference via [`SglangClient`] (for search/eval)
//!
//! # Key types
//!
//! - [`SglangClient`] — HTTP client for SGLang inference server
//! - [`InferenceHandle`] — cloneable handle wrapping `SglangClient`
//! - [`TacticGenerator`] — local candle inference (used by EBM training)
//! - [`PolicyConfig`] — configuration for local model loading
//! - [`GeneratedTactic`] — a generated tactic with log-probability
//! - [`Embedding`] — a mean-pooled hidden-state vector

pub mod handle;
pub mod llama;
pub mod model;
pub mod service;
pub mod sglang;
pub mod tokenizer;
pub mod types;

pub use handle::InferenceHandle;
pub use model::{extract_first_tactic, format_tactic_message, TacticGenerator};
pub use service::{spawn_generation_service, GenerationServiceHandle};
pub use sglang::{SglangClient, SglangConfig};
pub use tokenizer::LeanTokenizer;
pub use types::{DeviceConfig, Embedding, GeneratedTactic, PolicyConfig};
