//! LLM-based tactic generation and hidden-state extraction for proof search.
//!
//! Uses an SGLang HTTP server for inference, providing high-throughput tactic
//! generation via continuous batching and hidden-state extraction for EBM scoring.
//!
//! # Key types
//!
//! - [`SglangClient`] — HTTP client for SGLang inference server
//! - [`InferenceHandle`] — cloneable handle wrapping `SglangClient`
//! - [`GeneratedTactic`] — a generated tactic with log-probability
//! - [`Embedding`] — a mean-pooled hidden-state vector

pub mod handle;
pub mod prompt;
pub mod sglang;
pub mod types;

pub use handle::InferenceHandle;
pub use prompt::{extract_all_tactics, extract_first_tactic, format_tactic_message};
pub use sglang::{SglangClient, SglangConfig};
pub use types::{Embedding, GeneratedTactic};
