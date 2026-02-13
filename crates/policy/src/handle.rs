//! Cloneable inference handle wrapping [`SglangClient`].
//!
//! [`InferenceHandle`] provides both async and blocking methods for tactic
//! generation and embedding extraction, backed by an SGLang HTTP server.

use std::sync::Arc;

use crate::sglang::SglangClient;
use crate::types::{Embedding, GeneratedTactic};

/// Cloneable handle for LLM inference via SGLang.
///
/// Wraps an `Arc<SglangClient>` so it can be cheaply cloned and shared
/// across search tasks.
#[derive(Clone)]
pub struct InferenceHandle(Arc<SglangClient>);

impl InferenceHandle {
    /// Create a new handle wrapping the given client.
    pub fn new(client: SglangClient) -> Self {
        Self(Arc::new(client))
    }

    /// Generate N tactic candidates (async).
    pub async fn generate_candidates(
        &self,
        proof_state: &str,
        n: usize,
    ) -> anyhow::Result<Vec<GeneratedTactic>> {
        self.0.generate_candidates(proof_state, n).await
    }

    /// Generate N tactic candidates (blocking, for sync callers within a tokio runtime).
    pub fn generate_candidates_blocking(
        &self,
        proof_state: &str,
        n: usize,
    ) -> anyhow::Result<Vec<GeneratedTactic>> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(self.generate_candidates(proof_state, n))
        })
    }

    /// Encode a proof state to an embedding (async).
    pub async fn encode(&self, text: &str) -> anyhow::Result<Embedding> {
        self.0.encode(text).await
    }

    /// Encode a proof state to an embedding (blocking, for sync callers within a tokio runtime).
    pub fn encode_blocking(&self, text: &str) -> anyhow::Result<Embedding> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.encode(text))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_handle_is_clone_send_sync() {
        fn assert_clone_send_sync<T: Clone + Send + Sync>() {}
        assert_clone_send_sync::<InferenceHandle>();
    }
}
