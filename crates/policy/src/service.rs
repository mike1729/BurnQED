//! Channel-based generation service that owns the `TacticGenerator` exclusively.
//!
//! Instead of wrapping `TacticGenerator` in `Arc<Mutex<...>>` and suffering from
//! contention under concurrency, this module provides a dedicated service task
//! that processes generation/encode requests sequentially via an mpsc channel.
//!
//! # Usage
//!
//! ```ignore
//! let generator = TacticGenerator::load(&config)?;
//! let handle = spawn_generation_service(generator);
//!
//! // From any async task:
//! let tactics = handle.generate_candidates("⊢ True", 8).await?;
//!
//! // From sync code (inside a tokio runtime):
//! let tactics = handle.generate_candidates_blocking("⊢ True", 8)?;
//! ```

use tokio::sync::{mpsc, oneshot};

use crate::model::TacticGenerator;
use crate::types::{Embedding, GeneratedTactic};

/// A request sent to the generation service.
enum GenerationRequest {
    /// Generate N tactic candidates for a proof state.
    GenerateCandidates {
        proof_state: String,
        n: usize,
        reply: oneshot::Sender<anyhow::Result<Vec<GeneratedTactic>>>,
    },
    /// Encode a proof state to an embedding (for EBM).
    Encode {
        text: String,
        reply: oneshot::Sender<anyhow::Result<Embedding>>,
    },
}

/// Cloneable handle for sending requests to the generation service.
///
/// All clones share the same underlying channel to the service task.
/// When all handles are dropped, the service task shuts down cleanly.
#[derive(Clone)]
pub struct GenerationServiceHandle {
    tx: mpsc::Sender<GenerationRequest>,
}

impl GenerationServiceHandle {
    /// Generate N tactic candidates for a proof state (async).
    pub async fn generate_candidates(
        &self,
        proof_state: &str,
        n: usize,
    ) -> anyhow::Result<Vec<GeneratedTactic>> {
        let wait_start = std::time::Instant::now();
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(GenerationRequest::GenerateCandidates {
                proof_state: proof_state.to_owned(),
                n,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Generation service shut down"))?;
        let result = reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("Generation service dropped reply channel"))?;
        let wait_ms = wait_start.elapsed().as_millis() as u64;
        if wait_ms > 5000 {
            tracing::warn!(wait_ms, n, "Slow generation service response (channel backpressure or long inference)");
        } else if wait_ms > 100 {
            tracing::debug!(wait_ms, n, "Generation service round-trip");
        }
        result
    }

    /// Generate N tactic candidates (blocking, for sync callers within a tokio runtime).
    ///
    /// Uses `block_in_place` + `block_on` to bridge async→sync. This is safe because
    /// callers are expected to be inside a `tokio::spawn` task on a multi-threaded runtime.
    pub fn generate_candidates_blocking(
        &self,
        proof_state: &str,
        n: usize,
    ) -> anyhow::Result<Vec<GeneratedTactic>> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.generate_candidates(proof_state, n))
        })
    }

    /// Encode a proof state to an embedding (async).
    pub async fn encode(&self, text: &str) -> anyhow::Result<Embedding> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(GenerationRequest::Encode {
                text: text.to_owned(),
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Generation service shut down"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("Generation service dropped reply channel"))?
    }

    /// Encode a proof state to an embedding (blocking, for sync callers within a tokio runtime).
    pub fn encode_blocking(&self, text: &str) -> anyhow::Result<Embedding> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.encode(text))
        })
    }
}

/// Spawn the generation service, returning a cloneable handle.
///
/// The service task owns the `TacticGenerator` and processes requests
/// sequentially (FIFO). GPU inference runs on tokio's blocking thread pool,
/// keeping async worker threads free for Lean I/O.
///
/// The channel buffer (64) absorbs burst requests while maintaining
/// natural flow control — senders wait when the buffer is full.
///
/// # Shutdown
///
/// When all `GenerationServiceHandle` clones are dropped, the channel closes
/// and the service task exits cleanly.
pub fn spawn_generation_service(generator: TacticGenerator) -> GenerationServiceHandle {
    let (tx, mut rx) = mpsc::channel::<GenerationRequest>(64);

    tokio::task::spawn_blocking(move || {
        let mut generator = generator;
        while let Some(request) = rx.blocking_recv() {
            match request {
                GenerationRequest::GenerateCandidates {
                    proof_state,
                    n,
                    reply,
                } => {
                    let result = generator.generate_candidates(&proof_state, n);
                    let _ = reply.send(result);
                }
                GenerationRequest::Encode { text, reply } => {
                    let result = generator.encode_only(&text);
                    let _ = reply.send(result);
                }
            }
        }
        tracing::debug!("Generation service shut down (all handles dropped)");
    });

    GenerationServiceHandle { tx }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Unit tests for service channel mechanics are limited without a real
    // TacticGenerator (which requires model weights). The key compile-time
    // checks are:
    // - GenerationServiceHandle is Clone + Send + Sync
    // - spawn_generation_service accepts TacticGenerator by value

    #[test]
    fn test_handle_is_clone_send_sync() {
        fn assert_clone_send_sync<T: Clone + Send + Sync>() {}
        assert_clone_send_sync::<GenerationServiceHandle>();
    }
}
