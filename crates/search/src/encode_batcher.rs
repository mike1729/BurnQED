//! Global request coalescing for cross-search EBM encode batching.
//!
//! Mirrors [`GlobalBatcher`](crate::batcher::GlobalBatcher) but for
//! embedding encode requests. Multiple concurrent search tasks submit
//! small encode batches independently; `GlobalEncodeBatcher` collects
//! them into a shared channel and flushes as one large HTTP batch call.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};

/// Async trait for batch encoding proof states to embeddings.
#[async_trait]
pub trait BatchEncoder: Send + Sync {
    /// Encode multiple proof states into embedding vectors in a single call.
    async fn encode_batch(&self, states: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
}

/// A coalescing request: one or more states to encode.
struct EncodeCoalesceRequest {
    states: Vec<String>,
    tx: oneshot::Sender<Result<Vec<Vec<f32>>, String>>,
}

/// Middleware that coalesces encode requests from multiple concurrent
/// search tasks into single GPU-efficient batches.
///
/// Spawns a background tokio task that accumulates requests and flushes
/// when either `max_batch_states` is reached or `linger` duration elapses.
///
/// Callers use [`encode_batch_blocking`](Self::encode_batch_blocking) from
/// within `block_in_place` contexts (the same pattern as `EBMValueFn`).
pub struct GlobalEncodeBatcher {
    sender: mpsc::Sender<EncodeCoalesceRequest>,
}

impl GlobalEncodeBatcher {
    /// Create a new `GlobalEncodeBatcher` wrapping the given inner encoder.
    ///
    /// - `max_batch_states`: flush when accumulated states reach this count
    /// - `linger`: maximum time to wait for additional requests before flushing
    pub fn new(
        inner: Arc<dyn BatchEncoder>,
        max_batch_states: usize,
        linger: Duration,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<EncodeCoalesceRequest>(max_batch_states * 4);
        tokio::spawn(coalesce_loop(rx, inner, max_batch_states, linger));
        GlobalEncodeBatcher { sender: tx }
    }

    /// Submit a batch of states for encoding (blocking).
    ///
    /// Safe to call from within `block_in_place`. Uses `blocking_send`
    /// and `blocking_recv` — no `Handle::block_on` needed.
    pub fn encode_batch_blocking(&self, states: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        if states.is_empty() {
            return Ok(vec![]);
        }

        let (tx, rx) = oneshot::channel();
        let req = EncodeCoalesceRequest {
            states: states.iter().map(|s| s.to_string()).collect(),
            tx,
        };

        self.sender
            .blocking_send(req)
            .map_err(|_| anyhow::anyhow!("GlobalEncodeBatcher background task gone"))?;

        let result = rx
            .blocking_recv()
            .map_err(|_| anyhow::anyhow!("GlobalEncodeBatcher response dropped"))?;

        result.map_err(|e| anyhow::anyhow!(e))
    }
}

/// Background task that collects encode requests and flushes them in batches.
async fn coalesce_loop(
    mut rx: mpsc::Receiver<EncodeCoalesceRequest>,
    inner: Arc<dyn BatchEncoder>,
    max_batch_states: usize,
    linger: Duration,
) {
    let mut buffer: Vec<EncodeCoalesceRequest> = Vec::new();

    loop {
        // 1. Block until first request arrives (or channel closes)
        let first = rx.recv().await;
        let first = match first {
            Some(req) => req,
            None => {
                // Channel closed — flush remaining and exit
                if !buffer.is_empty() {
                    flush(&inner, &mut buffer, max_batch_states).await;
                }
                return;
            }
        };
        buffer.push(first);

        // 2. Drain any immediately available requests
        drain_available(&mut rx, &mut buffer);

        // 3. Check threshold or linger for more
        loop {
            let total_states: usize = buffer.iter().map(|r| r.states.len()).sum();
            if total_states >= max_batch_states {
                break;
            }

            match tokio::time::timeout(linger, rx.recv()).await {
                Ok(Some(req)) => {
                    buffer.push(req);
                    drain_available(&mut rx, &mut buffer);
                    // Loop back to check threshold
                }
                Ok(None) => {
                    // Channel closed — flush remaining and exit
                    flush(&inner, &mut buffer, max_batch_states).await;
                    return;
                }
                Err(_timeout) => {
                    // Linger expired — flush what we have
                    break;
                }
            }
        }

        // 4. Flush the batch
        flush(&inner, &mut buffer, max_batch_states).await;
    }
}

/// Drain all immediately available requests from the channel.
fn drain_available(
    rx: &mut mpsc::Receiver<EncodeCoalesceRequest>,
    buffer: &mut Vec<EncodeCoalesceRequest>,
) {
    while let Ok(req) = rx.try_recv() {
        buffer.push(req);
    }
}

/// Merge buffered requests, call inner encoder, distribute result slices.
async fn flush(inner: &Arc<dyn BatchEncoder>, buffer: &mut Vec<EncodeCoalesceRequest>, max_batch_states: usize) {
    if buffer.is_empty() {
        return;
    }

    let requests = std::mem::take(buffer);

    // Merge all states into one flat vec, tracking per-request offsets
    let mut all_states: Vec<String> = Vec::new();
    let mut offsets: Vec<(usize, usize)> = Vec::new(); // (start, len)

    for req in &requests {
        let start = all_states.len();
        all_states.extend(req.states.iter().cloned());
        offsets.push((start, req.states.len()));
    }

    // Encode in sub-batches to avoid OOM on quantized servers
    let result = encode_in_chunks(inner, &all_states, max_batch_states).await;

    // Distribute results to waiters
    match result {
        Ok(all_embeddings) => {
            for (i, req) in requests.into_iter().enumerate() {
                let (start, len) = offsets[i];
                let slice = all_embeddings[start..start + len].to_vec();
                let _ = req.tx.send(Ok(slice));
            }
        }
        Err(e) => {
            let err_str = e.to_string();
            for req in requests {
                let _ = req.tx.send(Err(err_str.clone()));
            }
        }
    }
}

/// Encode states in chunks of at most `chunk_size`, concatenating results.
async fn encode_in_chunks(
    inner: &Arc<dyn BatchEncoder>,
    states: &[String],
    chunk_size: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    if states.len() <= chunk_size {
        return inner.encode_batch(states).await;
    }

    let mut all_embeddings = Vec::with_capacity(states.len());
    for chunk in states.chunks(chunk_size) {
        let embeddings = inner.encode_batch(chunk).await?;
        all_embeddings.extend(embeddings);
    }
    Ok(all_embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// Mock encoder: produces a simple hash-based embedding.
    struct MockEncoder {
        dim: usize,
    }

    #[async_trait]
    impl BatchEncoder for MockEncoder {
        async fn encode_batch(&self, states: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(states
                .iter()
                .map(|s| {
                    let mut emb = vec![0.0_f32; self.dim];
                    for (i, byte) in s.bytes().enumerate() {
                        emb[i % self.dim] += byte as f32 / 255.0;
                    }
                    emb
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn test_single_request_passthrough() {
        let encoder = Arc::new(MockEncoder { dim: 8 });
        let batcher = GlobalEncodeBatcher::new(encoder, 64, Duration::from_millis(5));

        let result = tokio::task::spawn_blocking(move || {
            batcher.encode_batch_blocking(&["⊢ True"])
        })
        .await
        .unwrap()
        .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 8);
    }

    #[tokio::test]
    async fn test_batch_request_passthrough() {
        let encoder = Arc::new(MockEncoder { dim: 8 });
        let batcher = GlobalEncodeBatcher::new(encoder, 64, Duration::from_millis(5));

        let result = tokio::task::spawn_blocking(move || {
            batcher.encode_batch_blocking(&["⊢ A", "⊢ B", "⊢ C"])
        })
        .await
        .unwrap()
        .unwrap();

        assert_eq!(result.len(), 3);
        for emb in &result {
            assert_eq!(emb.len(), 8);
        }
    }

    #[tokio::test]
    async fn test_empty_input() {
        let encoder = Arc::new(MockEncoder { dim: 8 });
        let batcher = GlobalEncodeBatcher::new(encoder, 64, Duration::from_millis(5));

        let result = tokio::task::spawn_blocking(move || {
            batcher.encode_batch_blocking(&[])
        })
        .await
        .unwrap()
        .unwrap();

        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_coalescing_concurrent() {
        struct TrackingEncoder {
            inner: MockEncoder,
            batch_call_count: Arc<AtomicUsize>,
            batch_sizes: Arc<Mutex<Vec<usize>>>,
        }

        #[async_trait]
        impl BatchEncoder for TrackingEncoder {
            async fn encode_batch(&self, states: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
                self.batch_call_count.fetch_add(1, Ordering::SeqCst);
                self.batch_sizes.lock().unwrap().push(states.len());
                // Add small delay to ensure coalescing works
                tokio::time::sleep(Duration::from_millis(10)).await;
                self.inner.encode_batch(states).await
            }
        }

        let batch_call_count = Arc::new(AtomicUsize::new(0));
        let batch_sizes = Arc::new(Mutex::new(Vec::new()));

        let tracking = TrackingEncoder {
            inner: MockEncoder { dim: 8 },
            batch_call_count: batch_call_count.clone(),
            batch_sizes: batch_sizes.clone(),
        };

        // Use a longer linger to ensure coalescing
        let batcher = Arc::new(GlobalEncodeBatcher::new(
            Arc::new(tracking),
            64,
            Duration::from_millis(50),
        ));

        // Spawn 4 concurrent single-state requests
        let mut handles = Vec::new();
        for i in 0..4 {
            let b = batcher.clone();
            let state = format!("⊢ state_{i}");
            handles.push(tokio::task::spawn_blocking(move || {
                b.encode_batch_blocking(&[&state])
            }));
        }

        for h in handles {
            let result = h.await.unwrap();
            assert!(result.is_ok());
        }

        // Should have fewer batch calls than individual requests
        let calls = batch_call_count.load(Ordering::SeqCst);
        assert!(
            calls <= 2,
            "Expected coalescing to merge requests (got {calls} batch calls for 4 requests)"
        );

        let sizes = batch_sizes.lock().unwrap();
        let total_states: usize = sizes.iter().sum();
        assert_eq!(total_states, 4, "All 4 states should be processed");
    }

    #[tokio::test]
    async fn test_error_broadcast() {
        struct ErrorEncoder;

        #[async_trait]
        impl BatchEncoder for ErrorEncoder {
            async fn encode_batch(&self, _states: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
                Err(anyhow::anyhow!("server down"))
            }
        }

        let batcher = GlobalEncodeBatcher::new(
            Arc::new(ErrorEncoder),
            64,
            Duration::from_millis(5),
        );

        let result = tokio::task::spawn_blocking(move || {
            batcher.encode_batch_blocking(&["⊢ True"])
        })
        .await
        .unwrap();

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("server down"));
    }

    #[tokio::test]
    async fn test_shutdown_flush() {
        let encoder = Arc::new(MockEncoder { dim: 8 });
        let batcher = GlobalEncodeBatcher::new(encoder, 64, Duration::from_millis(100));

        // Submit request and immediately drop the batcher
        let result = tokio::task::spawn_blocking(move || {
            let res = batcher.encode_batch_blocking(&["⊢ True"]);
            drop(batcher);
            res
        })
        .await
        .unwrap();

        assert!(result.is_ok());
    }
}
