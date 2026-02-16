//! Global request coalescing for cross-search batching.
//!
//! With multiple concurrent search tasks, each submits small HTTP batches
//! independently. `GlobalBatcher` collects requests from all tasks into a
//! shared channel and flushes them as one large batch — a Nagle-like pattern
//! that maximizes GPU utilization via SGLang's RadixAttention.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use policy::GeneratedTactic;
use tokio::sync::{mpsc, oneshot};

use crate::engine::{PolicyProvider, SearchError};

/// A coalescing request: one or more states to generate candidates for.
struct CoalesceRequest {
    states: Vec<String>,
    n: usize,
    tx: oneshot::Sender<Result<Vec<Vec<GeneratedTactic>>, String>>,
}

/// Middleware that coalesces generation requests from multiple concurrent
/// search tasks into single GPU-efficient batches.
///
/// Sits between [`CachedPolicy`] and the underlying [`PolicyProvider`]
/// (typically `InferencePolicyProvider`). Spawns a background tokio task
/// that accumulates requests and flushes when either `max_batch_states`
/// is reached or `linger` duration elapses.
pub struct GlobalBatcher {
    sender: mpsc::Sender<CoalesceRequest>,
}

impl GlobalBatcher {
    /// Create a new `GlobalBatcher` wrapping the given inner policy provider.
    ///
    /// - `max_batch_states`: flush when accumulated states reach this count
    /// - `linger`: maximum time to wait for additional requests before flushing
    pub fn new(
        inner: Arc<dyn PolicyProvider>,
        max_batch_states: usize,
        linger: Duration,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<CoalesceRequest>(max_batch_states * 4);
        tokio::spawn(coalesce_loop(rx, inner, max_batch_states, linger));
        GlobalBatcher { sender: tx }
    }
}

/// Background task that collects requests and flushes them in batches.
async fn coalesce_loop(
    mut rx: mpsc::Receiver<CoalesceRequest>,
    inner: Arc<dyn PolicyProvider>,
    max_batch_states: usize,
    linger: Duration,
) {
    let mut buffer: Vec<CoalesceRequest> = Vec::new();

    loop {
        // 1. Block until first request arrives (or channel closes)
        let first = rx.recv().await;
        let first = match first {
            Some(req) => req,
            None => {
                // Channel closed — flush remaining and exit
                if !buffer.is_empty() {
                    flush(&inner, &mut buffer).await;
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
                    flush(&inner, &mut buffer).await;
                    return;
                }
                Err(_timeout) => {
                    // Linger expired — flush what we have
                    break;
                }
            }
        }

        // 4. Flush the batch
        flush(&inner, &mut buffer).await;
    }
}

/// Drain all immediately available requests from the channel.
fn drain_available(rx: &mut mpsc::Receiver<CoalesceRequest>, buffer: &mut Vec<CoalesceRequest>) {
    while let Ok(req) = rx.try_recv() {
        buffer.push(req);
    }
}

/// Group buffered requests by `n`, call inner provider, distribute results.
async fn flush(inner: &Arc<dyn PolicyProvider>, buffer: &mut Vec<CoalesceRequest>) {
    if buffer.is_empty() {
        return;
    }

    let requests = std::mem::take(buffer);

    // Group by n value. Almost always uniform (one group), but handle edge case.
    let mut groups: Vec<(usize, Vec<CoalesceRequest>)> = Vec::new();
    for req in requests {
        if let Some(group) = groups.iter_mut().find(|(n, _)| *n == req.n) {
            group.1.push(req);
        } else {
            groups.push((req.n, vec![req]));
        }
    }

    for (n, group) in groups {
        // Merge all states into one flat vec, tracking per-request offsets
        let mut all_states: Vec<String> = Vec::new();
        let mut offsets: Vec<(usize, usize)> = Vec::new(); // (start, len)

        for req in &group {
            let start = all_states.len();
            all_states.extend(req.states.iter().cloned());
            offsets.push((start, req.states.len()));
        }

        // Call inner provider with merged batch
        let result = inner.generate_candidates_batch(&all_states, n).await;

        // Distribute results to waiters
        match result {
            Ok(all_results) => {
                for (i, req) in group.into_iter().enumerate() {
                    let (start, len) = offsets[i];
                    let slice = all_results[start..start + len].to_vec();
                    let _ = req.tx.send(Ok(slice));
                }
            }
            Err(e) => {
                let err_str = e.to_string();
                for req in group {
                    let _ = req.tx.send(Err(err_str.clone()));
                }
            }
        }
    }
}

#[async_trait]
impl PolicyProvider for GlobalBatcher {
    async fn generate_candidates(
        &self,
        proof_state: &str,
        n: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError> {
        let (tx, rx) = oneshot::channel();
        let req = CoalesceRequest {
            states: vec![proof_state.to_string()],
            n,
            tx,
        };
        self.sender
            .send(req)
            .await
            .map_err(|_| SearchError::Policy(anyhow::anyhow!("GlobalBatcher background task gone")))?;

        let result = rx
            .await
            .map_err(|_| SearchError::Policy(anyhow::anyhow!("GlobalBatcher response dropped")))?;

        match result {
            Ok(mut vecs) => {
                if vecs.len() == 1 {
                    Ok(vecs.remove(0))
                } else {
                    Ok(vec![])
                }
            }
            Err(e) => Err(SearchError::Policy(anyhow::anyhow!(e))),
        }
    }

    async fn generate_candidates_batch(
        &self,
        states: &[String],
        n: usize,
    ) -> Result<Vec<Vec<GeneratedTactic>>, SearchError> {
        if states.is_empty() {
            return Ok(vec![]);
        }

        let (tx, rx) = oneshot::channel();
        let req = CoalesceRequest {
            states: states.to_vec(),
            n,
            tx,
        };
        self.sender
            .send(req)
            .await
            .map_err(|_| SearchError::Policy(anyhow::anyhow!("GlobalBatcher background task gone")))?;

        let result = rx
            .await
            .map_err(|_| SearchError::Policy(anyhow::anyhow!("GlobalBatcher response dropped")))?;

        result.map_err(|e| SearchError::Policy(anyhow::anyhow!(e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mocks::{make_tactic, MockPolicy};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// Helper: wrap a MockPolicy into a GlobalBatcher for testing.
    fn make_batcher(mock: MockPolicy, linger_ms: u64) -> GlobalBatcher {
        GlobalBatcher::new(Arc::new(mock), 64, Duration::from_millis(linger_ms))
    }

    #[tokio::test]
    async fn test_single_request_passthrough() {
        let mut mock = MockPolicy::new();
        mock.add_response("⊢ True", vec![make_tactic("trivial", -0.1)]);

        let batcher = make_batcher(mock, 5);
        let result = batcher.generate_candidates("⊢ True", 8).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "trivial");
    }

    #[tokio::test]
    async fn test_batch_request_passthrough() {
        let mut mock = MockPolicy::new();
        mock.add_response("⊢ A", vec![make_tactic("intro", -0.2)]);
        mock.add_response("⊢ B", vec![make_tactic("exact h", -0.3)]);

        let batcher = make_batcher(mock, 5);
        let results = batcher
            .generate_candidates_batch(
                &["⊢ A".to_string(), "⊢ B".to_string()],
                8,
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0].text, "intro");
        assert_eq!(results[1][0].text, "exact h");
    }

    #[tokio::test]
    async fn test_empty_input() {
        let mock = MockPolicy::new();
        let batcher = make_batcher(mock, 5);
        let results = batcher
            .generate_candidates_batch(&[], 8)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_coalescing_concurrent() {
        // Verify that concurrent submissions get merged into fewer inner batch calls.
        struct TrackingPolicy {
            inner: MockPolicy,
            batch_call_count: Arc<AtomicUsize>,
            batch_sizes: Arc<Mutex<Vec<usize>>>,
        }

        #[async_trait]
        impl PolicyProvider for TrackingPolicy {
            async fn generate_candidates(
                &self,
                proof_state: &str,
                n: usize,
            ) -> Result<Vec<GeneratedTactic>, SearchError> {
                self.inner.generate_candidates(proof_state, n).await
            }

            async fn generate_candidates_batch(
                &self,
                states: &[String],
                n: usize,
            ) -> Result<Vec<Vec<GeneratedTactic>>, SearchError> {
                self.batch_call_count.fetch_add(1, Ordering::SeqCst);
                self.batch_sizes.lock().unwrap().push(states.len());
                // Add small delay to ensure coalescing works
                tokio::time::sleep(Duration::from_millis(10)).await;
                // Delegate to sequential mock
                let mut results = Vec::with_capacity(states.len());
                for s in states {
                    results.push(self.inner.generate_candidates(s, n).await?);
                }
                Ok(results)
            }
        }

        let mock = MockPolicy::with_default(vec![make_tactic("default", -1.0)]);
        let batch_call_count = Arc::new(AtomicUsize::new(0));
        let batch_sizes = Arc::new(Mutex::new(Vec::new()));

        let tracking = TrackingPolicy {
            inner: mock,
            batch_call_count: batch_call_count.clone(),
            batch_sizes: batch_sizes.clone(),
        };

        // Use a longer linger to ensure coalescing
        let batcher = GlobalBatcher::new(Arc::new(tracking), 64, Duration::from_millis(50));
        let batcher = Arc::new(batcher);

        // Spawn 4 concurrent single-state requests
        let mut handles = Vec::new();
        for i in 0..4 {
            let b = batcher.clone();
            let state = format!("⊢ state_{i}");
            handles.push(tokio::spawn(async move {
                b.generate_candidates(&state, 8).await
            }));
        }

        // Wait for all to complete
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
        // Inner provider always errors
        struct ErrorPolicy;

        #[async_trait]
        impl PolicyProvider for ErrorPolicy {
            async fn generate_candidates(
                &self,
                _proof_state: &str,
                _n: usize,
            ) -> Result<Vec<GeneratedTactic>, SearchError> {
                Err(SearchError::Policy(anyhow::anyhow!("server down")))
            }

            async fn generate_candidates_batch(
                &self,
                _states: &[String],
                _n: usize,
            ) -> Result<Vec<Vec<GeneratedTactic>>, SearchError> {
                Err(SearchError::Policy(anyhow::anyhow!("server down")))
            }
        }

        let batcher = GlobalBatcher::new(Arc::new(ErrorPolicy), 64, Duration::from_millis(5));

        let result = batcher.generate_candidates("⊢ True", 8).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("server down"));
    }

    #[tokio::test]
    async fn test_shutdown_flush() {
        let mock = MockPolicy::with_default(vec![make_tactic("ok", -1.0)]);
        let batcher = make_batcher(mock, 100); // long linger

        // Submit request and immediately drop the batcher
        let result = batcher.generate_candidates("⊢ True", 8).await;
        assert!(result.is_ok());
        // Dropping batcher closes the channel, background task should exit cleanly
        drop(batcher);
    }

    #[tokio::test]
    async fn test_different_n_grouped() {
        // Requests with different n should be grouped separately
        struct NTrackingPolicy {
            inner: MockPolicy,
            calls: Arc<Mutex<Vec<(usize, usize)>>>, // (num_states, n)
        }

        #[async_trait]
        impl PolicyProvider for NTrackingPolicy {
            async fn generate_candidates(
                &self,
                proof_state: &str,
                n: usize,
            ) -> Result<Vec<GeneratedTactic>, SearchError> {
                self.inner.generate_candidates(proof_state, n).await
            }

            async fn generate_candidates_batch(
                &self,
                states: &[String],
                n: usize,
            ) -> Result<Vec<Vec<GeneratedTactic>>, SearchError> {
                self.calls.lock().unwrap().push((states.len(), n));
                let mut results = Vec::with_capacity(states.len());
                for s in states {
                    results.push(self.inner.generate_candidates(s, n).await?);
                }
                Ok(results)
            }
        }

        let mock = MockPolicy::with_default(vec![make_tactic("ok", -1.0)]);
        let calls = Arc::new(Mutex::new(Vec::new()));

        let tracking = NTrackingPolicy {
            inner: mock,
            calls: calls.clone(),
        };

        // Use long linger so both requests coalesce
        let batcher = GlobalBatcher::new(Arc::new(tracking), 64, Duration::from_millis(50));
        let batcher = Arc::new(batcher);

        // Submit two requests with different n values concurrently
        let b1 = batcher.clone();
        let b2 = batcher.clone();
        let h1 = tokio::spawn(async move {
            b1.generate_candidates("⊢ A", 4).await
        });
        let h2 = tokio::spawn(async move {
            b2.generate_candidates("⊢ B", 8).await
        });

        let r1 = h1.await.unwrap();
        let r2 = h2.await.unwrap();
        assert!(r1.is_ok());
        assert!(r2.is_ok());

        // Verify the calls were made with correct n values
        let recorded = calls.lock().unwrap();
        let n_values: Vec<usize> = recorded.iter().map(|(_, n)| *n).collect();
        assert!(n_values.contains(&4), "Should have a call with n=4");
        assert!(n_values.contains(&8), "Should have a call with n=8");
    }
}
