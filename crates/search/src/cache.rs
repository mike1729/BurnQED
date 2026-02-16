//! LRU cache wrapper for [`PolicyProvider`] that memoizes generation results.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use lru::LruCache;
use policy::GeneratedTactic;

use crate::engine::{PolicyProvider, SearchError};

/// Wraps a [`PolicyProvider`] with an LRU cache keyed by proof state text.
///
/// Eliminates redundant LLM calls when different tactic paths produce the
/// same sub-goal (common after `simp` normalizes expressions).
///
/// The batch implementation groups cache misses into a single call to the
/// inner provider's `generate_candidates_batch`, preserving parallel HTTP.
pub struct CachedPolicy<P> {
    inner: P,
    cache: Mutex<LruCache<String, Vec<GeneratedTactic>>>,
    hits: AtomicU32,
    misses: AtomicU32,
}

impl<P> CachedPolicy<P> {
    /// Create a new cached policy wrapper with the given LRU capacity.
    pub fn new(inner: P, capacity: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).expect("cache capacity must be > 0"),
            )),
            hits: AtomicU32::new(0),
            misses: AtomicU32::new(0),
        }
    }

    /// Get a reference to the inner policy.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Return (hits, misses) counters since last reset.
    pub fn counters(&self) -> (u32, u32) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    /// Reset hit/miss counters to zero.
    pub fn reset_counters(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

#[async_trait]
impl<P: PolicyProvider> PolicyProvider for CachedPolicy<P> {
    async fn generate_candidates(
        &self,
        proof_state: &str,
        n: usize,
    ) -> Result<Vec<GeneratedTactic>, SearchError> {
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(proof_state) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(cached.clone());
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let candidates = self.inner.generate_candidates(proof_state, n).await?;
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(proof_state.to_string(), candidates.clone());
        }
        Ok(candidates)
    }

    async fn generate_candidates_batch(
        &self,
        states: &[String],
        n: usize,
    ) -> Result<Vec<Vec<GeneratedTactic>>, SearchError> {
        let mut results: Vec<Option<Vec<GeneratedTactic>>> = vec![None; states.len()];
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_states: Vec<String> = Vec::new();

        // Phase 1: Check cache for all states
        {
            let mut cache = self.cache.lock().unwrap();
            for (i, s) in states.iter().enumerate() {
                if let Some(cached) = cache.get(s.as_str()) {
                    results[i] = Some(cached.clone());
                    self.hits.fetch_add(1, Ordering::Relaxed);
                } else {
                    miss_indices.push(i);
                    miss_states.push(s.clone());
                    self.misses.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        // Phase 2: Batch-delegate misses (preserves parallel HTTP via inner)
        if !miss_states.is_empty() {
            let batch_results = self
                .inner
                .generate_candidates_batch(&miss_states, n)
                .await?;
            let mut cache = self.cache.lock().unwrap();
            for (j, &idx) in miss_indices.iter().enumerate() {
                cache.put(miss_states[j].clone(), batch_results[j].clone());
                results[idx] = Some(batch_results[j].clone());
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    fn cache_stats(&self) -> Option<(u32, u32)> {
        Some(self.counters())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mocks::{make_tactic, MockPolicy};

    #[tokio::test]
    async fn test_cached_policy_cache_hit() {
        let mut mock = MockPolicy::new();
        mock.add_response("⊢ True", vec![make_tactic("trivial", -0.1)]);

        let cached = CachedPolicy::new(mock, 10);

        // First call: cache miss → delegates to inner
        let r1 = cached.generate_candidates("⊢ True", 8).await.unwrap();
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0].text, "trivial");

        // Second call: cache hit → same result
        let r2 = cached.generate_candidates("⊢ True", 8).await.unwrap();
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].text, "trivial");

        // Verify counters
        let (hits, misses) = cached.counters();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[tokio::test]
    async fn test_cached_policy_cache_miss() {
        let mock = MockPolicy::new();
        let cached = CachedPolicy::new(mock, 10);

        let r = cached.generate_candidates("⊢ unknown", 8).await.unwrap();
        assert!(r.is_empty());

        let (hits, misses) = cached.counters();
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);
    }

    #[tokio::test]
    async fn test_cached_policy_lru_eviction() {
        let policy = MockPolicy::with_default(vec![make_tactic("default", -1.0)]);
        let cached = CachedPolicy::new(policy, 2); // capacity 2

        // Fill cache with 2 entries
        cached.generate_candidates("state_a", 8).await.unwrap();
        cached.generate_candidates("state_b", 8).await.unwrap();

        // Add a third → evicts "state_a"
        cached.generate_candidates("state_c", 8).await.unwrap();

        // Verify cache state
        let cache = cached.cache.lock().unwrap();
        assert!(cache.peek("state_a").is_none());
        assert!(cache.peek("state_b").is_some());
        assert!(cache.peek("state_c").is_some());
    }

    #[tokio::test]
    async fn test_cached_policy_batch() {
        let mut mock = MockPolicy::new();
        mock.add_response("⊢ A", vec![make_tactic("intro", -0.2)]);
        mock.add_response("⊢ B", vec![make_tactic("exact h", -0.3)]);

        let cached = CachedPolicy::new(mock, 10);
        let results = cached
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
    async fn test_cached_policy_batch_mixed_hits_misses() {
        let mut mock = MockPolicy::new();
        mock.add_response("⊢ A", vec![make_tactic("intro", -0.2)]);
        mock.add_response("⊢ B", vec![make_tactic("exact h", -0.3)]);
        mock.add_response("⊢ C", vec![make_tactic("simp", -0.4)]);

        let cached = CachedPolicy::new(mock, 10);

        // Pre-warm: "⊢ A" in cache
        cached.generate_candidates("⊢ A", 8).await.unwrap();
        cached.reset_counters();

        // Batch with 1 hit ("⊢ A") and 2 misses ("⊢ B", "⊢ C")
        let results = cached
            .generate_candidates_batch(
                &["⊢ A".to_string(), "⊢ B".to_string(), "⊢ C".to_string()],
                8,
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0][0].text, "intro");
        assert_eq!(results[1][0].text, "exact h");
        assert_eq!(results[2][0].text, "simp");

        let (hits, misses) = cached.counters();
        assert_eq!(hits, 1);
        assert_eq!(misses, 2);
    }

    #[tokio::test]
    async fn test_cached_policy_batch_all_hits() {
        let mut mock = MockPolicy::new();
        mock.add_response("⊢ A", vec![make_tactic("intro", -0.2)]);
        mock.add_response("⊢ B", vec![make_tactic("exact h", -0.3)]);

        let cached = CachedPolicy::new(mock, 10);

        // Pre-warm both
        cached.generate_candidates("⊢ A", 8).await.unwrap();
        cached.generate_candidates("⊢ B", 8).await.unwrap();
        cached.reset_counters();

        let results = cached
            .generate_candidates_batch(
                &["⊢ A".to_string(), "⊢ B".to_string()],
                8,
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0].text, "intro");
        assert_eq!(results[1][0].text, "exact h");

        let (hits, misses) = cached.counters();
        assert_eq!(hits, 2);
        assert_eq!(misses, 0);
    }

    #[tokio::test]
    async fn test_cached_policy_batch_delegates_to_inner_batch() {
        // Verify that cache misses are grouped into a single inner batch call,
        // not N individual calls. We use BatchTrackingPolicy to verify.
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;

        struct BatchTrackingPolicy {
            inner: MockPolicy,
            batch_call_count: Arc<AtomicUsize>,
            batch_sizes: Arc<Mutex<Vec<usize>>>,
        }

        #[async_trait]
        impl PolicyProvider for BatchTrackingPolicy {
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
                self.batch_call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.batch_sizes.lock().unwrap().push(states.len());
                // Delegate to sequential default for mock
                let mut results = Vec::with_capacity(states.len());
                for s in states {
                    results.push(self.inner.generate_candidates(s, n).await?);
                }
                Ok(results)
            }
        }

        let mut mock = MockPolicy::new();
        mock.add_response("⊢ A", vec![make_tactic("intro", -0.2)]);
        mock.add_response("⊢ B", vec![make_tactic("exact h", -0.3)]);
        mock.add_response("⊢ C", vec![make_tactic("simp", -0.4)]);

        let batch_call_count = Arc::new(AtomicUsize::new(0));
        let batch_sizes = Arc::new(Mutex::new(Vec::new()));

        let tracking = BatchTrackingPolicy {
            inner: mock,
            batch_call_count: batch_call_count.clone(),
            batch_sizes: batch_sizes.clone(),
        };

        let cached = CachedPolicy::new(tracking, 10);

        // Pre-warm "⊢ A" so it's a cache hit
        cached.generate_candidates("⊢ A", 8).await.unwrap();

        // Reset batch tracking
        batch_call_count.store(0, std::sync::atomic::Ordering::Relaxed);
        batch_sizes.lock().unwrap().clear();

        // Batch: "⊢ A" (hit), "⊢ B" (miss), "⊢ C" (miss)
        let results = cached
            .generate_candidates_batch(
                &["⊢ A".to_string(), "⊢ B".to_string(), "⊢ C".to_string()],
                8,
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 3);

        // Should have made exactly 1 batch call to inner with 2 misses
        let calls = batch_call_count.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(calls, 1, "Should batch misses into a single call, got {calls}");

        let sizes = batch_sizes.lock().unwrap();
        assert_eq!(sizes[0], 2, "Batch should contain 2 misses, got {}", sizes[0]);
    }

    #[tokio::test]
    async fn test_cache_stats_via_trait() {
        let policy = MockPolicy::with_default(vec![make_tactic("default", -1.0)]);
        let cached = CachedPolicy::new(policy, 10);

        cached.generate_candidates("a", 8).await.unwrap(); // miss
        cached.generate_candidates("a", 8).await.unwrap(); // hit
        cached.generate_candidates("b", 8).await.unwrap(); // miss

        let stats = cached.cache_stats();
        assert_eq!(stats, Some((1, 2)));
    }

    #[tokio::test]
    async fn test_reset_counters() {
        let policy = MockPolicy::with_default(vec![make_tactic("default", -1.0)]);
        let cached = CachedPolicy::new(policy, 10);

        cached.generate_candidates("a", 8).await.unwrap();
        cached.generate_candidates("a", 8).await.unwrap();
        assert_eq!(cached.counters(), (1, 1));

        cached.reset_counters();
        assert_eq!(cached.counters(), (0, 0));
    }
}
