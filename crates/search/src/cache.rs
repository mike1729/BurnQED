//! LRU cache wrapper for [`PolicyProvider`] that memoizes generation results.

use std::num::NonZeroUsize;
use std::sync::Mutex;

use async_trait::async_trait;
use lru::LruCache;
use policy::GeneratedTactic;

use crate::engine::{PolicyProvider, SearchError};

/// Wraps a [`PolicyProvider`] with an LRU cache keyed by proof state text.
///
/// Eliminates redundant LLM calls when different tactic paths produce the
/// same sub-goal (common after `simp` normalizes expressions).
pub struct CachedPolicy<P> {
    inner: P,
    cache: Mutex<LruCache<String, Vec<GeneratedTactic>>>,
}

impl<P> CachedPolicy<P> {
    /// Create a new cached policy wrapper with the given LRU capacity.
    pub fn new(inner: P, capacity: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).expect("cache capacity must be > 0"),
            )),
        }
    }

    /// Get a reference to the inner policy.
    pub fn inner(&self) -> &P {
        &self.inner
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
            if let Some(hits) = cache.get(proof_state) {
                return Ok(hits.clone());
            }
        }
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
        // Check cache for each state individually, delegate misses to inner
        let mut results = Vec::with_capacity(states.len());
        for s in states {
            results.push(self.generate_candidates(s, n).await?);
        }
        Ok(results)
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
    }

    #[tokio::test]
    async fn test_cached_policy_cache_miss() {
        let mock = MockPolicy::new();
        let cached = CachedPolicy::new(mock, 10);

        let r = cached.generate_candidates("⊢ unknown", 8).await.unwrap();
        assert!(r.is_empty());
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
}
