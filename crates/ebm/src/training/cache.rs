//! Precomputed embedding cache for EBM training.
//!
//! Since the LLM encoder is frozen, `state_pp` → embedding is deterministic.
//! This module precomputes all unique embeddings once and stores them in a
//! `HashMap<String, Vec<f32>>` for O(1) lookups during training. Optionally
//! persists to Parquet for reuse across runs.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::*;
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

use super::data::ContrastiveSampler;

/// Format ETA from wall-clock elapsed time and progress.
fn format_eta(start: Instant, done: usize, total: usize) -> String {
    if done == 0 {
        return "?".to_string();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let remaining = elapsed * (total - done) as f64 / done as f64;
    if remaining < 60.0 {
        format!("{:.0}s", remaining)
    } else if remaining < 3600.0 {
        format!("{:.0}m", remaining / 60.0)
    } else {
        format!("{:.1}h", remaining / 3600.0)
    }
}

/// Precomputed embedding cache backed by a `HashMap`.
///
/// Maps proof state strings to their embedding vectors. Used to avoid
/// redundant LLM encoder calls during EBM training.
pub struct EmbeddingCache {
    embeddings: HashMap<String, Vec<f32>>,
    dim: usize,
}

impl EmbeddingCache {
    /// Create an empty cache with the given embedding dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            embeddings: HashMap::new(),
            dim,
        }
    }

    /// Precompute embeddings for all unique states in the sampler.
    ///
    /// Calls `encode_fn` once per unique `state_pp`, with an indicatif progress bar.
    /// States that fail to encode are logged and skipped.
    pub fn precompute(
        sampler: &ContrastiveSampler,
        encode_fn: &dyn Fn(&str) -> anyhow::Result<Vec<f32>>,
        dim: usize,
    ) -> Self {
        let unique_states = sampler.unique_states();
        let total = unique_states.len();

        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({msg}) Encoding embeddings")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("=> "),
        );

        let mut embeddings = HashMap::with_capacity(total);
        let mut errors = 0usize;
        let start = Instant::now();

        for (i, state) in unique_states.iter().enumerate() {
            match encode_fn(state) {
                Ok(emb) => {
                    debug_assert_eq!(emb.len(), dim, "Embedding dim mismatch: expected {dim}, got {}", emb.len());
                    embeddings.insert(state.to_string(), emb);
                }
                Err(e) => {
                    errors += 1;
                    tracing::debug!(state = %state, error = %e, "Failed to encode state");
                }
            }
            pb.inc(1);
            if (i + 1) % 100 == 0 {
                pb.set_message(format_eta(start, i + 1, total));
            }
        }

        pb.finish_with_message("done");

        tracing::info!(
            total_states = total,
            cached = embeddings.len(),
            errors = errors,
            "Embedding precomputation complete"
        );

        Self { embeddings, dim }
    }

    /// Precompute embeddings concurrently for a set of states.
    ///
    /// Fires up to `concurrency` encode requests in parallel via
    /// `buffer_unordered`, which lets SGLang's continuous batching scheduler
    /// batch them on the GPU. States already present in `self` are skipped.
    ///
    /// Returns `(newly_encoded, errors)`.
    pub async fn precompute_concurrent<F, Fut>(
        &mut self,
        states: &HashSet<&str>,
        encode_fn: F,
        concurrency: usize,
        dim: usize,
        checkpoint_path: Option<&Path>,
        checkpoint_interval: usize,
    ) -> (usize, usize)
    where
        F: Fn(String) -> Fut,
        Fut: Future<Output = anyhow::Result<Vec<f32>>>,
    {
        // Filter to states not already cached
        let missing: Vec<String> = states
            .iter()
            .filter(|s| !self.embeddings.contains_key(**s))
            .map(|s| s.to_string())
            .collect();

        if missing.is_empty() {
            tracing::info!("All {} states already cached — skipping precompute", states.len());
            return (0, 0);
        }

        let total = missing.len();
        tracing::info!(
            total_states = states.len(),
            cached = states.len() - total,
            to_encode = total,
            concurrency,
            "Starting concurrent embedding precomputation"
        );

        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({msg}) Encoding embeddings")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("=> "),
        );

        let mut encoded = 0usize;
        let mut errors = 0usize;
        let start = Instant::now();
        let mut done = 0usize;
        let mut last_checkpoint = 0usize;

        // Stream of (state, result) pairs with bounded concurrency
        let mut result_stream = stream::iter(missing.into_iter().map(|state| {
            let fut = encode_fn(state.clone());
            async move { (state, fut.await) }
        }))
        .buffer_unordered(concurrency);

        while let Some((state, result)) = result_stream.next().await {
            match result {
                Ok(emb) => {
                    debug_assert_eq!(
                        emb.len(),
                        dim,
                        "Embedding dim mismatch: expected {dim}, got {}",
                        emb.len()
                    );
                    self.embeddings.insert(state, emb);
                    encoded += 1;
                }
                Err(e) => {
                    errors += 1;
                    tracing::debug!(state = %state, error = %e, "Failed to encode state");
                }
            }
            done += 1;
            pb.inc(1);
            if done % 100 == 0 {
                pb.set_message(format_eta(start, done, total));
            }

            // Periodic checkpoint save
            if let Some(cp_path) = checkpoint_path {
                if encoded - last_checkpoint >= checkpoint_interval {
                    if let Err(e) = self.save(cp_path) {
                        tracing::warn!(error = %e, "Failed to save checkpoint");
                    } else {
                        tracing::info!(
                            encoded,
                            total_cached = self.embeddings.len(),
                            path = %cp_path.display(),
                            "Checkpoint saved"
                        );
                    }
                    last_checkpoint = encoded;
                }
            }
        }

        // Final checkpoint save
        if let Some(cp_path) = checkpoint_path {
            if encoded > last_checkpoint {
                if let Err(e) = self.save(cp_path) {
                    tracing::warn!(error = %e, "Failed to save final checkpoint");
                } else {
                    tracing::info!(
                        encoded,
                        total_cached = self.embeddings.len(),
                        path = %cp_path.display(),
                        "Final checkpoint saved"
                    );
                }
            }
        }

        pb.finish_with_message("done");

        if errors > 0 {
            tracing::warn!(
                errors,
                "Some states failed to encode (use RUST_LOG=debug for details)"
            );
        }

        tracing::info!(
            total_states = states.len(),
            newly_encoded = encoded,
            errors,
            total_cached = self.embeddings.len(),
            "Concurrent embedding precomputation complete"
        );

        (encoded, errors)
    }

    /// Precompute embeddings in batches for GPU-optimal throughput.
    ///
    /// Chunks missing states into groups of `batch_size`, sends each chunk to
    /// `encode_batch_fn` (which should issue a single batched HTTP request),
    /// and processes up to `concurrency` chunks concurrently via
    /// `buffer_unordered`. Progress bar increments per state (not per batch).
    ///
    /// If `checkpoint_path` is provided, saves progress every `checkpoint_interval`
    /// states so encoding can be resumed after crashes/restarts.
    ///
    /// Returns `(newly_encoded, errors)`.
    pub async fn precompute_batched<F, Fut>(
        &mut self,
        states: &HashSet<&str>,
        encode_batch_fn: F,
        batch_size: usize,
        concurrency: usize,
        dim: usize,
    ) -> (usize, usize)
    where
        F: Fn(Vec<String>) -> Fut,
        Fut: Future<Output = anyhow::Result<Vec<anyhow::Result<Vec<f32>>>>>,
    {
        self.precompute_batched_with_checkpoint(states, encode_batch_fn, batch_size, concurrency, dim, None, 20_000).await
    }

    /// Like [`precompute_batched`](Self::precompute_batched) but with periodic
    /// checkpoint saves for crash resilience.
    pub async fn precompute_batched_with_checkpoint<F, Fut>(
        &mut self,
        states: &HashSet<&str>,
        encode_batch_fn: F,
        batch_size: usize,
        concurrency: usize,
        dim: usize,
        checkpoint_path: Option<&Path>,
        checkpoint_interval: usize,
    ) -> (usize, usize)
    where
        F: Fn(Vec<String>) -> Fut,
        Fut: Future<Output = anyhow::Result<Vec<anyhow::Result<Vec<f32>>>>>,
    {
        // Filter to states not already cached
        let missing: Vec<String> = states
            .iter()
            .filter(|s| !self.embeddings.contains_key(**s))
            .map(|s| s.to_string())
            .collect();

        if missing.is_empty() {
            tracing::info!("All {} states already cached — skipping precompute", states.len());
            return (0, 0);
        }

        let total = missing.len();
        let batch_size = batch_size.max(1);
        let num_batches = (total + batch_size - 1) / batch_size;

        tracing::info!(
            total_states = states.len(),
            cached = states.len() - total,
            to_encode = total,
            batch_size,
            num_batches,
            concurrency,
            "Starting batched embedding precomputation"
        );

        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({msg}) Encoding embeddings (batched)")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("=> "),
        );

        let mut encoded = 0usize;
        let mut errors = 0usize;
        let start = Instant::now();
        let mut done = 0usize;
        let mut last_checkpoint = 0usize;

        // Chunk missing states into batches, stream with bounded concurrency
        let chunks: Vec<Vec<String>> = missing
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut result_stream = stream::iter(chunks.into_iter().map(|chunk| {
            let fut = encode_batch_fn(chunk.clone());
            async move { (chunk, fut.await) }
        }))
        .buffer_unordered(concurrency);

        while let Some((chunk, batch_result)) = result_stream.next().await {
            match batch_result {
                Ok(results) => {
                    for (state, result) in chunk.iter().zip(results.into_iter()) {
                        match result {
                            Ok(emb) => {
                                debug_assert_eq!(
                                    emb.len(),
                                    dim,
                                    "Embedding dim mismatch: expected {dim}, got {}",
                                    emb.len()
                                );
                                self.embeddings.insert(state.clone(), emb);
                                encoded += 1;
                            }
                            Err(e) => {
                                errors += 1;
                                tracing::debug!(state = %state, error = %e, "Failed to encode state in batch");
                            }
                        }
                        pb.inc(1);
                        done += 1;
                    }
                }
                Err(e) => {
                    // Whole batch failed — count all states as errors
                    errors += chunk.len();
                    done += chunk.len();
                    tracing::warn!(
                        batch_size = chunk.len(),
                        error = %e,
                        "Entire batch failed to encode"
                    );
                    pb.inc(chunk.len() as u64);
                }
            }
            pb.set_message(format_eta(start, done, total));

            // Periodic checkpoint save
            if let Some(cp_path) = checkpoint_path {
                if encoded - last_checkpoint >= checkpoint_interval {
                    if let Err(e) = self.save(cp_path) {
                        tracing::warn!(error = %e, "Failed to save checkpoint");
                    } else {
                        tracing::info!(
                            encoded,
                            total_cached = self.embeddings.len(),
                            path = %cp_path.display(),
                            "Checkpoint saved"
                        );
                    }
                    last_checkpoint = encoded;
                }
            }
        }

        // Final checkpoint save (capture any remaining since last checkpoint)
        if let Some(cp_path) = checkpoint_path {
            if encoded > last_checkpoint {
                if let Err(e) = self.save(cp_path) {
                    tracing::warn!(error = %e, "Failed to save final checkpoint");
                } else {
                    tracing::info!(
                        encoded,
                        total_cached = self.embeddings.len(),
                        path = %cp_path.display(),
                        "Final checkpoint saved"
                    );
                }
            }
        }

        pb.finish_with_message("done");

        if errors > 0 {
            tracing::warn!(
                errors,
                "Some states failed to encode (use RUST_LOG=debug for details)"
            );
        }

        tracing::info!(
            total_states = states.len(),
            newly_encoded = encoded,
            errors,
            total_cached = self.embeddings.len(),
            "Batched embedding precomputation complete"
        );

        (encoded, errors)
    }

    /// Look up the embedding for a proof state.
    pub fn get(&self, state_pp: &str) -> Option<&[f32]> {
        self.embeddings.get(state_pp).map(|v| v.as_slice())
    }

    /// Look up the embedding for a proof state, returning an error if missing.
    pub fn get_or_err(&self, state_pp: &str) -> anyhow::Result<Vec<f32>> {
        self.embeddings
            .get(state_pp)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Embedding not found in cache for state: {}", state_pp))
    }

    /// Insert an embedding into the cache.
    ///
    /// Used for lazy (on-demand) cache population during training.
    pub fn insert(&mut self, state: String, embedding: Vec<f32>) {
        debug_assert_eq!(
            embedding.len(),
            self.dim,
            "Embedding dim mismatch: expected {}, got {}",
            self.dim,
            embedding.len()
        );
        self.embeddings.insert(state, embedding);
    }

    /// Number of cached embeddings.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Save the cache to a Parquet file (atomic write via temp file + rename).
    ///
    /// Schema: `state_pp: Utf8`, `embedding: List<Float32>`.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let schema = Arc::new(embedding_cache_schema());

        let mut state_pps = Vec::with_capacity(self.embeddings.len());
        let mut all_values = Vec::new();
        let mut offsets = vec![0i32];

        for (state, emb) in &self.embeddings {
            state_pps.push(state.as_str());
            all_values.extend_from_slice(emb);
            offsets.push(all_values.len() as i32);
        }

        let state_array: StringArray = state_pps.iter().map(|s| Some(*s)).collect();
        let values_array = Float32Array::from(all_values);
        let offsets_array = OffsetBuffer::new(offsets.into());
        let list_array = ListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            offsets_array,
            Arc::new(values_array),
            None,
        );

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(state_array), Arc::new(list_array)],
        )?;

        // Write to temp file then rename for atomic save (prevents corruption on crash)
        let tmp_path = path.with_extension("parquet.tmp");
        let file = std::fs::File::create(&tmp_path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        std::fs::rename(&tmp_path, path)?;

        tracing::info!(
            entries = self.embeddings.len(),
            dim = self.dim,
            path = %path.display(),
            "Saved embedding cache to Parquet"
        );

        Ok(())
    }

    /// Load a cache from a Parquet file.
    ///
    /// Validates that all embeddings have the same dimension.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        let mut embeddings = HashMap::new();
        let mut dim: Option<usize> = None;

        for batch_result in reader {
            let batch = batch_result?;

            let states = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow::anyhow!("Column 0 is not StringArray"))?;

            let lists = batch
                .column(1)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| anyhow::anyhow!("Column 1 is not ListArray"))?;

            for i in 0..batch.num_rows() {
                let state = states.value(i).to_string();
                let values = lists.value(i);
                let float_array = values
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow::anyhow!("List values are not Float32Array"))?;

                let emb: Vec<f32> = float_array.values().to_vec();

                match dim {
                    None => dim = Some(emb.len()),
                    Some(d) => {
                        if emb.len() != d {
                            anyhow::bail!(
                                "Dimension mismatch in cache: expected {d}, got {} for state '{state}'",
                                emb.len()
                            );
                        }
                    }
                }

                embeddings.insert(state, emb);
            }
        }

        let dim = dim.unwrap_or(0);

        tracing::info!(
            entries = embeddings.len(),
            dim = dim,
            path = %path.display(),
            "Loaded embedding cache from Parquet"
        );

        Ok(Self { embeddings, dim })
    }
}

/// Arrow schema for embedding cache Parquet files.
fn embedding_cache_schema() -> Schema {
    Schema::new(vec![
        Field::new("state_pp", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false,
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_get() {
        let mut cache = EmbeddingCache::new(4);
        cache
            .embeddings
            .insert("⊢ True".to_string(), vec![1.0, 2.0, 3.0, 4.0]);

        // Hit
        assert_eq!(cache.get("⊢ True"), Some(&[1.0, 2.0, 3.0, 4.0][..]));
        assert_eq!(
            cache.get_or_err("⊢ True").unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );

        // Miss
        assert!(cache.get("⊢ False").is_none());
        assert!(cache.get_or_err("⊢ False").is_err());

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
        assert_eq!(cache.dim(), 4);
    }

    #[test]
    fn test_cache_insert() {
        let mut cache = EmbeddingCache::new(3);
        assert!(cache.is_empty());

        cache.insert("⊢ A".to_string(), vec![1.0, 2.0, 3.0]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get("⊢ A"), Some(&[1.0, 2.0, 3.0][..]));

        // Overwrite existing key
        cache.insert("⊢ A".to_string(), vec![4.0, 5.0, 6.0]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get("⊢ A"), Some(&[4.0, 5.0, 6.0][..]));

        // Insert second key
        cache.insert("⊢ B".to_string(), vec![7.0, 8.0, 9.0]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cache.parquet");

        let mut cache = EmbeddingCache::new(3);
        cache
            .embeddings
            .insert("state_a".to_string(), vec![0.1, 0.2, 0.3]);
        cache
            .embeddings
            .insert("state_b".to_string(), vec![0.4, 0.5, 0.6]);
        cache
            .embeddings
            .insert("state_c".to_string(), vec![0.7, 0.8, 0.9]);

        cache.save(&path).unwrap();
        assert!(path.exists());

        let loaded = EmbeddingCache::load(&path).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.dim(), 3);
        assert_eq!(loaded.get("state_a").unwrap(), &[0.1, 0.2, 0.3]);
        assert_eq!(loaded.get("state_b").unwrap(), &[0.4, 0.5, 0.6]);
        assert_eq!(loaded.get("state_c").unwrap(), &[0.7, 0.8, 0.9]);
    }

    #[test]
    fn test_precompute_deduplicates() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Build sampler with duplicate state_pp values
        let records = vec![
            super::super::data::ProofStateRecord {
                theorem_name: "thm".to_string(),
                state_pp: "⊢ A".to_string(),
                label: "positive".to_string(),
                depth_from_root: 0,
                remaining_depth: 1,
                llm_log_prob: -0.5,
            },
            super::super::data::ProofStateRecord {
                theorem_name: "thm".to_string(),
                state_pp: "⊢ B".to_string(),
                label: "negative".to_string(),
                depth_from_root: 1,
                remaining_depth: -1,
                llm_log_prob: -0.5,
            },
            super::super::data::ProofStateRecord {
                theorem_name: "thm".to_string(),
                state_pp: "⊢ A".to_string(), // duplicate of first
                label: "positive".to_string(),
                depth_from_root: 0,
                remaining_depth: 1,
                llm_log_prob: -0.5,
            },
        ];

        let sampler = ContrastiveSampler::from_trajectory_records(records, 1).unwrap();

        let call_count = AtomicUsize::new(0);
        let encode_fn = |_state: &str| -> anyhow::Result<Vec<f32>> {
            call_count.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0.1, 0.2])
        };

        let cache = EmbeddingCache::precompute(&sampler, &encode_fn, 2);

        // 2 unique states ("⊢ A" and "⊢ B"), so encode_fn should be called exactly 2 times
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
        assert_eq!(cache.len(), 2);
        assert!(cache.get("⊢ A").is_some());
        assert!(cache.get("⊢ B").is_some());
    }

    #[tokio::test]
    async fn test_precompute_batched_groups_and_handles_errors() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let call_count = Arc::new(AtomicUsize::new(0));

        // Pre-populate cache with one state
        let mut cache = EmbeddingCache::new(2);
        cache.insert("⊢ cached".to_string(), vec![9.0, 9.0]);

        // 5 states: 1 cached, 3 new, 1 that will fail inside a batch
        let states: HashSet<&str> =
            ["⊢ cached", "⊢ a", "⊢ b", "⊢ c", "⊢ fail"].into_iter().collect();

        let cc = call_count.clone();
        let encode_batch_fn = move |texts: Vec<String>| {
            let cc = cc.clone();
            async move {
                cc.fetch_add(1, Ordering::SeqCst);
                let results: Vec<anyhow::Result<Vec<f32>>> = texts
                    .iter()
                    .map(|t| {
                        if t == "⊢ fail" {
                            Err(anyhow::anyhow!("simulated failure"))
                        } else {
                            Ok(vec![1.0, 2.0])
                        }
                    })
                    .collect();
                Ok(results)
            }
        };

        let (encoded, errors) = cache
            .precompute_batched(&states, encode_batch_fn, 2, 4, 2)
            .await;

        // "⊢ cached" skipped; 3 new encoded; 1 error
        assert_eq!(encoded, 3);
        assert_eq!(errors, 1);
        assert_eq!(cache.get("⊢ cached"), Some(&[9.0, 9.0][..]));
        assert!(cache.get("⊢ a").is_some());
        assert!(cache.get("⊢ b").is_some());
        assert!(cache.get("⊢ c").is_some());
        assert!(cache.get("⊢ fail").is_none());
        assert_eq!(cache.len(), 4); // 1 pre-existing + 3 new

        // 4 missing states, batch_size=2 → 2 batch calls
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_precompute_batched_whole_batch_failure() {
        let mut cache = EmbeddingCache::new(2);
        let states: HashSet<&str> = ["⊢ x", "⊢ y"].into_iter().collect();

        let encode_batch_fn = |_texts: Vec<String>| async move {
            Err(anyhow::anyhow!("HTTP timeout"))
        };

        let (encoded, errors) = cache
            .precompute_batched(&states, encode_batch_fn, 4, 1, 2)
            .await;

        assert_eq!(encoded, 0);
        assert_eq!(errors, 2);
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn test_precompute_concurrent_skips_cached_and_reports_errors() {
        // Pre-populate cache with one state
        let mut cache = EmbeddingCache::new(2);
        cache.insert("⊢ cached".to_string(), vec![9.0, 9.0]);

        // States to precompute: one already cached, two new, one that will fail
        let states: HashSet<&str> = ["⊢ cached", "⊢ new_a", "⊢ new_b", "⊢ fail"]
            .into_iter()
            .collect();

        let encode_fn = |state: String| async move {
            if state == "⊢ fail" {
                anyhow::bail!("simulated encode failure");
            }
            Ok(vec![1.0, 2.0])
        };

        let (encoded, errors) = cache
            .precompute_concurrent(&states, encode_fn, 4, 2, None, 10_000)
            .await;

        // "⊢ cached" should be skipped (not re-encoded)
        assert_eq!(cache.get("⊢ cached"), Some(&[9.0, 9.0][..]));
        // Two new states should be encoded
        assert_eq!(encoded, 2);
        assert!(cache.get("⊢ new_a").is_some());
        assert!(cache.get("⊢ new_b").is_some());
        // One error
        assert_eq!(errors, 1);
        assert!(cache.get("⊢ fail").is_none());
        // Total cached: 1 pre-existing + 2 new = 3
        assert_eq!(cache.len(), 3);
    }
}
