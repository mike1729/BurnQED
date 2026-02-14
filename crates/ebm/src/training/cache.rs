//! Precomputed embedding cache for EBM training.
//!
//! Since the LLM encoder is frozen, `state_pp` → embedding is deterministic.
//! This module precomputes all unique embeddings once and stores them in a
//! `HashMap<String, Vec<f32>>` for O(1) lookups during training. Optionally
//! persists to Parquet for reuse across runs.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::*;
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

use super::data::ContrastiveSampler;

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
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) Encoding embeddings")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("=> "),
        );

        let mut embeddings = HashMap::with_capacity(total);
        let mut errors = 0usize;

        for state in &unique_states {
            match encode_fn(state) {
                Ok(emb) => {
                    debug_assert_eq!(emb.len(), dim, "Embedding dim mismatch: expected {dim}, got {}", emb.len());
                    embeddings.insert(state.to_string(), emb);
                }
                Err(e) => {
                    errors += 1;
                    tracing::warn!(state = %state, error = %e, "Failed to encode state");
                }
            }
            pb.inc(1);
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

    /// Save the cache to a Parquet file.
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

        let file = std::fs::File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;

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
}
