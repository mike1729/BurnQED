//! Contrastive data pipeline for EBM training.
//!
//! Reads trajectory Parquet files via the `trajectory` crate and constructs
//! contrastive samples for training. Does NOT tokenize or encode — provides
//! raw proof state strings. The training loop calls `encoder.encode_only()`
//! on each string (the encoder lives outside burn, accessed via a closure).

use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::path::{Path, PathBuf};

use trajectory::TrajectoryReader;

/// A proof state record for contrastive training.
#[derive(Clone, Debug)]
pub struct ProofStateRecord {
    /// Name of the theorem.
    pub theorem_name: String,
    /// Pretty-printed proof state.
    pub state_pp: String,
    /// Label: "positive" or "negative".
    pub label: String,
    /// Depth from root in the search tree.
    pub depth_from_root: u32,
    /// Remaining steps to QED on proof path, -1 if unknown/off path.
    pub remaining_depth: i32,
    /// Log probability from the LLM policy for the tactic.
    pub llm_log_prob: f64,
}

/// Index for efficient contrastive sampling.
///
/// Groups record indices by theorem and label for fast negative mining.
struct ContrastiveIndex {
    /// Positive record indices per theorem.
    pos_by_theorem: HashMap<String, Vec<usize>>,
    /// Negative record indices per theorem.
    neg_by_theorem: HashMap<String, Vec<usize>>,
    /// All negative record indices (for easy negatives from other theorems).
    all_negatives: Vec<usize>,
    /// Theorems that have BOTH positive AND negative records.
    eligible_theorems: Vec<String>,
}

impl ContrastiveIndex {
    /// Build the contrastive index from a slice of records.
    fn build(records: &[ProofStateRecord]) -> Self {
        let mut pos_by_theorem: HashMap<String, Vec<usize>> = HashMap::new();
        let mut neg_by_theorem: HashMap<String, Vec<usize>> = HashMap::new();
        let mut all_negatives = Vec::new();

        for (i, record) in records.iter().enumerate() {
            match record.label.as_str() {
                "positive" => {
                    pos_by_theorem
                        .entry(record.theorem_name.clone())
                        .or_default()
                        .push(i);
                }
                "negative" => {
                    neg_by_theorem
                        .entry(record.theorem_name.clone())
                        .or_default()
                        .push(i);
                    all_negatives.push(i);
                }
                _ => {} // skip unknown labels
            }
        }

        // Eligible theorems must have BOTH positive AND negative records
        let eligible_theorems: Vec<String> = pos_by_theorem
            .keys()
            .filter(|t| neg_by_theorem.contains_key(*t))
            .cloned()
            .collect();

        ContrastiveIndex {
            pos_by_theorem,
            neg_by_theorem,
            all_negatives,
            eligible_theorems,
        }
    }
}

/// A single contrastive training example.
#[derive(Clone, Debug)]
pub struct ContrastiveSample {
    /// The positive (on-path) proof state.
    pub positive: ProofStateRecord,
    /// Exactly K negative proof states.
    pub negatives: Vec<ProofStateRecord>,
    /// Remaining depth for the positive state.
    pub remaining_depth: i32,
}

/// Samples contrastive batches from trajectory data.
///
/// Implements a negative mining strategy with two categories:
/// - **Hard (70%)**: dead-end states from the same theorem
/// - **Easy (30%)**: random states from other theorems
///
/// Medium negatives (positive states used as negatives) are disabled by default
/// because tactic pair augmentation adds many positive-only records. Using
/// positives as negatives creates contradictory gradients and collapses training.
/// Use `with_ratios()` to override if needed.
pub struct ContrastiveSampler {
    records: Vec<ProofStateRecord>,
    index: ContrastiveIndex,
    k_negatives: usize,
    hard_ratio: f64,
    medium_ratio: f64,
    // easy_ratio = 1.0 - hard - medium
}

impl ContrastiveSampler {
    /// Create a sampler from pre-built records.
    ///
    /// # Errors
    /// Returns an error if no eligible theorems (with both pos + neg records) exist.
    pub fn from_trajectory_records(
        records: Vec<ProofStateRecord>,
        k_negatives: usize,
    ) -> anyhow::Result<Self> {
        let index = ContrastiveIndex::build(&records);
        if index.eligible_theorems.is_empty() {
            anyhow::bail!(
                "No eligible theorems found (need theorems with both positive and negative records)"
            );
        }
        Ok(ContrastiveSampler {
            records,
            index,
            k_negatives,
            hard_ratio: 0.7,
            medium_ratio: 0.0,
        })
    }

    /// Create a sampler from Parquet trajectory files.
    ///
    /// Reads via `TrajectoryReader::read_multiple`, converts to `ProofStateRecord`,
    /// and filters out Unknown labels.
    pub fn from_parquet(
        paths: &[PathBuf],
        k_negatives: usize,
    ) -> anyhow::Result<Self> {
        let records = load_records_from_parquet(paths)?;
        Self::from_trajectory_records(records, k_negatives)
    }

    /// Override the negative mining ratios.
    ///
    /// `hard_ratio` + `medium_ratio` must be <= 1.0.
    /// Easy ratio is computed as `1.0 - hard - medium`.
    pub fn with_ratios(mut self, hard_ratio: f64, medium_ratio: f64) -> Self {
        assert!(
            hard_ratio + medium_ratio <= 1.0 + 1e-9,
            "hard + medium ratios must be <= 1.0"
        );
        self.hard_ratio = hard_ratio;
        self.medium_ratio = medium_ratio;
        self
    }

    /// Sample a single contrastive example.
    ///
    /// Picks a random positive from an eligible theorem, then mines K negatives
    /// using the hard/medium/easy strategy.
    pub fn sample(&self, rng: &mut impl Rng) -> ContrastiveSample {
        // Pick a random eligible theorem
        let theorem = self.index.eligible_theorems.choose(rng).unwrap();

        // Pick a random positive from this theorem
        let pos_indices = &self.index.pos_by_theorem[theorem];
        let &pos_idx = pos_indices.choose(rng).unwrap();
        let positive = self.records[pos_idx].clone();

        // Compute category counts
        let n_hard = (self.k_negatives as f64 * self.hard_ratio).round() as usize;
        let n_medium = (self.k_negatives as f64 * self.medium_ratio).round() as usize;
        let n_easy = self.k_negatives.saturating_sub(n_hard + n_medium);

        let mut negatives = Vec::with_capacity(self.k_negatives);

        // Hard negatives: dead-end states from SAME theorem
        if let Some(neg_indices) = self.index.neg_by_theorem.get(theorem) {
            self.sample_from_pool(neg_indices, n_hard, rng, &mut negatives);
        }

        // Medium negatives: other positives from same theorem (off-path siblings)
        // Exclude the current positive
        let other_pos: Vec<usize> = pos_indices
            .iter()
            .filter(|&&i| i != pos_idx)
            .copied()
            .collect();
        self.sample_from_pool(&other_pos, n_medium, rng, &mut negatives);

        // Easy negatives: random states from OTHER theorems
        let other_negs: Vec<usize> = self
            .index
            .all_negatives
            .iter()
            .filter(|&&i| self.records[i].theorem_name != *theorem)
            .copied()
            .collect();
        self.sample_from_pool(&other_negs, n_easy, rng, &mut negatives);

        // Pad with random negatives if undersupplied
        while negatives.len() < self.k_negatives {
            if !self.index.all_negatives.is_empty() {
                let &idx = self.index.all_negatives.choose(rng).unwrap();
                negatives.push(self.records[idx].clone());
            } else {
                // Fallback: use any record as negative
                let idx = rng.gen_range(0..self.records.len());
                negatives.push(self.records[idx].clone());
            }
        }

        let remaining_depth = positive.remaining_depth;
        ContrastiveSample {
            positive,
            negatives,
            remaining_depth,
        }
    }

    /// Sample a batch of contrastive examples.
    pub fn sample_batch(
        &self,
        batch_size: usize,
        rng: &mut impl Rng,
    ) -> Vec<ContrastiveSample> {
        (0..batch_size).map(|_| self.sample(rng)).collect()
    }

    /// Extract all unique `state_pp` values from the records.
    pub fn unique_states(&self) -> std::collections::HashSet<&str> {
        self.records.iter().map(|r| r.state_pp.as_str()).collect()
    }

    /// Total number of records in the sampler.
    pub fn num_records(&self) -> usize {
        self.records.len()
    }

    /// Number of eligible theorems (with both pos and neg records).
    pub fn num_eligible_theorems(&self) -> usize {
        self.index.eligible_theorems.len()
    }

    /// Sample up to `count` records from a pool of indices with replacement.
    fn sample_from_pool(
        &self,
        pool: &[usize],
        count: usize,
        rng: &mut impl Rng,
        out: &mut Vec<ProofStateRecord>,
    ) {
        if pool.is_empty() {
            return;
        }
        for _ in 0..count {
            let &idx = pool.choose(rng).unwrap();
            out.push(self.records[idx].clone());
        }
    }
}

/// Load trajectory records from Parquet files and convert to ProofStateRecords.
///
/// Uses `TrajectoryReader::read_multiple` for I/O. Filters out "unknown" labels.
pub fn load_records_from_parquet(
    paths: &[PathBuf],
) -> anyhow::Result<Vec<ProofStateRecord>> {
    let trajectory_records = TrajectoryReader::read_multiple(paths)?;

    let records: Vec<ProofStateRecord> = trajectory_records
        .into_iter()
        .filter(|r| {
            let label = r.label.to_string();
            label == "positive" || label == "negative"
        })
        .map(|r| ProofStateRecord {
            theorem_name: r.theorem_name,
            state_pp: r.state_pp,
            label: r.label.to_string(),
            depth_from_root: r.depth_from_root,
            remaining_depth: r.remaining_depth,
            llm_log_prob: r.llm_log_prob,
        })
        .collect();

    Ok(records)
}

/// A single tactic pair record from the training JSONL file.
#[derive(serde::Deserialize)]
struct TacticPairJson {
    state: String,
    tactic: String,
    theorem: String,
    depth: u32,
}

/// A single step along a known-good proof path, used by the generate-negatives pipeline.
#[derive(Debug, Clone)]
pub struct TacticStep {
    /// Pretty-printed proof state at this step.
    pub state: String,
    /// Ground-truth tactic applied at this step.
    pub tactic: String,
    /// 0-indexed depth from the proof root.
    pub depth: u32,
}

/// Load ground-truth proof states from a tactic pairs JSONL file.
///
/// Each line is `{"state", "tactic", "theorem", "depth"}`. Records are grouped
/// by theorem to compute `remaining_depth = max_depth_in_theorem - depth`.
/// All records are labeled "positive" with `llm_log_prob = 0.0`.
///
/// If `filter_theorems` is `Some`, only records whose `theorem` field matches
/// a name in the set are loaded. This keeps the merge targeted to theorems
/// that already have negative records from search trajectories.
pub fn load_tactic_pairs(
    path: &Path,
    filter_theorems: Option<&HashSet<String>>,
) -> anyhow::Result<Vec<ProofStateRecord>> {
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open tactic pairs file {}: {e}", path.display()))?;
    let reader = std::io::BufReader::new(file);

    // First pass: parse all matching records, grouped by theorem
    let mut by_theorem: HashMap<String, Vec<(String, u32)>> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let pair: TacticPairJson = serde_json::from_str(&line)
            .map_err(|e| anyhow::anyhow!("Failed to parse tactic pair JSON: {e}"))?;

        if let Some(filter) = filter_theorems {
            if !filter.contains(&pair.theorem) {
                continue;
            }
        }

        by_theorem
            .entry(pair.theorem)
            .or_default()
            .push((pair.state, pair.depth));
    }

    // Second pass: compute remaining_depth per theorem and build records
    let mut records = Vec::new();
    let mut theorem_count = 0u32;

    for (theorem_name, states) in &by_theorem {
        let max_depth = states.iter().map(|(_, d)| *d).max().unwrap_or(0);
        theorem_count += 1;

        for (state_pp, depth) in states {
            let remaining_depth = max_depth as i32 - *depth as i32;
            records.push(ProofStateRecord {
                theorem_name: theorem_name.clone(),
                state_pp: state_pp.clone(),
                label: "positive".to_string(),
                depth_from_root: *depth,
                remaining_depth,
                llm_log_prob: 0.0,
            });
        }
    }

    tracing::info!(
        records = records.len(),
        theorems = theorem_count,
        "Loaded tactic pair records"
    );

    Ok(records)
}

/// Load tactic pairs from a JSONL file, grouped by theorem with steps sorted by depth.
///
/// Returns `Vec<(theorem_name, steps)>` for deterministic iteration. Filters out
/// theorems with non-contiguous depths (gaps indicate incomplete traces). If
/// `max_theorems` is set, randomly samples that many theorems from the result.
pub fn load_tactic_pairs_grouped(
    path: &Path,
    max_theorems: Option<usize>,
    min_steps: Option<usize>,
) -> anyhow::Result<Vec<(String, Vec<TacticStep>)>> {
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open tactic pairs file {}: {e}", path.display()))?;
    let reader = std::io::BufReader::new(file);

    // Parse all records grouped by theorem
    let mut by_theorem: HashMap<String, Vec<TacticStep>> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let pair: TacticPairJson = serde_json::from_str(&line)
            .map_err(|e| anyhow::anyhow!("Failed to parse tactic pair JSON: {e}"))?;

        by_theorem
            .entry(pair.theorem)
            .or_default()
            .push(TacticStep {
                state: pair.state,
                tactic: pair.tactic,
                depth: pair.depth,
            });
    }

    // Sort each theorem's steps by depth and filter out non-contiguous traces
    let mut result: Vec<(String, Vec<TacticStep>)> = by_theorem
        .into_iter()
        .filter_map(|(name, mut steps)| {
            steps.sort_by_key(|s| s.depth);

            // Check for contiguous depths: 0, 1, 2, ...
            let contiguous = steps
                .iter()
                .enumerate()
                .all(|(i, s)| s.depth == i as u32);
            if !contiguous {
                tracing::debug!(theorem = name, "Skipping theorem with non-contiguous depths");
                return None;
            }

            Some((name, steps))
        })
        .collect();

    // Filter by minimum number of proof steps
    if let Some(min) = min_steps {
        let before = result.len();
        result.retain(|(_, steps)| steps.len() >= min);
        tracing::info!(
            before,
            after = result.len(),
            min_steps = min,
            "Filtered theorems by minimum step count"
        );
    }

    // Sort by theorem name for deterministic ordering
    result.sort_by(|a, b| a.0.cmp(&b.0));

    // Sample if max_theorems is set
    if let Some(max) = max_theorems {
        if max < result.len() {
            let mut rng = rand::thread_rng();
            result.as_mut_slice().shuffle(&mut rng);
            result.truncate(max);
            // Re-sort for deterministic iteration after sampling
            result.sort_by(|a, b| a.0.cmp(&b.0));
        }
    }

    tracing::info!(
        theorems = result.len(),
        total_steps = result.iter().map(|(_, s)| s.len()).sum::<usize>(),
        "Loaded tactic pairs grouped by theorem"
    );

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(theorem: &str, label: &str, depth: u32) -> ProofStateRecord {
        ProofStateRecord {
            theorem_name: theorem.to_string(),
            state_pp: format!("⊢ state_{theorem}_{label}_{depth}"),
            label: label.to_string(),
            depth_from_root: depth,
            remaining_depth: if label == "positive" {
                3_i32 - depth as i32
            } else {
                -1
            },
            llm_log_prob: -0.5,
        }
    }

    fn make_test_records() -> Vec<ProofStateRecord> {
        let mut records = Vec::new();
        // Theorem A: 3 positive, 3 negative
        for d in 0..3 {
            records.push(make_record("thm_a", "positive", d));
        }
        for d in 0..3 {
            records.push(make_record("thm_a", "negative", d));
        }
        // Theorem B: 2 positive, 2 negative
        for d in 0..2 {
            records.push(make_record("thm_b", "positive", d));
        }
        for d in 0..2 {
            records.push(make_record("thm_b", "negative", d));
        }
        records
    }

    #[test]
    fn test_build_index() {
        let records = make_test_records();
        let index = ContrastiveIndex::build(&records);

        assert_eq!(index.pos_by_theorem["thm_a"].len(), 3);
        assert_eq!(index.neg_by_theorem["thm_a"].len(), 3);
        assert_eq!(index.pos_by_theorem["thm_b"].len(), 2);
        assert_eq!(index.neg_by_theorem["thm_b"].len(), 2);
        assert_eq!(index.all_negatives.len(), 5); // 3 + 2
        assert_eq!(index.eligible_theorems.len(), 2);
    }

    #[test]
    fn test_eligible_theorems() {
        // Theorem with only positives should NOT be eligible
        let mut records = make_test_records();
        records.push(make_record("thm_c", "positive", 0));
        records.push(make_record("thm_c", "positive", 1));
        // thm_c has no negatives

        let index = ContrastiveIndex::build(&records);
        assert!(!index.eligible_theorems.contains(&"thm_c".to_string()));
        assert_eq!(index.eligible_theorems.len(), 2); // only thm_a and thm_b
    }

    #[test]
    fn test_sample_returns_correct_k() {
        let records = make_test_records();
        let sampler =
            ContrastiveSampler::from_trajectory_records(records, 4).unwrap();

        let mut rng = rand::thread_rng();
        let sample = sampler.sample(&mut rng);

        assert_eq!(
            sample.negatives.len(),
            4,
            "Expected exactly 4 negatives, got {}",
            sample.negatives.len()
        );
        assert_eq!(sample.positive.label, "positive");
    }

    #[test]
    fn test_hard_negatives_from_same_theorem() {
        let records = make_test_records();
        let sampler =
            ContrastiveSampler::from_trajectory_records(records, 4)
                .unwrap()
                .with_ratios(1.0, 0.0); // 100% hard negatives

        let mut rng = rand::thread_rng();
        // Sample many times and check that hard negatives come from same theorem
        for _ in 0..20 {
            let sample = sampler.sample(&mut rng);
            let pos_theorem = &sample.positive.theorem_name;
            // With hard_ratio=1.0, all negatives should come from same theorem
            // (they're from neg_by_theorem[theorem])
            for neg in &sample.negatives {
                assert_eq!(
                    &neg.theorem_name, pos_theorem,
                    "Hard negative should be from same theorem"
                );
            }
        }
    }

    #[test]
    fn test_easy_negatives_from_other_theorems() {
        let records = make_test_records();
        let sampler =
            ContrastiveSampler::from_trajectory_records(records, 4)
                .unwrap()
                .with_ratios(0.0, 0.0); // 100% easy negatives

        let mut rng = rand::thread_rng();
        let mut found_other_theorem = false;
        // Easy negatives should come from OTHER theorems (when available)
        for _ in 0..50 {
            let sample = sampler.sample(&mut rng);
            let pos_theorem = &sample.positive.theorem_name;
            for neg in &sample.negatives {
                if neg.theorem_name != *pos_theorem {
                    found_other_theorem = true;
                }
            }
        }
        assert!(
            found_other_theorem,
            "Easy negatives should include states from other theorems"
        );
    }

    #[test]
    fn test_sample_batch_size() {
        let records = make_test_records();
        let sampler =
            ContrastiveSampler::from_trajectory_records(records, 2).unwrap();

        let mut rng = rand::thread_rng();
        let batch = sampler.sample_batch(8, &mut rng);
        assert_eq!(batch.len(), 8);
        for sample in &batch {
            assert_eq!(sample.negatives.len(), 2);
        }
    }

    #[test]
    fn test_load_tactic_pairs_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pairs.jsonl");
        let content = r#"{"state":"⊢ True","tactic":"trivial","theorem":"True_self","depth":0}
{"state":"⊢ True ∧ True","tactic":"constructor","theorem":"True_self","depth":1}
{"state":"⊢ False → False","tactic":"intro","theorem":"false_imp","depth":0}
"#;
        std::fs::write(&path, content).unwrap();

        let records = load_tactic_pairs(&path, None).unwrap();
        assert_eq!(records.len(), 3);
        assert!(records.iter().all(|r| r.label == "positive"));
        assert!(records.iter().all(|r| r.llm_log_prob == 0.0));

        // True_self has max_depth=1, so depth=0 → remaining=1, depth=1 → remaining=0
        let true_self: Vec<_> = records.iter().filter(|r| r.theorem_name == "True_self").collect();
        assert_eq!(true_self.len(), 2);
        let depths: HashSet<i32> = true_self.iter().map(|r| r.remaining_depth).collect();
        assert!(depths.contains(&0));
        assert!(depths.contains(&1));
    }

    #[test]
    fn test_load_tactic_pairs_filtered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pairs.jsonl");
        let content = r#"{"state":"⊢ True","tactic":"trivial","theorem":"True_self","depth":0}
{"state":"⊢ False → False","tactic":"intro","theorem":"false_imp","depth":0}
{"state":"⊢ 1 + 1 = 2","tactic":"norm_num","theorem":"one_plus_one","depth":0}
"#;
        std::fs::write(&path, content).unwrap();

        let mut filter = HashSet::new();
        filter.insert("True_self".to_string());
        filter.insert("one_plus_one".to_string());

        let records = load_tactic_pairs(&path, Some(&filter)).unwrap();
        assert_eq!(records.len(), 2);
        let names: HashSet<_> = records.iter().map(|r| r.theorem_name.as_str()).collect();
        assert!(names.contains("True_self"));
        assert!(names.contains("one_plus_one"));
        assert!(!names.contains("false_imp"));
    }

    #[test]
    fn test_load_tactic_pairs_remaining_depth() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pairs.jsonl");
        // Theorem with 4 steps: depth 0,1,2,3 → remaining 3,2,1,0
        let content = r#"{"state":"s0","tactic":"t0","theorem":"thm","depth":0}
{"state":"s1","tactic":"t1","theorem":"thm","depth":1}
{"state":"s2","tactic":"t2","theorem":"thm","depth":2}
{"state":"s3","tactic":"t3","theorem":"thm","depth":3}
"#;
        std::fs::write(&path, content).unwrap();

        let records = load_tactic_pairs(&path, None).unwrap();
        assert_eq!(records.len(), 4);
        for r in &records {
            let expected_remaining = 3 - r.depth_from_root as i32;
            assert_eq!(r.remaining_depth, expected_remaining);
        }
    }

    #[test]
    fn test_default_ratios_no_medium_negatives() {
        // With default ratios (hard=0.7, medium=0.0), no positive state
        // should ever appear as a negative.
        let records = make_test_records();
        let sampler =
            ContrastiveSampler::from_trajectory_records(records, 4).unwrap();

        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let sample = sampler.sample(&mut rng);
            for neg in &sample.negatives {
                assert_eq!(
                    neg.label, "negative",
                    "Default ratios should never use positive states as negatives, \
                     but got positive state '{}' from theorem '{}'",
                    neg.state_pp, neg.theorem_name
                );
            }
        }
    }

    #[test]
    fn test_unique_states() {
        let mut records = make_test_records(); // 10 records with 10 unique state_pp values
        // Duplicate some state_pp values
        records.push(ProofStateRecord {
            theorem_name: "thm_a".to_string(),
            state_pp: records[0].state_pp.clone(), // duplicate
            label: "positive".to_string(),
            depth_from_root: 0,
            remaining_depth: 3,
            llm_log_prob: -0.5,
        });
        records.push(ProofStateRecord {
            theorem_name: "thm_b".to_string(),
            state_pp: records[6].state_pp.clone(), // duplicate
            label: "negative".to_string(),
            depth_from_root: 0,
            remaining_depth: -1,
            llm_log_prob: -0.5,
        });
        let sampler =
            ContrastiveSampler::from_trajectory_records(records, 2).unwrap();
        let unique = sampler.unique_states();
        // Original 10 unique states, duplicates should be deduped
        assert_eq!(unique.len(), 10);
    }

    #[test]
    fn test_load_tactic_pairs_grouped_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pairs.jsonl");
        let content = r#"{"state":"⊢ True","tactic":"trivial","theorem":"True_self","depth":0}
{"state":"⊢ True ∧ True","tactic":"constructor","theorem":"True_self","depth":1}
{"state":"⊢ False → False","tactic":"intro h","theorem":"false_imp","depth":0}
{"state":"h : False\n⊢ False","tactic":"exact h","theorem":"false_imp","depth":1}
{"state":"⊢ 1 + 1 = 2","tactic":"norm_num","theorem":"one_plus_one","depth":0}
"#;
        std::fs::write(&path, content).unwrap();

        let grouped = load_tactic_pairs_grouped(&path, None, None).unwrap();
        assert_eq!(grouped.len(), 3);

        // Sorted by theorem name
        assert_eq!(grouped[0].0, "True_self");
        assert_eq!(grouped[1].0, "false_imp");
        assert_eq!(grouped[2].0, "one_plus_one");

        // True_self has 2 steps sorted by depth
        assert_eq!(grouped[0].1.len(), 2);
        assert_eq!(grouped[0].1[0].depth, 0);
        assert_eq!(grouped[0].1[0].tactic, "trivial");
        assert_eq!(grouped[0].1[1].depth, 1);
        assert_eq!(grouped[0].1[1].tactic, "constructor");

        // one_plus_one has 1 step
        assert_eq!(grouped[2].1.len(), 1);
    }

    #[test]
    fn test_load_tactic_pairs_grouped_filters_gaps() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pairs.jsonl");
        // "gappy" has depths [0, 2] — gap at 1 → should be excluded
        let content = r#"{"state":"s0","tactic":"t0","theorem":"gappy","depth":0}
{"state":"s2","tactic":"t2","theorem":"gappy","depth":2}
{"state":"⊢ True","tactic":"trivial","theorem":"contiguous","depth":0}
{"state":"⊢ True ∧ True","tactic":"constructor","theorem":"contiguous","depth":1}
"#;
        std::fs::write(&path, content).unwrap();

        let grouped = load_tactic_pairs_grouped(&path, None, None).unwrap();
        assert_eq!(grouped.len(), 1);
        assert_eq!(grouped[0].0, "contiguous");
    }

    #[test]
    fn test_load_tactic_pairs_grouped_sampling() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pairs.jsonl");
        let mut content = String::new();
        for i in 0..5 {
            content.push_str(&format!(
                r#"{{"state":"⊢ s{i}","tactic":"t{i}","theorem":"thm_{i}","depth":0}}"#
            ));
            content.push('\n');
        }
        std::fs::write(&path, content).unwrap();

        let grouped = load_tactic_pairs_grouped(&path, Some(2), None).unwrap();
        assert_eq!(grouped.len(), 2);
        // Each entry should still be a valid theorem with steps
        for (name, steps) in &grouped {
            assert!(name.starts_with("thm_"));
            assert_eq!(steps.len(), 1);
        }
    }

    #[test]
    fn test_load_tactic_pairs_grouped_min_steps() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pairs.jsonl");
        // thm_short: 1 step, thm_medium: 2 steps, thm_long: 3 steps
        let content = r#"{"state":"s0","tactic":"t0","theorem":"thm_short","depth":0}
{"state":"s0","tactic":"t0","theorem":"thm_medium","depth":0}
{"state":"s1","tactic":"t1","theorem":"thm_medium","depth":1}
{"state":"s0","tactic":"t0","theorem":"thm_long","depth":0}
{"state":"s1","tactic":"t1","theorem":"thm_long","depth":1}
{"state":"s2","tactic":"t2","theorem":"thm_long","depth":2}
"#;
        std::fs::write(&path, content).unwrap();

        // min_steps=2 should keep thm_medium and thm_long
        let grouped = load_tactic_pairs_grouped(&path, None, Some(2)).unwrap();
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].0, "thm_long");
        assert_eq!(grouped[0].1.len(), 3);
        assert_eq!(grouped[1].0, "thm_medium");
        assert_eq!(grouped[1].1.len(), 2);

        // min_steps=3 should keep only thm_long
        let grouped = load_tactic_pairs_grouped(&path, None, Some(3)).unwrap();
        assert_eq!(grouped.len(), 1);
        assert_eq!(grouped[0].0, "thm_long");
    }
}
