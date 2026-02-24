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
    /// Pantograph state ID for this node.
    pub state_id: u64,
    /// State ID of the parent node. None for root nodes.
    pub parent_state_id: Option<u64>,
}

/// Index for efficient contrastive sampling.
///
/// Groups record indices by theorem and label for fast negative mining.
/// Distinguishes sibling negatives (share a parent with a positive) from
/// non-sibling same-theorem negatives for 3-tier sampling.
struct ContrastiveIndex {
    /// Positive record indices per theorem.
    pos_by_theorem: HashMap<String, Vec<usize>>,
    /// Sibling negatives: share a parent_state_id with a positive in the same theorem.
    sibling_neg_by_theorem: HashMap<String, Vec<usize>>,
    /// Non-sibling negatives: same theorem but not a sibling of any positive.
    non_sibling_neg_by_theorem: HashMap<String, Vec<usize>>,
    /// All negative record indices (for easy negatives from other theorems).
    all_negatives: Vec<usize>,
    /// Theorems that have BOTH positive AND negative records.
    eligible_theorems: Vec<String>,
}

impl ContrastiveIndex {
    /// Build the contrastive index from a slice of records.
    ///
    /// Two passes: first collects positive parent IDs per theorem, then
    /// classifies each negative as sibling (shares parent with a positive)
    /// or non-sibling.
    fn build(records: &[ProofStateRecord]) -> Self {
        let mut pos_by_theorem: HashMap<String, Vec<usize>> = HashMap::new();
        let mut all_negatives = Vec::new();
        let mut neg_indices_by_theorem: HashMap<String, Vec<usize>> = HashMap::new();

        // First pass: index positives and collect all negatives
        // Also build positive_parents: set of (theorem, parent_state_id) for positives
        let mut positive_parents: HashSet<(String, u64)> = HashSet::new();

        for (i, record) in records.iter().enumerate() {
            match record.label.as_str() {
                "positive" => {
                    pos_by_theorem
                        .entry(record.theorem_name.clone())
                        .or_default()
                        .push(i);
                    if let Some(pid) = record.parent_state_id {
                        positive_parents.insert((record.theorem_name.clone(), pid));
                    }
                }
                "negative" => {
                    neg_indices_by_theorem
                        .entry(record.theorem_name.clone())
                        .or_default()
                        .push(i);
                    all_negatives.push(i);
                }
                _ => {} // skip unknown labels
            }
        }

        // Second pass: classify negatives as sibling vs non-sibling
        let mut sibling_neg_by_theorem: HashMap<String, Vec<usize>> = HashMap::new();
        let mut non_sibling_neg_by_theorem: HashMap<String, Vec<usize>> = HashMap::new();

        for (theorem, indices) in &neg_indices_by_theorem {
            for &i in indices {
                let is_sibling = records[i]
                    .parent_state_id
                    .map(|pid| positive_parents.contains(&(theorem.clone(), pid)))
                    .unwrap_or(false);

                if is_sibling {
                    sibling_neg_by_theorem
                        .entry(theorem.clone())
                        .or_default()
                        .push(i);
                } else {
                    non_sibling_neg_by_theorem
                        .entry(theorem.clone())
                        .or_default()
                        .push(i);
                }
            }
        }

        // Eligible theorems: any theorem with positive records.
        // Theorems without same-theorem negatives will backfill from the
        // global negative pool (cross-theorem easy negatives). This ensures
        // positive-only theorems from generate-negatives (upfront recording)
        // still participate in training.
        let eligible_theorems: Vec<String> = pos_by_theorem.keys().cloned().collect();

        ContrastiveIndex {
            pos_by_theorem,
            sibling_neg_by_theorem,
            non_sibling_neg_by_theorem,
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
/// Implements a 3-tier negative mining strategy:
/// - **Hard (30%)**: sibling negatives — share a `parent_state_id` with a positive
///   (diverge at one tactic choice, most informative for contrastive learning)
/// - **Medium (40%)**: same-theorem negatives that are NOT siblings of any positive
/// - **Easy (30%)**: random negatives from other theorems
///
/// When a tier is exhausted for a theorem, overflows into the next tier
/// (hard → medium → easy → all_negatives).
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
            anyhow::bail!("No eligible theorems found (need at least one theorem with positive records)");
        }
        if index.all_negatives.is_empty() {
            anyhow::bail!(
                "No negative records found — cannot build contrastive pairs. \
                 Ensure trajectory data contains divergent states (negatives)."
            );
        }
        let sibling_neg_count: usize = index.sibling_neg_by_theorem.values().map(|v| v.len()).sum();
        let non_sibling_neg_count: usize = index.non_sibling_neg_by_theorem.values().map(|v| v.len()).sum();
        tracing::info!(
            eligible = index.eligible_theorems.len(),
            sibling_neg = sibling_neg_count,
            non_sibling_neg = non_sibling_neg_count,
            total_neg = index.all_negatives.len(),
            "ContrastiveSampler initialized"
        );
        Ok(ContrastiveSampler {
            records,
            index,
            k_negatives,
            hard_ratio: 0.3,
            medium_ratio: 0.4,
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

    /// Sample a single contrastive example using the sampler's configured ratios.
    ///
    /// Picks a random positive from an eligible theorem, then mines K negatives
    /// using the 3-tier hard/medium/easy strategy. If a tier is exhausted,
    /// overflows into the next tier (hard → medium → easy → all_negatives).
    pub fn sample(&self, rng: &mut impl Rng) -> ContrastiveSample {
        self.sample_with_ratios(rng, self.hard_ratio, self.medium_ratio)
    }

    /// Sample a single contrastive example with explicit hard/medium ratios.
    ///
    /// Like [`sample`], but overrides the sampler's configured ratios for this
    /// single draw. Useful for final validation with a different distribution
    /// (e.g., natural search distribution) without cloning the sampler.
    pub fn sample_with_ratios(
        &self,
        rng: &mut impl Rng,
        hard_ratio: f64,
        medium_ratio: f64,
    ) -> ContrastiveSample {
        // Pick a random eligible theorem
        let theorem = self.index.eligible_theorems.choose(rng).unwrap();

        // Pick a random positive from this theorem
        let pos_indices = &self.index.pos_by_theorem[theorem];
        let &pos_idx = pos_indices.choose(rng).unwrap();
        let positive = self.records[pos_idx].clone();

        // Compute category counts
        let n_hard = (self.k_negatives as f64 * hard_ratio).round() as usize;
        let n_medium = (self.k_negatives as f64 * medium_ratio).round() as usize;
        let n_easy = self.k_negatives.saturating_sub(n_hard + n_medium);

        let mut negatives = Vec::with_capacity(self.k_negatives);

        // Hard negatives: sibling negatives (share parent_state_id with a positive)
        let hard_pool = self.index.sibling_neg_by_theorem.get(theorem);
        if let Some(pool) = hard_pool {
            self.sample_from_pool(pool, n_hard, rng, &mut negatives);
        }
        let hard_shortfall = n_hard.saturating_sub(negatives.len());

        // Medium negatives: same-theorem negatives that are NOT siblings
        // Also absorbs hard shortfall.
        let n_medium_total = n_medium + hard_shortfall;
        let before_medium = negatives.len();
        if let Some(pool) = self.index.non_sibling_neg_by_theorem.get(theorem) {
            self.sample_from_pool(pool, n_medium_total, rng, &mut negatives);
        }
        let medium_shortfall = n_medium_total.saturating_sub(negatives.len() - before_medium);

        // Easy negatives: random states from OTHER theorems
        // Also absorbs medium shortfall.
        let n_easy_total = n_easy + medium_shortfall;
        for _ in 0..n_easy_total {
            if self.index.all_negatives.is_empty() {
                break;
            }
            // Try up to 3 times to find a different theorem, then accept anyway
            for attempt in 0..3 {
                let &idx = self.index.all_negatives.choose(rng).unwrap();
                if attempt == 2 || self.records[idx].theorem_name != *theorem {
                    negatives.push(self.records[idx].clone());
                    break;
                }
            }
        }

        // Pad with random negatives if still undersupplied
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

    /// Sample a batch of contrastive examples with explicit hard/medium ratios.
    pub fn sample_batch_with_ratios(
        &self,
        batch_size: usize,
        rng: &mut impl Rng,
        hard_ratio: f64,
        medium_ratio: f64,
    ) -> Vec<ContrastiveSample> {
        (0..batch_size)
            .map(|_| self.sample_with_ratios(rng, hard_ratio, medium_ratio))
            .collect()
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
            state_id: r.state_id,
            parent_state_id: r.parent_state_id,
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
                state_id: 0,
                parent_state_id: None,
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

    fn make_record_with_ids(
        theorem: &str,
        label: &str,
        depth: u32,
        state_id: u64,
        parent_state_id: Option<u64>,
    ) -> ProofStateRecord {
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
            state_id,
            parent_state_id,
        }
    }

    fn make_record(theorem: &str, label: &str, depth: u32) -> ProofStateRecord {
        make_record_with_ids(theorem, label, depth, 0, None)
    }

    /// Build test records with sibling structure:
    ///
    /// Theorem A (parent chain: None→0→1→2):
    ///   pos depth=0: state_id=100, parent=None
    ///   pos depth=1: state_id=101, parent=100
    ///   pos depth=2: state_id=102, parent=101
    ///   neg depth=0: state_id=200, parent=100  (sibling of pos depth=1)
    ///   neg depth=1: state_id=201, parent=101  (sibling of pos depth=2)
    ///   neg depth=2: state_id=202, parent=999  (NOT a sibling — different parent)
    ///
    /// Theorem B (parent chain: None→10→11):
    ///   pos depth=0: state_id=110, parent=None
    ///   pos depth=1: state_id=111, parent=110
    ///   neg depth=0: state_id=210, parent=110  (sibling of pos depth=1)
    ///   neg depth=1: state_id=211, parent=888  (NOT a sibling)
    fn make_test_records() -> Vec<ProofStateRecord> {
        let mut records = Vec::new();
        // Theorem A positives
        records.push(make_record_with_ids("thm_a", "positive", 0, 100, None));
        records.push(make_record_with_ids("thm_a", "positive", 1, 101, Some(100)));
        records.push(make_record_with_ids("thm_a", "positive", 2, 102, Some(101)));
        // Theorem A negatives: 2 siblings + 1 non-sibling
        records.push(make_record_with_ids("thm_a", "negative", 0, 200, Some(100)));  // sibling (parent=100)
        records.push(make_record_with_ids("thm_a", "negative", 1, 201, Some(101)));  // sibling (parent=101)
        records.push(make_record_with_ids("thm_a", "negative", 2, 202, Some(999)));  // non-sibling
        // Theorem B positives
        records.push(make_record_with_ids("thm_b", "positive", 0, 110, None));
        records.push(make_record_with_ids("thm_b", "positive", 1, 111, Some(110)));
        // Theorem B negatives: 1 sibling + 1 non-sibling
        records.push(make_record_with_ids("thm_b", "negative", 0, 210, Some(110)));  // sibling (parent=110)
        records.push(make_record_with_ids("thm_b", "negative", 1, 211, Some(888)));  // non-sibling
        records
    }

    #[test]
    fn test_build_index() {
        let records = make_test_records();
        let index = ContrastiveIndex::build(&records);

        assert_eq!(index.pos_by_theorem["thm_a"].len(), 3);
        assert_eq!(index.pos_by_theorem["thm_b"].len(), 2);
        // Theorem A: 2 siblings (parent=100, parent=101) + 1 non-sibling (parent=999)
        assert_eq!(index.sibling_neg_by_theorem["thm_a"].len(), 2);
        assert_eq!(index.non_sibling_neg_by_theorem["thm_a"].len(), 1);
        // Theorem B: 1 sibling (parent=110) + 1 non-sibling (parent=888)
        assert_eq!(index.sibling_neg_by_theorem["thm_b"].len(), 1);
        assert_eq!(index.non_sibling_neg_by_theorem["thm_b"].len(), 1);
        assert_eq!(index.all_negatives.len(), 5); // 3 + 2
        assert_eq!(index.eligible_theorems.len(), 2);
    }

    #[test]
    fn test_eligible_theorems() {
        // ALL theorems with positives are eligible (negatives backfilled
        // from global pool for theorems without same-theorem negatives)
        let mut records = make_test_records();
        records.push(make_record("thm_c", "positive", 0));
        records.push(make_record("thm_c", "positive", 1));
        // thm_c has no negatives but is still eligible

        let index = ContrastiveIndex::build(&records);
        assert!(index.eligible_theorems.contains(&"thm_c".to_string()));
        assert_eq!(index.eligible_theorems.len(), 3); // thm_a, thm_b, and thm_c
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
    fn test_hard_negatives_are_siblings() {
        // Build records where thm_a has enough siblings to fill 100% hard
        let mut records = Vec::new();
        // 1 positive with parent=100
        records.push(make_record_with_ids("thm_a", "positive", 1, 50, Some(100)));
        // 4 sibling negatives (all share parent=100 with the positive)
        for i in 0..4 {
            records.push(make_record_with_ids("thm_a", "negative", 1, 300 + i, Some(100)));
        }
        // 1 non-sibling negative (different parent)
        records.push(make_record_with_ids("thm_a", "negative", 2, 400, Some(999)));
        // Need at least one other theorem for the sampler
        records.push(make_record_with_ids("thm_b", "positive", 0, 500, None));
        records.push(make_record_with_ids("thm_b", "negative", 0, 600, Some(777)));

        let sampler =
            ContrastiveSampler::from_trajectory_records(records.clone(), 4)
                .unwrap()
                .with_ratios(1.0, 0.0); // 100% hard (siblings)

        let mut rng = rand::thread_rng();
        // Sample many times from thm_a and verify all negs are siblings
        let sibling_ids: HashSet<u64> = [300, 301, 302, 303].into();
        for _ in 0..50 {
            let sample = sampler.sample(&mut rng);
            if sample.positive.theorem_name == "thm_a" {
                for neg in &sample.negatives {
                    assert!(
                        sibling_ids.contains(&neg.state_id),
                        "Hard negative should be sibling (state_id={}), not non-sibling",
                        neg.state_id
                    );
                }
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
    fn test_default_ratios_all_negatives_labeled_negative() {
        // With default 3-tier ratios (hard=0.3, medium=0.4, easy=0.3),
        // all sampled negatives should have label="negative" — no positive
        // states should ever appear as negatives.
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
    fn test_backfill_when_siblings_exhausted() {
        // Theorem with 1 positive, 0 siblings, 2 non-siblings.
        // Requesting 4 negatives with hard_ratio=0.5, medium=0.25, easy=0.25
        // Hard pool is empty → all hard slots overflow to medium, then easy.
        let mut records = Vec::new();
        records.push(make_record_with_ids("thm_x", "positive", 0, 10, None));
        // No sibling negatives (parent IDs don't match any positive's parent)
        records.push(make_record_with_ids("thm_x", "negative", 1, 20, Some(777)));
        records.push(make_record_with_ids("thm_x", "negative", 2, 21, Some(888)));
        // Another theorem for cross-theorem pool
        records.push(make_record_with_ids("thm_y", "positive", 0, 30, None));
        records.push(make_record_with_ids("thm_y", "negative", 0, 40, Some(999)));

        let sampler =
            ContrastiveSampler::from_trajectory_records(records, 4)
                .unwrap()
                .with_ratios(0.5, 0.25); // hard=0.5, medium=0.25, easy=0.25

        let mut rng = rand::thread_rng();
        // Should always produce exactly 4 negatives despite no siblings
        for _ in 0..50 {
            let sample = sampler.sample(&mut rng);
            assert_eq!(
                sample.negatives.len(),
                4,
                "Should backfill to exactly K negatives"
            );
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
            state_id: 0,
            parent_state_id: None,
        });
        records.push(ProofStateRecord {
            theorem_name: "thm_b".to_string(),
            state_pp: records[6].state_pp.clone(), // duplicate
            label: "negative".to_string(),
            depth_from_root: 0,
            remaining_depth: -1,
            llm_log_prob: -0.5,
            state_id: 0,
            parent_state_id: None,
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
