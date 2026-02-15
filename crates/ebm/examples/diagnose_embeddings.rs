//! Diagnose EBM training signal: embedding similarity, energy distribution, and sampler stats.
//!
//! Usage:
//!   cargo run --release -p ebm --example diagnose_embeddings -- \
//!     --embeddings checkpoints/ebm/iter_1/embeddings.parquet \
//!     --trajectories trajectories/iter_0.parquet trajectories/iter_0_noisy.parquet \
//!                    trajectories/iter_1.parquet \
//!     --checkpoint checkpoints/ebm/iter_1

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use burn::backend::NdArray;
use clap::Parser;
use rand::Rng;

use ebm::{
    load_records_from_parquet, ContrastiveSampler, EBMScorer, EmbeddingCache, EnergyHeadConfig,
};

#[derive(Parser)]
struct Args {
    /// Path to embeddings cache Parquet.
    #[arg(long)]
    embeddings: PathBuf,
    /// Path(s) to trajectory Parquet files.
    #[arg(long, num_args = 1..)]
    trajectories: Vec<PathBuf>,
    /// Path to EBM checkpoint directory (optional — for energy distribution analysis).
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Number of negative samples per positive for the sampler.
    #[arg(long, default_value_t = 7)]
    k_negatives: usize,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn print_stats(name: &str, vals: &mut Vec<f32>) {
    if vals.is_empty() {
        println!("  {name}: no data");
        return;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    let mean = vals.iter().sum::<f32>() / n as f32;
    let std = (vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32).sqrt();
    println!("  {name} (n={n}):");
    println!("    Mean={mean:.4}  Std={std:.4}");
    println!(
        "    Min={:.4}  P10={:.4}  P50={:.4}  P90={:.4}  Max={:.4}",
        vals[0],
        vals[n / 10],
        vals[n / 2],
        vals[9 * n / 10],
        vals[n - 1]
    );
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // 1. Load embeddings
    println!("Loading embeddings from {}...", args.embeddings.display());
    let cache = EmbeddingCache::load(&args.embeddings)?;
    println!("  Loaded {} embeddings, dim={}", cache.len(), cache.dim());

    // 2. Load trajectory records
    println!(
        "Loading trajectories from {} file(s)...",
        args.trajectories.len()
    );
    let records = load_records_from_parquet(&args.trajectories)?;
    println!("  Loaded {} records", records.len());

    let sampler = ContrastiveSampler::from_trajectory_records(records.clone(), args.k_negatives)?;
    println!(
        "  Sampler: {} records, {} eligible theorems",
        sampler.num_records(),
        sampler.num_eligible_theorems()
    );

    // 3. Group records by theorem and label
    let mut by_theorem: HashMap<&str, (Vec<&ebm::ProofStateRecord>, Vec<&ebm::ProofStateRecord>)> =
        HashMap::new();
    for r in &records {
        let entry = by_theorem
            .entry(r.theorem_name.as_str())
            .or_insert_with(|| (Vec::new(), Vec::new()));
        if r.label == "positive" {
            entry.0.push(r);
        } else {
            entry.1.push(r);
        }
    }

    // 3b. Per-theorem sample distribution diagnostics
    println!("\n=== Per-Theorem Sample Distribution ===");
    let mut pos_counts: Vec<usize> = Vec::new();
    let mut neg_counts: Vec<usize> = Vec::new();
    let mut tactic_pair_count = 0usize;
    let mut search_pos_count = 0usize;
    let mut sparse_hard_neg_count = 0usize;

    for (_thm, (positives, negatives)) in &by_theorem {
        pos_counts.push(positives.len());
        neg_counts.push(negatives.len());

        // Distinguish tactic pair positives (llm_log_prob == 0.0) from search positives
        for p in positives {
            if p.llm_log_prob == 0.0 {
                tactic_pair_count += 1;
            } else {
                search_pos_count += 1;
            }
        }

        // Theorems with fewer than 3 hard negatives risk repetitive sampling
        if negatives.len() < 3 {
            sparse_hard_neg_count += 1;
        }
    }

    pos_counts.sort();
    neg_counts.sort();
    let total_theorems = by_theorem.len();
    let eligible = by_theorem
        .values()
        .filter(|(p, n)| !p.is_empty() && !n.is_empty())
        .count();

    if !pos_counts.is_empty() {
        let n = pos_counts.len();
        println!("  Theorems: {total_theorems} total, {eligible} eligible (have both pos+neg)");
        println!(
            "  Pos per theorem: min={} median={} max={}",
            pos_counts[0],
            pos_counts[n / 2],
            pos_counts[n - 1]
        );
    }
    if !neg_counts.is_empty() {
        let n = neg_counts.len();
        println!(
            "  Neg per theorem: min={} median={} max={}",
            neg_counts[0],
            neg_counts[n / 2],
            neg_counts[n - 1]
        );
    }

    let total_pos = tactic_pair_count + search_pos_count;
    if total_pos > 0 {
        println!(
            "  Positive source: {} tactic-pair ({:.1}%) + {} search ({:.1}%)",
            tactic_pair_count,
            tactic_pair_count as f64 / total_pos as f64 * 100.0,
            search_pos_count,
            search_pos_count as f64 / total_pos as f64 * 100.0,
        );
    }
    println!(
        "  Theorems with <3 hard negatives: {} ({:.1}%)",
        sparse_hard_neg_count,
        sparse_hard_neg_count as f64 / total_theorems.max(1) as f64 * 100.0,
    );

    // 4. Embedding statistics
    println!("\n=== Embedding Norm Statistics ===");
    let mut norms: Vec<f32> = Vec::new();
    for state in sampler.unique_states() {
        if let Some(emb) = cache.get(state) {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            norms.push(norm);
        }
    }
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !norms.is_empty() {
        let n = norms.len();
        println!("  Count: {n}");
        println!("  Min:    {:.4}", norms[0]);
        println!("  P25:    {:.4}", norms[n / 4]);
        println!("  Median: {:.4}", norms[n / 2]);
        println!("  P75:    {:.4}", norms[3 * n / 4]);
        println!("  Max:    {:.4}", norms[n - 1]);
        println!(
            "  Mean:   {:.4}",
            norms.iter().sum::<f32>() / n as f32
        );
    }

    // 5. Intra-theorem cosine similarity (pos vs neg within same theorem)
    println!("\n=== Intra-Theorem Similarity (Positive vs Negative) ===");
    let mut pos_neg_sims: Vec<f32> = Vec::new();
    let mut pos_pos_sims: Vec<f32> = Vec::new();
    let mut neg_neg_sims: Vec<f32> = Vec::new();
    let mut pos_neg_l2s: Vec<f32> = Vec::new();

    for (_thm, (positives, negatives)) in &by_theorem {
        // Get embeddings
        let pos_embs: Vec<&[f32]> = positives
            .iter()
            .filter_map(|r| cache.get(r.state_pp.as_str()))
            .collect();
        let neg_embs: Vec<&[f32]> = negatives
            .iter()
            .filter_map(|r| cache.get(r.state_pp.as_str()))
            .collect();

        // Pos vs neg
        for p in &pos_embs {
            for n in &neg_embs {
                pos_neg_sims.push(cosine_similarity(p, n));
                pos_neg_l2s.push(l2_distance(p, n));
            }
        }

        // Pos vs pos
        for i in 0..pos_embs.len() {
            for j in (i + 1)..pos_embs.len() {
                pos_pos_sims.push(cosine_similarity(pos_embs[i], pos_embs[j]));
            }
        }

        // Neg vs neg
        for i in 0..neg_embs.len() {
            for j in (i + 1)..neg_embs.len() {
                neg_neg_sims.push(cosine_similarity(neg_embs[i], neg_embs[j]));
            }
        }
    }

    print_stats("Pos-vs-Neg cosine", &mut pos_neg_sims);
    print_stats("Pos-vs-Pos cosine", &mut pos_pos_sims);
    print_stats("Neg-vs-Neg cosine", &mut neg_neg_sims);
    print_stats("Pos-vs-Neg L2", &mut pos_neg_l2s);

    // 6. Cross-theorem similarity (random pairs from different theorems)
    println!("\n=== Cross-Theorem Similarity (Random Sample) ===");
    let all_states: Vec<&str> = sampler.unique_states().into_iter().collect();
    let mut cross_sims: Vec<f32> = Vec::new();
    let sample_size = 5000.min(all_states.len() * (all_states.len() - 1) / 2);
    let mut rng = rand::thread_rng();
    for _ in 0..sample_size {
        let i = rng.gen_range(0..all_states.len());
        let j = rng.gen_range(0..all_states.len());
        if i == j {
            continue;
        }
        if let (Some(a), Some(b)) = (cache.get(all_states[i]), cache.get(all_states[j])) {
            cross_sims.push(cosine_similarity(a, b));
        }
    }
    print_stats("Cross-theorem cosine", &mut cross_sims);

    // 7. Energy distribution (if checkpoint provided)
    if let Some(ref ckpt_dir) = args.checkpoint {
        println!("\n=== Energy Distribution (Trained EBM) ===");
        match load_and_score_ebm(ckpt_dir, &sampler, &cache, &records, &mut rng) {
            Ok(()) => {}
            Err(e) => {
                println!("  ERROR: Failed to load/score EBM checkpoint: {e}");
                println!("  (Skipping energy distribution analysis)");
            }
        }
    }

    // 8. Flush stdout before state length analysis (ensures all sections visible even on crash)
    use std::io::Write;
    std::io::stdout().flush().ok();

    // 9. State length analysis
    println!("\n=== State Length Statistics ===");
    let mut pos_lens: Vec<usize> = Vec::new();
    let mut neg_lens: Vec<usize> = Vec::new();
    for r in &records {
        if r.label == "positive" {
            pos_lens.push(r.state_pp.len());
        } else {
            neg_lens.push(r.state_pp.len());
        }
    }
    pos_lens.sort();
    neg_lens.sort();
    if !pos_lens.is_empty() {
        let n = pos_lens.len();
        println!(
            "  Positive (n={}): mean={:.0} median={} max={}",
            n,
            pos_lens.iter().sum::<usize>() as f64 / n as f64,
            pos_lens[n / 2],
            pos_lens[n - 1]
        );
    }
    if !neg_lens.is_empty() {
        let n = neg_lens.len();
        println!(
            "  Negative (n={}): mean={:.0} median={} max={}",
            n,
            neg_lens.iter().sum::<usize>() as f64 / n as f64,
            neg_lens[n / 2],
            neg_lens[n - 1]
        );
    }

    Ok(())
}

/// Load EBM checkpoint and compute energy distribution stats.
///
/// Extracted so errors here don't kill the rest of diagnostics.
fn load_and_score_ebm(
    ckpt_dir: &Path,
    sampler: &ContrastiveSampler,
    cache: &EmbeddingCache,
    records: &[ebm::ProofStateRecord],
    rng: &mut impl Rng,
) -> anyhow::Result<()> {
    let cfg_path = ckpt_dir.join("energy_head_config.json");
    let config_json = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("Cannot read {}: {e}", cfg_path.display()))?;
    let head_config: EnergyHeadConfig = serde_json::from_str(&config_json)
        .map_err(|e| anyhow::anyhow!("Cannot parse EnergyHeadConfig: {e}"))?;

    // Copy embeddings into an Arc<HashMap> for the 'static closure
    let emb_map: std::sync::Arc<HashMap<String, Vec<f32>>> = {
        let mut map = HashMap::new();
        for state in sampler.unique_states() {
            if let Some(emb) = cache.get(state) {
                map.insert(state.to_string(), emb.to_vec());
            }
        }
        std::sync::Arc::new(map)
    };

    let encode_fn: Box<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync> =
        Box::new(move |state: &str| {
            emb_map
                .get(state)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("not in cache: {state}"))
        });

    let checkpoint_path = ckpt_dir.join("final");
    let device = Default::default();
    let scorer = EBMScorer::<NdArray<f32>>::load(
        &checkpoint_path,
        &head_config,
        encode_fn,
        device,
    )?;

    let mut pos_energies: Vec<f32> = Vec::new();
    let mut neg_energies: Vec<f32> = Vec::new();

    // Batch scoring: collect records with cached embeddings, score in chunks
    let scorable: Vec<&ebm::ProofStateRecord> = records
        .iter()
        .filter(|r| cache.get(&r.state_pp).is_some())
        .collect();
    println!("  Scoring {} states in batches of 512...", scorable.len());

    for chunk in scorable.chunks(512) {
        let states: Vec<&str> = chunk.iter().map(|r| r.state_pp.as_str()).collect();
        match scorer.score_states(&states) {
            Ok(scores) => {
                for (r, e) in chunk.iter().zip(scores) {
                    if r.label == "positive" {
                        pos_energies.push(e as f32);
                    } else {
                        neg_energies.push(e as f32);
                    }
                }
            }
            Err(e) => {
                println!("  Warning: batch scoring failed: {e}");
            }
        }
    }

    // Note: score_state() returns -energy (higher = more provable)
    print_stats("Positive scores", &mut pos_energies);
    print_stats("Negative scores", &mut neg_energies);

    if !pos_energies.is_empty() && !neg_energies.is_empty() {
        let pos_mean = pos_energies.iter().sum::<f32>() / pos_energies.len() as f32;
        let neg_mean = neg_energies.iter().sum::<f32>() / neg_energies.len() as f32;
        let gap = pos_mean - neg_mean;
        println!("\n  Score gap (pos_mean - neg_mean): {gap:.4}");
        println!("  (Higher score = more provable; gap > 0 means model is working)");

        // Ranking accuracy: for random (pos, neg) pairs, how often does the model
        // correctly rank the positive state higher (more provable)?
        let mut correct = 0u64;
        let mut total = 0u64;
        let pairs = 10000.min(pos_energies.len() * neg_energies.len());
        for _ in 0..pairs {
            let pi = rng.gen_range(0..pos_energies.len());
            let ni = rng.gen_range(0..neg_energies.len());
            total += 1;
            if pos_energies[pi] > neg_energies[ni] {
                correct += 1;
            }
        }
        println!(
            "  Ranking accuracy (pos > neg): {}/{} = {:.1}%",
            correct,
            total,
            correct as f64 / total as f64 * 100.0
        );
    }

    Ok(())
}
