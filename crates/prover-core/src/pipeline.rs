//! Proof search pipeline, evaluation, and EBM training utilities.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use indicatif::{ProgressBar, ProgressStyle};

use ebm::{
    ContrastiveSampler, EBMScorer, EBMTrainingConfig, EBMValueFn, EmbeddingCache, EnergyHeadConfig,
};
use lean_repl::LeanPool;
use policy::{PolicyConfig, TacticGenerator};
use search::{MutexPolicyProvider, SearchEngine};
use trajectory::{TheoremIndex, TrajectoryReader, TrajectoryWriter};

use crate::config::{build_lean_pool_config, load_search_toml};

/// Arguments for the `search` subcommand.
#[derive(Debug)]
pub struct SearchArgs {
    /// Path to the search config TOML file.
    pub config: PathBuf,
    /// Path to the HuggingFace model directory.
    pub model_path: PathBuf,
    /// Path to the theorem index JSON file.
    pub theorems: PathBuf,
    /// Path for the output Parquet file.
    pub output: PathBuf,
    /// Optional CLI override for number of Lean workers.
    pub num_workers: Option<usize>,
    /// Load model and pool but don't search. Verifies environment setup.
    pub dry_run: bool,
    /// Path to EBM checkpoint directory for value-guided search.
    pub ebm_path: Option<PathBuf>,
}

/// Arguments for the `eval` subcommand.
#[derive(Debug)]
pub struct EvalArgs {
    /// Path to the trajectory Parquet file.
    pub input: PathBuf,
}

/// Arguments for the `train-ebm` subcommand.
#[derive(Debug)]
pub struct TrainEbmArgs {
    /// Path(s) to trajectory Parquet files.
    pub trajectories: Vec<PathBuf>,
    /// Directory for saving checkpoints.
    pub output_dir: PathBuf,
    /// Path to the HuggingFace LLM model directory (for encoding).
    pub llm_path: PathBuf,
    /// Resume training from a checkpoint directory.
    pub resume_from: Option<PathBuf>,
    /// Total training steps.
    pub steps: usize,
    /// Learning rate.
    pub lr: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Number of negative samples per positive.
    pub k_negatives: usize,
    /// Path to precomputed embedding cache (Parquet). If omitted, precomputes from LLM.
    pub embeddings_cache: Option<PathBuf>,
    /// Save precomputed embeddings to this path for reuse.
    pub save_embeddings: Option<PathBuf>,
}

/// Backend types used for EBM inference and training.
type InferenceBackend = NdArray<f32>;
type TrainingBackend = Autodiff<NdArray<f32>>;

/// Run proof search over a batch of theorems and write trajectory data to Parquet.
pub async fn run_search(args: SearchArgs) -> anyhow::Result<()> {
    let start = Instant::now();

    // 1. Load config
    let toml = load_search_toml(&args.config)?;

    // 2. Build Lean pool
    let lean_config = build_lean_pool_config(&toml.lean_pool, args.num_workers)?;
    tracing::info!(
        num_workers = lean_config.num_workers,
        "Starting Lean worker pool"
    );
    let pool = Arc::new(LeanPool::new(lean_config).await?);

    // 3. Load LLM policy
    tracing::info!(model = %args.model_path.display(), "Loading policy model");
    let policy_config = PolicyConfig::new(args.model_path.clone());
    let generator = TacticGenerator::load(&policy_config)?;

    // 4. Set up policy + optional EBM scorer
    let (policy, value_fn) = if let Some(ref ebm_path) = args.ebm_path {
        tracing::info!(path = %ebm_path.display(), "Loading EBM scorer");

        // Load EnergyHeadConfig from JSON alongside checkpoint
        let config_path = ebm_path.join("energy_head_config.json");
        let config_json = std::fs::read_to_string(&config_path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to read EnergyHeadConfig from {}: {e}",
                config_path.display()
            )
        })?;
        let head_config: EnergyHeadConfig = serde_json::from_str(&config_json).map_err(|e| {
            anyhow::anyhow!("Failed to parse EnergyHeadConfig: {e}")
        })?;

        // Share the generator between policy and EBM encode closure
        let shared_gen = Arc::new(std::sync::Mutex::new(generator));
        let policy = MutexPolicyProvider::new_shared(shared_gen.clone());

        // Create encode_fn closure that uses the shared generator
        let encode_gen = shared_gen.clone();
        let encode_fn: Box<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync> =
            Box::new(move |state: &str| {
                let mut gen = encode_gen
                    .lock()
                    .map_err(|e| anyhow::anyhow!("Generator lock poisoned: {e}"))?;
                let embedding = gen.encode_only(state)?;
                Ok(embedding.data)
            });

        // Load EBM scorer from checkpoint
        let checkpoint_path = ebm_path.join("final");
        let device = Default::default();
        let scorer =
            EBMScorer::<InferenceBackend>::load(&checkpoint_path, &head_config, encode_fn, device)?;
        let value_fn = EBMValueFn::new(scorer);

        tracing::info!("EBM scorer loaded successfully");
        (policy, Some(value_fn))
    } else {
        let policy = MutexPolicyProvider::new(generator);
        (policy, None)
    };

    // 5. Load theorem index
    let index = TheoremIndex::from_json(&args.theorems)?;
    tracing::info!(count = index.len(), "Loaded theorems");

    // 5b. Dry-run: verify setup and exit early
    if args.dry_run {
        println!("Dry run — setup verified successfully");
        println!("  Model: {}", args.model_path.display());
        println!("  Theorems: {} loaded", index.len());
        println!("  Workers: {}", pool.num_workers());
        println!(
            "  EBM: {}",
            if args.ebm_path.is_some() {
                "loaded"
            } else {
                "none"
            }
        );
        pool.shutdown().await;
        return Ok(());
    }

    // 6. Run search with progress bar
    let engine = SearchEngine::new(toml.search);
    let mut writer = TrajectoryWriter::new(args.output.clone());
    let mut proved_count: u32 = 0;
    let mut failed_count: u32 = 0;
    let total = index.len() as u32;

    // Aggregate stats
    let mut total_nodes: u64 = 0;
    let mut total_lean_ms: u64 = 0;
    let mut total_gen_ms: u64 = 0;
    let mut searched_count: u32 = 0;
    let mut interrupted = false;

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .expect("valid progress bar template")
            .progress_chars("=> "),
    );

    let scorer_ref: Option<&dyn search::ValueScorer> =
        value_fn.as_ref().map(|v| v as &dyn search::ValueScorer);

    for task in &index.theorems {
        if interrupted {
            break;
        }

        pb.set_message(task.name.clone());

        let search_fut =
            engine.search_one(&pool, &policy, scorer_ref, &task.name, &task.statement);

        tokio::select! {
            result = search_fut => {
                match result {
                    Ok(result) => {
                        searched_count += 1;
                        total_nodes += result.nodes_expanded as u64;
                        total_lean_ms += result.stats.total_lean_time_ms;
                        total_gen_ms += result.stats.total_generate_time_ms;

                        if result.proved {
                            proved_count += 1;
                            tracing::info!(
                                theorem = task.name,
                                tactics = ?result.proof_tactics,
                                nodes = result.nodes_expanded,
                                time_ms = result.wall_time_ms,
                                "Proved"
                            );
                        } else {
                            failed_count += 1;
                        }
                        let labeled = TrajectoryWriter::from_search_result(&result);
                        writer.record_all(labeled);
                    }
                    Err(e) => {
                        failed_count += 1;
                        tracing::warn!(theorem = task.name, error = %e, "Search failed, skipping");
                    }
                }
            }
            _ = tokio::signal::ctrl_c(), if !interrupted => {
                tracing::warn!("Interrupted by CTRL-C, finishing with partial results");
                interrupted = true;
            }
        }

        pb.inc(1);
    }

    pb.finish_with_message("done");

    // 7. Write Parquet
    let record_count = writer.len();
    writer.finish()?;

    // 8. Shutdown pool
    pool.shutdown().await;

    // 9. Print enhanced summary
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    let partial_note = if interrupted {
        " (Partial — interrupted by CTRL-C)"
    } else {
        ""
    };

    let prove_pct = if searched_count > 0 {
        proved_count as f64 / searched_count as f64 * 100.0
    } else {
        0.0
    };
    let fail_pct = if searched_count > 0 {
        failed_count as f64 / searched_count as f64 * 100.0
    } else {
        0.0
    };
    let avg_nodes = if searched_count > 0 {
        total_nodes as f64 / searched_count as f64
    } else {
        0.0
    };
    let avg_time = if searched_count > 0 {
        elapsed_secs / searched_count as f64
    } else {
        0.0
    };

    println!();
    println!("══════════════════════════════════════════");
    println!(" burn-qed Search Results{partial_note}");
    println!("──────────────────────────────────────────");
    println!(" Theorems searched:  {searched_count}");
    println!(" Proved:             {proved_count} ({prove_pct:.1}%)");
    println!(" Failed:             {failed_count} ({fail_pct:.1}%)");
    println!(" Total nodes:        {total_nodes}");
    println!(" Avg nodes/theorem:  {avg_nodes:.1}");
    println!(" Avg time/theorem:   {avg_time:.1}s");
    println!(" Total Lean time:    {:.1}s", total_lean_ms as f64 / 1000.0);
    println!(" Total gen time:     {:.1}s", total_gen_ms as f64 / 1000.0);
    println!(" Total wall time:    {elapsed_secs:.1}s");
    println!(" Trajectory file:    {}", args.output.display());
    println!("   Records:          {record_count}");
    if args.ebm_path.is_some() {
        println!(" EBM scorer:         active");
    }
    println!("══════════════════════════════════════════");

    Ok(())
}

/// Print statistics from a trajectory Parquet file.
pub fn run_eval(args: EvalArgs) -> anyhow::Result<()> {
    let summary = TrajectoryReader::read_summary(&args.input)?;

    println!("--- Trajectory Summary ---");
    println!("File: {}", args.input.display());
    println!("Total records: {}", summary.total_records);
    println!("Positive: {}", summary.positive_count);
    println!("Negative: {}", summary.negative_count);
    println!(
        "Unknown: {}",
        summary.total_records - summary.positive_count - summary.negative_count
    );
    println!("Unique theorems: {}", summary.unique_theorems);
    println!("Proved theorems: {}", summary.proved_theorems);
    if summary.unique_theorems > 0 {
        let rate = summary.proved_theorems as f64 / summary.unique_theorems as f64 * 100.0;
        println!("Prove rate: {rate:.1}%");
    }

    Ok(())
}

/// Train the Energy-Based Model from trajectory data.
pub fn run_train_ebm(args: TrainEbmArgs) -> anyhow::Result<()> {
    tracing::info!(
        trajectories = args.trajectories.len(),
        steps = args.steps,
        lr = args.lr,
        batch_size = args.batch_size,
        k_negatives = args.k_negatives,
        "Starting EBM training"
    );

    // 1. Load LLM encoder
    tracing::info!(model = %args.llm_path.display(), "Loading LLM encoder");
    let policy_config = PolicyConfig::new(args.llm_path);
    let generator = TacticGenerator::load(&policy_config)?;
    let hidden_size = generator.hidden_size();

    // 2. Create or resume energy head
    let head_config = EnergyHeadConfig::new(hidden_size);
    let device: <TrainingBackend as burn::prelude::Backend>::Device = Default::default();
    let model = if let Some(ref resume_path) = args.resume_from {
        tracing::info!(path = %resume_path.display(), "Resuming from checkpoint");
        let checkpoint_path = resume_path.join("final");
        ebm::resume_from_checkpoint::<TrainingBackend>(&checkpoint_path, &head_config, &device)?
    } else {
        head_config.init::<TrainingBackend>(&device)
    };

    // 3. Load trajectory data
    tracing::info!("Loading trajectory data from {} file(s)", args.trajectories.len());
    let sampler = ContrastiveSampler::from_parquet(&args.trajectories, args.k_negatives)?;
    tracing::info!(
        records = sampler.num_records(),
        eligible_theorems = sampler.num_eligible_theorems(),
        "Trajectory data loaded"
    );

    // 4. Build embedding cache (precompute or load from disk)
    let cache = if let Some(ref cache_path) = args.embeddings_cache {
        tracing::info!(path = %cache_path.display(), "Loading embedding cache from disk");
        EmbeddingCache::load(cache_path)?
    } else {
        tracing::info!("Precomputing embeddings for all unique states");
        let gen = std::sync::Mutex::new(generator);
        let raw_encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
            let mut g = gen
                .lock()
                .map_err(|e| anyhow::anyhow!("Generator lock poisoned: {e}"))?;
            let embedding = g.encode_only(state)?;
            Ok(embedding.data)
        };
        EmbeddingCache::precompute(&sampler, &raw_encode_fn, hidden_size)
    };

    // 4b. Optionally save cache for reuse
    if let Some(ref save_path) = args.save_embeddings {
        cache.save(save_path)?;
    }

    // 5. Create encode_fn from cache (O(1) lookups)
    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> { cache.get_or_err(state) };

    // 6. Build training config
    let output_dir_str = args.output_dir.to_string_lossy().to_string();
    let training_config = EBMTrainingConfig::new()
        .with_lr(args.lr)
        .with_total_steps(args.steps)
        .with_batch_size(args.batch_size)
        .with_k_negatives(args.k_negatives)
        .with_checkpoint_dir(output_dir_str.clone());

    // 7. Train
    let _trained = ebm::train(&training_config, model, &encode_fn, &sampler, &device)?;

    // 8. Save EnergyHeadConfig alongside checkpoint
    std::fs::create_dir_all(&args.output_dir)?;
    let config_path = args.output_dir.join("energy_head_config.json");
    let config_json = serde_json::to_string_pretty(&head_config)?;
    std::fs::write(&config_path, &config_json)?;
    tracing::info!(
        path = %config_path.display(),
        "Saved EnergyHeadConfig"
    );

    println!();
    println!("══════════════════════════════════════════");
    println!(" EBM Training Complete");
    println!("──────────────────────────────────────────");
    println!(" Steps:              {}", args.steps);
    println!(" Learning rate:      {}", args.lr);
    println!(" Batch size:         {}", args.batch_size);
    println!(" K negatives:        {}", args.k_negatives);
    println!(" Records:            {}", sampler.num_records());
    println!(" Eligible theorems:  {}", sampler.num_eligible_theorems());
    println!(" Cached embeddings:  {}", cache.len());
    println!(" Checkpoint dir:     {}", args.output_dir.display());
    println!("══════════════════════════════════════════");

    Ok(())
}
