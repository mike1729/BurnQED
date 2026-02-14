//! Proof search pipeline, evaluation, comparison, and EBM training utilities.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use indicatif::{ProgressBar, ProgressStyle};

use ebm::{
    ContrastiveSampler, EBMScorer, EBMTrainingConfig, EBMValueFn, EmbeddingCache, EnergyHeadConfig,
};
use lean_repl::LeanPool;
use policy::{InferenceHandle, SglangClient, SglangConfig};
use search::{InferencePolicyProvider, SearchConfig, SearchEngine};
use trajectory::{TheoremIndex, TrajectoryReader, TrajectoryWriter};

use crate::config::{build_lean_pool_config, load_search_toml};
use crate::results::{median, BudgetResult, IterationResult, TheoremResult};

/// Arguments for the `search` subcommand.
#[derive(Debug)]
pub struct SearchArgs {
    /// Path to the search config TOML file.
    pub config: PathBuf,
    /// URL of the SGLang inference server.
    pub server_url: String,
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
    /// Resume from a partial trajectory file — skip already-searched theorems.
    pub resume_from: Option<PathBuf>,
    /// Override sampling temperature for tactic generation.
    pub temperature: Option<f64>,
    /// Override maximum tokens per generated tactic.
    pub max_tactic_tokens: Option<usize>,
    /// Override number of candidate tactics per expansion.
    pub num_candidates: Option<usize>,
    /// Number of theorems to search in parallel (default: 1 = sequential).
    pub concurrency: usize,
    /// Maximum number of theorems to search (truncates the index).
    pub max_theorems: Option<usize>,
    /// Lean modules to import (e.g., `["Init", "Mathlib"]`).
    pub imports: Option<Vec<String>>,
}

/// Arguments for the `summary` subcommand (formerly `eval`).
#[derive(Debug)]
pub struct SummaryArgs {
    /// Path to the trajectory Parquet file.
    pub input: PathBuf,
}

/// Arguments for the `eval` subcommand.
#[derive(Debug)]
pub struct EvalArgs {
    /// Path to the search config TOML file.
    pub config: PathBuf,
    /// URL of the SGLang inference server.
    pub server_url: String,
    /// Path to EBM checkpoint directory for value-guided search.
    pub ebm_path: Option<PathBuf>,
    /// Path to the theorem index JSON file.
    pub theorems: PathBuf,
    /// Node budgets to evaluate at (sorted ascending).
    pub budgets: Vec<u32>,
    /// Number of attempts per theorem per budget (best-of-N).
    pub pass_n: u32,
    /// Path to write JSON evaluation results.
    pub output: Option<PathBuf>,
    /// Optional CLI override for number of Lean workers.
    pub num_workers: Option<usize>,
    /// Number of theorems to search in parallel (default: 1 = sequential).
    pub concurrency: usize,
    /// Maximum number of theorems to evaluate (truncates the index).
    pub max_theorems: Option<usize>,
    /// Override maximum tokens per generated tactic.
    pub max_tactic_tokens: Option<usize>,
    /// Override number of candidate tactics per expansion.
    pub num_candidates: Option<usize>,
    /// Lean modules to import (e.g., `["Init", "Mathlib"]`).
    pub imports: Option<Vec<String>>,
}

/// Arguments for the `compare` subcommand.
#[derive(Debug)]
pub struct CompareArgs {
    /// Paths to evaluation result JSON files.
    pub results: Vec<PathBuf>,
}

/// Arguments for the `train-ebm` subcommand.
#[derive(Debug)]
pub struct TrainEbmArgs {
    /// Path(s) to trajectory Parquet files.
    pub trajectories: Vec<PathBuf>,
    /// Directory for saving checkpoints.
    pub output_dir: PathBuf,
    /// URL of the SGLang inference server (for encoding).
    pub server_url: String,
    /// Hidden size of the LLM (embedding dimension). Default: 4096 for DeepSeek-Prover-V2-7B.
    pub hidden_size: usize,
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

/// Backend for EBM inference (no autodiff).
type InferenceBackend = NdArray<f32>;
/// Backend for EBM training (with autodiff for gradient computation).
type TrainingBackend = Autodiff<NdArray<f32>>;

/// Loaded policy model and optional EBM value function.
struct LoadedPolicy {
    pool: Arc<LeanPool>,
    policy: InferencePolicyProvider,
    value_fn: Option<EBMValueFn>,
    /// Display string for the inference backend (model path or server URL).
    inference_label: String,
}

/// Load Lean pool, SGLang policy, and optional EBM scorer.
///
/// This is the shared setup used by both `run_search` and `run_eval`.
async fn load_policy_and_ebm(
    config_path: &Path,
    server_url: &str,
    ebm_path: Option<&Path>,
    num_workers: Option<usize>,
    temperature: Option<f64>,
    max_tactic_tokens: Option<usize>,
    imports: Option<&[String]>,
) -> anyhow::Result<(SearchConfig, LoadedPolicy)> {
    // 1. Load config
    let toml = load_search_toml(config_path)?;

    // 2. Build Lean pool
    let lean_config = build_lean_pool_config(&toml.lean_pool, num_workers, imports)?;
    tracing::info!(
        num_workers = lean_config.num_workers,
        "Starting Lean worker pool"
    );
    let pool = Arc::new(LeanPool::new(lean_config).await?);

    // 3. Connect to SGLang server
    tracing::info!(url = server_url, "Connecting to SGLang inference server");
    let config = SglangConfig {
        server_url: server_url.to_string(),
        temperature: temperature.unwrap_or(0.6),
        top_p: 0.95,
        max_tactic_tokens: max_tactic_tokens.unwrap_or(48),
        hidden_size: 4096,
    };
    let client = SglangClient::new(config).await?;
    let inference_handle = InferenceHandle::new(client);

    let policy = InferencePolicyProvider::new(inference_handle.clone());

    // 4. Optional EBM scorer
    let value_fn = load_ebm_scorer(ebm_path, &inference_handle)?;

    Ok((
        toml.search,
        LoadedPolicy {
            pool,
            policy,
            value_fn,
            inference_label: server_url.to_string(),
        },
    ))
}

/// Load an optional EBM scorer using the provided inference handle for encoding.
fn load_ebm_scorer(
    ebm_path: Option<&Path>,
    inference_handle: &InferenceHandle,
) -> anyhow::Result<Option<EBMValueFn>> {
    let ebm_dir = match ebm_path {
        Some(dir) => dir,
        None => return Ok(None),
    };

    tracing::info!(path = %ebm_dir.display(), "Loading EBM scorer");

    // Load EnergyHeadConfig from JSON alongside checkpoint
    let cfg_path = ebm_dir.join("energy_head_config.json");
    let config_json = std::fs::read_to_string(&cfg_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read EnergyHeadConfig from {}: {e}",
            cfg_path.display()
        )
    })?;
    let head_config: EnergyHeadConfig = serde_json::from_str(&config_json)
        .map_err(|e| anyhow::anyhow!("Failed to parse EnergyHeadConfig: {e}"))?;

    // Create encode_fn closure using the inference handle + embedding cache.
    // The cache avoids redundant forward passes on repeated proof states.
    let encode_handle = inference_handle.clone();
    let embedding_cache: Arc<std::sync::Mutex<HashMap<String, Vec<f32>>>> =
        Arc::new(std::sync::Mutex::new(HashMap::new()));
    let encode_fn: Box<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync> =
        Box::new(move |state: &str| {
            // Fast path: check cache first
            {
                let cache = embedding_cache
                    .lock()
                    .map_err(|e| anyhow::anyhow!("Embedding cache lock poisoned: {e}"))?;
                if let Some(cached) = cache.get(state) {
                    return Ok(cached.clone());
                }
            }
            // Cache miss: compute embedding via inference handle
            let embedding = encode_handle.encode_blocking(state)?;
            let data = embedding.data;
            // Insert into cache
            embedding_cache
                .lock()
                .map_err(|e| anyhow::anyhow!("Embedding cache lock poisoned: {e}"))?
                .insert(state.to_owned(), data.clone());
            Ok(data)
        });

    // Load EBM scorer from checkpoint
    let checkpoint_path = ebm_dir.join("final");
    let device = Default::default();
    let scorer =
        EBMScorer::<InferenceBackend>::load(&checkpoint_path, &head_config, encode_fn, device)?;
    let value_fn = EBMValueFn::new(scorer);

    tracing::info!("EBM scorer loaded successfully");
    Ok(Some(value_fn))
}

/// Outcome of a single theorem search task, returned from spawned tasks.
struct SearchOutcome {
    name: String,
    result: Result<trajectory::SearchResult, search::SearchError>,
}

/// Run proof search over a batch of theorems and write trajectory data to Parquet.
///
/// When `concurrency > 1`, multiple theorems are searched in parallel using a
/// `JoinSet` bounded by a semaphore. Results flow back to the main loop for
/// single-threaded `TrajectoryWriter` processing.
pub async fn run_search(args: SearchArgs) -> anyhow::Result<()> {
    let start = Instant::now();
    let concurrency = args.concurrency.max(1);

    // EBM-active overrides: more candidates + higher temperature for diversity
    let temperature = args.temperature.or_else(|| {
        args.ebm_path.as_ref().map(|_| {
            tracing::info!("EBM active — defaulting temperature to 1.0 for candidate diversity");
            1.0
        })
    });

    let (mut search_config, loaded) = load_policy_and_ebm(
        &args.config,
        &args.server_url,
        args.ebm_path.as_deref(),
        args.num_workers,
        temperature,
        args.max_tactic_tokens,
        args.imports.as_deref(),
    )
    .await?;

    // Apply CLI override for num_candidates, or EBM default
    if let Some(n) = args.num_candidates {
        search_config.num_candidates = n;
    } else if args.ebm_path.is_some() {
        tracing::info!("EBM active — defaulting num_candidates to 8");
        search_config.num_candidates = 8;
    }

    // Load theorem index
    let mut index = TheoremIndex::from_json(&args.theorems)?;
    if let Some(max) = args.max_theorems {
        if max < index.len() {
            tracing::info!(total = index.len(), max, "Truncating theorem index");
            index.theorems.truncate(max);
        }
    }
    tracing::info!(count = index.len(), "Loaded theorems");

    // Dry-run: verify setup and exit early
    if args.dry_run {
        println!("Dry run — setup verified successfully");
        println!("  Inference: {}", loaded.inference_label);
        println!("  Theorems: {} loaded", index.len());
        println!("  Workers: {}", loaded.pool.num_workers());
        println!(
            "  EBM: {}",
            if args.ebm_path.is_some() {
                "loaded"
            } else {
                "none"
            }
        );
        if let Some(temp) = args.temperature {
            println!("  Temperature: {temp}");
        }
        println!("  Concurrency: {concurrency}");
        loaded.pool.shutdown().await;
        return Ok(());
    }

    // Filter theorems for resume
    let done_names: HashSet<String> = if let Some(ref resume_path) = args.resume_from {
        let names = TrajectoryReader::read_theorem_names(resume_path)?;
        tracing::info!(
            done = names.len(),
            remaining = index.len() - names.len(),
            "Resuming from partial trajectory"
        );
        names
    } else {
        HashSet::new()
    };

    let theorems_to_search: Vec<_> = index
        .theorems
        .iter()
        .filter(|t| !done_names.contains(&t.name))
        .collect();

    // Run search with progress bar
    let engine = SearchEngine::new(search_config);
    let mut writer = TrajectoryWriter::new(args.output.clone());
    let mut proved_count: u32 = 0;
    let mut failed_count: u32 = 0;
    let mut error_count: u32 = 0;
    let total = theorems_to_search.len() as u32;

    // Aggregate stats
    let mut total_nodes: u64 = 0;
    let mut total_lean_ms: u64 = 0;
    let mut total_gen_ms: u64 = 0;
    let mut searched_count: u32 = 0;
    let mut all_candidate_counts: Vec<usize> = Vec::new();

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .expect("valid progress bar template")
            .progress_chars("=> "),
    );

    // CTRL-C via AtomicBool — shared across spawned tasks
    let interrupted = Arc::new(AtomicBool::new(false));
    let sig_flag = interrupted.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        sig_flag.store(true, Ordering::Relaxed);
        tracing::warn!("Interrupted by CTRL-C, finishing in-flight searches");
    });

    // Arc-wrap shared state for concurrent access
    let pool = loaded.pool; // Already Arc<LeanPool>
    let policy = Arc::new(loaded.policy);
    let value_fn: Option<Arc<EBMValueFn>> = loaded.value_fn.map(Arc::new);

    // Spawn phase: submit theorems bounded by semaphore
    let concurrency_sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let mut join_set = tokio::task::JoinSet::new();

    for task in &theorems_to_search {
        if interrupted.load(Ordering::Relaxed) {
            break;
        }

        let permit = concurrency_sem.clone().acquire_owned().await.unwrap();
        if interrupted.load(Ordering::Relaxed) {
            break;
        }

        pb.set_message(task.name.clone());

        let pool = Arc::clone(&pool);
        let policy = Arc::clone(&policy);
        let value_fn = value_fn.clone();
        let engine = engine.clone();
        let name = task.name.clone();
        let statement = task.statement.clone();
        let interrupted = Arc::clone(&interrupted);

        join_set.spawn(async move {
            let _permit = permit; // held for task lifetime
            if interrupted.load(Ordering::Relaxed) {
                return SearchOutcome {
                    name,
                    result: Err(search::SearchError::ProofStart("interrupted".into())),
                };
            }
            let scorer_ref: Option<&dyn search::ValueScorer> =
                value_fn.as_deref().map(|v| v as &dyn search::ValueScorer);
            let result = engine
                .search_one(&pool, &*policy, scorer_ref, &name, &statement)
                .await;
            SearchOutcome { name, result }
        });
    }

    // Collect phase: process results as they complete (single-threaded)
    while let Some(join_result) = join_set.join_next().await {
        match join_result {
            Ok(outcome) => match outcome.result {
                Ok(result) => {
                    searched_count += 1;
                    total_nodes += result.nodes_expanded as u64;
                    total_lean_ms += result.stats.total_lean_time_ms;
                    total_gen_ms += result.stats.total_generate_time_ms;
                    all_candidate_counts.extend_from_slice(&result.stats.candidates_per_expansion);

                    if result.proved {
                        proved_count += 1;
                        tracing::info!(
                            theorem = outcome.name,
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
                    error_count += 1;
                    tracing::warn!(theorem = outcome.name, error = %e, "Search failed, skipping");
                }
            },
            Err(e) => {
                tracing::error!(error = %e, "Search task panicked");
            }
        }

        pb.inc(1);

        // Periodic auto-save
        const AUTOSAVE_INTERVAL: u32 = 50;
        if searched_count % AUTOSAVE_INTERVAL == 0 && searched_count > 0 {
            writer.flush_partial()?;
            tracing::info!(searched = searched_count, "Auto-saved checkpoint");
        }
    }

    pb.finish_with_message("done");

    // Write Parquet
    let record_count = writer.len();
    writer.finish()?;

    // Merge resumed records if applicable
    if let Some(ref resume_path) = args.resume_from {
        if resume_path != &args.output {
            let old_records = TrajectoryReader::read_all(resume_path)?;
            let new_records = TrajectoryReader::read_all(&args.output)?;
            let mut merged_writer = TrajectoryWriter::new(args.output.clone());
            merged_writer.record_all(old_records);
            merged_writer.record_all(new_records);
            merged_writer.finish()?;
            tracing::info!("Merged resumed records into {}", args.output.display());
        }
    }

    // Shutdown pool
    pool.shutdown().await;

    // Print enhanced summary
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    let partial_note = if interrupted.load(Ordering::Relaxed) {
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
    let total_attempted = searched_count + error_count;
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
    println!(" Theorems attempted: {total_attempted}");
    println!(" Proved:             {proved_count} ({prove_pct:.1}%)");
    println!(" Failed:             {failed_count} ({fail_pct:.1}%)");
    if error_count > 0 {
        println!(" Errors:             {error_count} (start_proof failed)");
    }
    println!(" Total nodes:        {total_nodes}");
    println!(" Avg nodes/theorem:  {avg_nodes:.1}");
    println!(" Avg time/theorem:   {avg_time:.1}s");
    println!(" Total Lean time:    {:.1}s", total_lean_ms as f64 / 1000.0);
    println!(" Total gen time:     {:.1}s", total_gen_ms as f64 / 1000.0);
    if !all_candidate_counts.is_empty() {
        let n = all_candidate_counts.len();
        let sum: usize = all_candidate_counts.iter().sum();
        let min = all_candidate_counts.iter().min().unwrap();
        let max = all_candidate_counts.iter().max().unwrap();
        let avg = sum as f64 / n as f64;
        println!(" Candidates/expand:  avg={avg:.1} min={min} max={max} ({n} expansions)");
    }
    println!(" Total wall time:    {elapsed_secs:.1}s");
    println!(" Inference:          {}", loaded.inference_label);
    println!(" Trajectory file:    {}", args.output.display());
    println!("   Records:          {record_count}");
    if args.ebm_path.is_some() {
        println!(" EBM scorer:         active");
    }
    if let Some(temp) = args.temperature {
        println!(" Temperature:        {temp}");
    }
    if concurrency > 1 {
        println!(" Concurrency:        {concurrency}");
    }
    if args.resume_from.is_some() {
        println!(" Resumed from:       {} theorems already done", done_names.len());
    }
    println!("══════════════════════════════════════════");

    Ok(())
}

/// Print statistics from a trajectory Parquet file.
pub fn run_summary(args: SummaryArgs) -> anyhow::Result<()> {
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

/// Outcome of a single theorem eval task (best-of-N), returned from spawned tasks.
struct EvalOutcome {
    name: String,
    best: TheoremResult,
    times: Vec<f64>,
    total_nodes: f64,
}

/// Evaluate a model at multiple search budgets, printing a formatted table
/// and optionally writing JSON results.
///
/// Budgets are processed sequentially (cumulative_solved_set accumulates across
/// budgets). Within each budget, theorem searches run in parallel bounded by
/// the `concurrency` semaphore.
pub async fn run_eval(args: EvalArgs) -> anyhow::Result<()> {
    let concurrency = args.concurrency.max(1);

    // EBM-active overrides: more candidates + higher temperature for diversity
    let temperature = if args.ebm_path.is_some() {
        tracing::info!("EBM active — defaulting temperature to 1.0 for candidate diversity");
        Some(1.0)
    } else {
        None
    };

    let (mut base_config, loaded) = load_policy_and_ebm(
        &args.config,
        &args.server_url,
        args.ebm_path.as_deref(),
        args.num_workers,
        temperature,
        args.max_tactic_tokens,
        args.imports.as_deref(),
    )
    .await?;

    // Apply CLI override for num_candidates, or EBM default
    if let Some(n) = args.num_candidates {
        base_config.num_candidates = n;
    } else if args.ebm_path.is_some() {
        tracing::info!("EBM active — defaulting num_candidates to 8");
        base_config.num_candidates = 8;
    }

    let mut index = TheoremIndex::from_json(&args.theorems)?;
    if let Some(max) = args.max_theorems {
        if max < index.len() {
            tracing::info!(total = index.len(), max, "Truncating theorem index");
            index.theorems.truncate(max);
        }
    }
    tracing::info!(count = index.len(), "Loaded theorems");

    let mut budgets = args.budgets;
    budgets.sort();
    budgets.dedup();

    let total = index.len() as u32;
    let mut budget_results = Vec::new();
    let mut cumulative_solved_set: HashSet<String> = HashSet::new();

    // Arc-wrap shared state for concurrent access
    let pool = loaded.pool; // Already Arc<LeanPool>
    let policy = Arc::new(loaded.policy);
    let value_fn: Option<Arc<EBMValueFn>> = loaded.value_fn.map(Arc::new);
    let concurrency_sem = Arc::new(tokio::sync::Semaphore::new(concurrency));

    for &budget in &budgets {
        tracing::info!(budget, "Evaluating at budget");

        let mut search_config = base_config.clone();
        search_config.max_nodes = budget;
        let engine = SearchEngine::new(search_config);

        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(&format!(
                    "{{spinner:.green}} [budget={budget}] [{{bar:30.cyan/blue}}] {{pos}}/{{len}} {{msg}}"
                ))
                .expect("valid progress bar template")
                .progress_chars("=> "),
        );

        // Spawn phase: submit all theorems for this budget
        let mut join_set = tokio::task::JoinSet::new();
        let pass_n = args.pass_n;

        for task in &index.theorems {
            let permit = concurrency_sem.clone().acquire_owned().await.unwrap();

            let pool = Arc::clone(&pool);
            let policy = Arc::clone(&policy);
            let value_fn = value_fn.clone();
            let engine = engine.clone();
            let name = task.name.clone();
            let statement = task.statement.clone();

            join_set.spawn(async move {
                let _permit = permit;
                let mut best: Option<TheoremResult> = None;
                let mut times = Vec::new();
                let mut total_nodes = 0.0;

                for _ in 0..pass_n {
                    let scorer_ref: Option<&dyn search::ValueScorer> =
                        value_fn.as_deref().map(|v| v as &dyn search::ValueScorer);
                    match engine
                        .search_one(&pool, &*policy, scorer_ref, &name, &statement)
                        .await
                    {
                        Ok(result) => {
                            let time_secs = result.wall_time_ms as f64 / 1000.0;
                            times.push(time_secs);
                            total_nodes += result.nodes_expanded as f64;
                            let tr = TheoremResult {
                                name: name.clone(),
                                proved: result.proved,
                                nodes_used: result.nodes_expanded,
                                time_secs,
                            };
                            best = Some(match best {
                                None => tr,
                                Some(prev) if !prev.proved && tr.proved => tr,
                                Some(prev) => prev,
                            });
                        }
                        Err(e) => {
                            tracing::warn!(theorem = name, error = %e, "Eval search failed");
                            if best.is_none() {
                                best = Some(TheoremResult {
                                    name: name.clone(),
                                    proved: false,
                                    nodes_used: 0,
                                    time_secs: 0.0,
                                });
                            }
                        }
                    }
                }

                EvalOutcome {
                    name,
                    best: best.unwrap_or_else(|| TheoremResult {
                        name: String::new(),
                        proved: false,
                        nodes_used: 0,
                        time_secs: 0.0,
                    }),
                    times,
                    total_nodes,
                }
            });
        }

        // Collect phase: aggregate results
        let mut per_theorem = Vec::new();
        let mut solved = 0u32;
        let mut all_times: Vec<f64> = Vec::new();
        let mut total_nodes_sum: f64 = 0.0;

        while let Some(join_result) = join_set.join_next().await {
            match join_result {
                Ok(outcome) => {
                    if outcome.best.proved {
                        solved += 1;
                        cumulative_solved_set.insert(outcome.name);
                    }
                    all_times.extend(outcome.times);
                    total_nodes_sum += outcome.total_nodes;
                    per_theorem.push(outcome.best);
                }
                Err(e) => {
                    tracing::error!(error = %e, "Eval task panicked");
                }
            }
            pb.inc(1);
        }

        pb.finish_and_clear();

        let rate = if total > 0 {
            solved as f64 / total as f64
        } else {
            0.0
        };
        let n_attempts = all_times.len().max(1) as f64;
        let avg_nodes = total_nodes_sum / n_attempts;
        let avg_time_secs = all_times.iter().sum::<f64>() / n_attempts;
        let median_time_secs = median(&mut all_times);

        budget_results.push(BudgetResult {
            budget,
            solved,
            total,
            rate,
            avg_nodes,
            avg_time_secs,
            median_time_secs,
            per_theorem,
        });
    }

    let cumulative_solved = cumulative_solved_set.len() as u32;
    let cumulative_rate = if total > 0 {
        cumulative_solved as f64 / total as f64
    } else {
        0.0
    };

    // Print formatted table
    println!();
    println!("┌──────────┬───────────┬──────────┬───────────┬───────────┐");
    println!("│ Budget   │ Solved    │ Rate     │ Avg Nodes │ Avg Time  │");
    println!("├──────────┼───────────┼──────────┼───────────┼───────────┤");
    for br in &budget_results {
        println!(
            "│ {:<8} │ {:>4}/{:<4} │ {:>5.1}%   │ {:>9.1} │ {:>7.1}s  │",
            br.budget, br.solved, br.total, br.rate * 100.0, br.avg_nodes, br.avg_time_secs
        );
    }
    println!("└──────────┴───────────┴──────────┴───────────┴───────────┘");
    println!(
        "Cumulative (any budget): {}/{} ({:.1}%)",
        cumulative_solved,
        total,
        cumulative_rate * 100.0
    );

    // Shutdown pool
    pool.shutdown().await;

    // Build and write IterationResult
    let result = IterationResult {
        iteration: None,
        timestamp: chrono::Utc::now().to_rfc3339(),
        llm_path: loaded.inference_label.clone(),
        ebm_path: args.ebm_path.map(|p| p.display().to_string()),
        benchmark: args.theorems.display().to_string(),
        total_theorems: total,
        budget_results,
        cumulative_solved,
        cumulative_rate,
    };

    let output_path = args
        .output
        .unwrap_or_else(|| PathBuf::from("eval_results/eval.json"));
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(&output_path, &json)?;
    println!("Results written to {}", output_path.display());

    Ok(())
}

/// Compare evaluation results across multiple iterations, printing a delta table.
pub fn run_compare(args: CompareArgs) -> anyhow::Result<()> {
    let mut results: Vec<IterationResult> = Vec::new();
    for path in &args.results {
        let json = std::fs::read_to_string(path).map_err(|e| {
            anyhow::anyhow!("Failed to read {}: {e}", path.display())
        })?;
        let result: IterationResult = serde_json::from_str(&json).map_err(|e| {
            anyhow::anyhow!("Failed to parse {}: {e}", path.display())
        })?;
        results.push(result);
    }

    if results.is_empty() {
        println!("No results to compare.");
        return Ok(());
    }

    // Find common budgets across all results
    let common_budgets: Vec<u32> = if let Some(first) = results.first() {
        first
            .budget_results
            .iter()
            .map(|br| br.budget)
            .filter(|b| {
                results
                    .iter()
                    .all(|r| r.budget_results.iter().any(|br| br.budget == *b))
            })
            .collect()
    } else {
        vec![]
    };

    // Print header
    print!("{:<12}", "Iteration");
    for b in &common_budgets {
        print!("│ Budget {:<5}", b);
    }
    println!("│ Cumulative");

    let sep_width = 12 + common_budgets.len() * 14 + 14;
    println!("{}", "─".repeat(sep_width));

    // Print each result row
    for (i, r) in results.iter().enumerate() {
        let label = r
            .iteration
            .map(|n| format!("{n}"))
            .unwrap_or_else(|| format!("{i}"));
        print!("{:<12}", label);
        for b in &common_budgets {
            if let Some(br) = r.budget_results.iter().find(|br| br.budget == *b) {
                print!("│ {:>5.1}%      ", br.rate * 100.0);
            } else {
                print!("│ {:>12}", "N/A");
            }
        }
        println!("│ {:>5.1}%", r.cumulative_rate * 100.0);
    }

    // Print delta row if there are exactly 2 results
    if results.len() == 2 {
        println!("{}", "─".repeat(sep_width));
        let (r0, r1) = (&results[0], &results[1]);
        let l0 = r0
            .iteration
            .map(|n| format!("{n}"))
            .unwrap_or_else(|| "0".to_string());
        let l1 = r1
            .iteration
            .map(|n| format!("{n}"))
            .unwrap_or_else(|| "1".to_string());
        print!("Δ ({l0}→{l1})    ");
        for b in &common_budgets {
            let rate0 = r0
                .budget_results
                .iter()
                .find(|br| br.budget == *b)
                .map(|br| br.rate)
                .unwrap_or(0.0);
            let rate1 = r1
                .budget_results
                .iter()
                .find(|br| br.budget == *b)
                .map(|br| br.rate)
                .unwrap_or(0.0);
            let delta = (rate1 - rate0) * 100.0;
            print!("│ {:>+5.1}%      ", delta);
        }
        let cum_delta = (r1.cumulative_rate - r0.cumulative_rate) * 100.0;
        println!("│ {:>+5.1}%", cum_delta);
    }

    Ok(())
}

/// Train the Energy-Based Model from trajectory data.
pub fn run_train_ebm(args: TrainEbmArgs) -> anyhow::Result<()> {
    let hidden_size = args.hidden_size;

    tracing::info!(
        trajectories = args.trajectories.len(),
        steps = args.steps,
        lr = args.lr,
        batch_size = args.batch_size,
        k_negatives = args.k_negatives,
        hidden_size,
        "Starting EBM training"
    );

    // 1. Create or resume energy head
    let head_config = EnergyHeadConfig::new(hidden_size);
    let device: <TrainingBackend as burn::prelude::Backend>::Device = Default::default();
    let model = if let Some(ref resume_path) = args.resume_from {
        tracing::info!(path = %resume_path.display(), "Resuming from checkpoint");
        let checkpoint_path = resume_path.join("final");
        ebm::resume_from_checkpoint::<TrainingBackend>(&checkpoint_path, &head_config, &device)?
    } else {
        head_config.init::<TrainingBackend>(&device)
    };

    // 2. Load trajectory data
    tracing::info!("Loading trajectory data from {} file(s)", args.trajectories.len());
    let sampler = match ContrastiveSampler::from_parquet(&args.trajectories, args.k_negatives) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "Cannot train EBM — not enough contrastive data");
            println!();
            println!("══════════════════════════════════════════");
            println!(" EBM Training Skipped");
            println!("──────────────────────────────────────────");
            println!(" Reason: {e}");
            println!(" Hint:   Need proved theorems with dead-end");
            println!("         branches to create contrastive pairs.");
            println!("══════════════════════════════════════════");
            return Ok(());
        }
    };
    tracing::info!(
        records = sampler.num_records(),
        eligible_theorems = sampler.num_eligible_theorems(),
        "Trajectory data loaded"
    );

    // 3. Build lazy embedding cache (warm from disk or start empty)
    let cache = if let Some(ref cache_path) = args.embeddings_cache {
        tracing::info!(path = %cache_path.display(), "Loading embedding cache from disk (warm start)");
        EmbeddingCache::load(cache_path)?
    } else {
        EmbeddingCache::new(hidden_size)
    };

    // Connect to SGLang for on-demand encoding of cache misses
    let handle = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            let config = SglangConfig {
                server_url: args.server_url.clone(),
                temperature: 0.0,
                top_p: 1.0,
                max_tactic_tokens: 1,
                hidden_size,
            };
            let client = SglangClient::new(config).await?;
            Ok::<_, anyhow::Error>(InferenceHandle::new(client))
        })
    })?;

    tracing::info!(
        warm_entries = cache.len(),
        "Lazy embedding cache — embeddings computed on demand during training"
    );

    // 4. Create lazy encode_fn: check cache → on miss call SGLang → insert → return
    let cache = RefCell::new(cache);
    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        // Fast path: cache hit
        if let Some(emb) = cache.borrow().get(state) {
            return Ok(emb.to_vec());
        }
        // Cache miss: encode via SGLang
        let embedding = handle.encode_blocking(state)?;
        let data = embedding.data;
        cache.borrow_mut().insert(state.to_owned(), data.clone());
        Ok(data)
    };

    // 5. Build training config
    let output_dir_str = args.output_dir.to_string_lossy().to_string();
    let training_config = EBMTrainingConfig::new()
        .with_lr(args.lr)
        .with_total_steps(args.steps)
        .with_batch_size(args.batch_size)
        .with_k_negatives(args.k_negatives)
        .with_checkpoint_dir(output_dir_str.clone());

    // 6. Train
    let _trained = ebm::train(&training_config, model, &encode_fn, &sampler, &device)?;

    // 7. Optionally save embedding cache for reuse
    let cache = cache.into_inner();
    if let Some(ref save_path) = args.save_embeddings {
        cache.save(save_path)?;
    }

    tracing::info!(
        cached = cache.len(),
        dim = cache.dim(),
        "Lazy embedding cache stats after training"
    );

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
