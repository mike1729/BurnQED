//! Proof search pipeline, evaluation, comparison, and EBM training utilities.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
#[cfg(feature = "cuda")]
use burn::backend::CudaJit;
use indicatif::{ProgressBar, ProgressStyle};

use ebm::{
    ContrastiveSampler, EBMScorer, EBMTrainingConfig, EBMValueFn, EmbeddingCache, EnergyHeadConfig,
    ProofStateRecord,
};
use lean_repl::{LeanPool, TacticResult};
use policy::{InferenceHandle, SglangClient, SglangConfig};
use search::{
    CachedPolicy, CachingEncoder, GlobalBatcher, GlobalEncodeBatcher, InferencePolicyProvider,
    SearchConfig, SearchEngine,
};
use trajectory::{TheoremIndex, TrajectoryLabel, TrajectoryReader, TrajectoryRecord, TrajectoryWriter};

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
    /// Output as JSON instead of human-readable text.
    pub json: bool,
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
    /// Path to tactic pairs JSONL file for augmenting training data.
    pub tactic_pairs: Option<PathBuf>,
    /// Number of concurrent encode requests during embedding precomputation.
    pub encode_concurrency: usize,
    /// Batch size for batched encode requests (0 = use individual concurrent requests).
    pub encode_batch_size: usize,
}

/// Backend for EBM inference (no autodiff).
type InferenceBackend = NdArray<f32>;
/// Backend for EBM training (with autodiff for gradient computation).
#[cfg(not(feature = "cuda"))]
type TrainingBackend = Autodiff<NdArray<f32>>;
#[cfg(feature = "cuda")]
type TrainingBackend = Autodiff<CudaJit>;

/// Loaded policy model and optional EBM value function.
struct LoadedPolicy {
    pool: Arc<LeanPool>,
    policy: CachedPolicy<GlobalBatcher>,
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
    concurrency: usize,
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

    let raw_policy = InferencePolicyProvider::new(inference_handle.clone());
    let exp = toml.search.batch_expansion_size;
    let n_cands = toml.search.num_candidates;
    let max_batch = exp * concurrency.max(1); // coalesce across concurrent searches
    let batcher = GlobalBatcher::new(
        Arc::new(raw_policy),
        max_batch,
        Duration::from_millis(5),        // linger
    );
    let policy = CachedPolicy::new(batcher, 10_000);

    // 4. Optional EBM scorer
    let encode_batch_limit = exp * n_cands; // e.g. 8×8=64 or 4×8=32
    let value_fn = load_ebm_scorer(ebm_path, &inference_handle, encode_batch_limit)?;

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
///
/// Sets up a `CachingEncoder` → `GlobalEncodeBatcher` pipeline so concurrent
/// search tasks coalesce their encode requests into GPU-efficient batches.
/// The `EBMValueFn` encodes **outside** its internal Mutex, so only the fast
/// MLP forward pass holds the lock.
fn load_ebm_scorer(
    ebm_path: Option<&Path>,
    inference_handle: &InferenceHandle,
    encode_batch_limit: usize,
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

    let hidden_size = head_config.d_encoder;

    // CachingEncoder: handles embedding cache + HTTP encode calls
    let caching_encoder = CachingEncoder::new(inference_handle.clone(), hidden_size);

    // GlobalEncodeBatcher: coalesces concurrent encode requests
    let encode_batcher = Arc::new(GlobalEncodeBatcher::new(
        Arc::new(caching_encoder),
        encode_batch_limit,              // derived from batch_expansion_size × num_candidates
        Duration::from_millis(5),        // linger
    ));

    // Batch encode closure routed through the batcher (used by EBMValueFn)
    let batcher_ref = encode_batcher.clone();
    let batch_encode_fn: Box<dyn Fn(&[&str]) -> anyhow::Result<Vec<Vec<f32>>> + Send + Sync> =
        Box::new(move |states: &[&str]| batcher_ref.encode_batch_blocking(states));

    // Single encode closure routed through the batcher (batch of 1, used by EBMScorer::score_state)
    let batcher_ref2 = encode_batcher.clone();
    let encode_fn: Box<dyn Fn(&str) -> anyhow::Result<Vec<f32>> + Send + Sync> =
        Box::new(move |state: &str| {
            let results = batcher_ref2.encode_batch_blocking(&[state])?;
            results
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("encode_batch returned empty result"))
        });

    // Load EBM scorer from checkpoint
    let checkpoint_path = ebm_dir.join("final");
    let device = Default::default();
    let scorer =
        EBMScorer::<InferenceBackend>::load(&checkpoint_path, &head_config, encode_fn, device)?;
    let value_fn = EBMValueFn::with_batch_encode(scorer, batch_encode_fn);

    tracing::info!("EBM scorer loaded successfully");
    Ok(Some(value_fn))
}

// ---------------------------------------------------------------------------
// Generate-negatives helpers
// ---------------------------------------------------------------------------

/// Arguments for the `generate-negatives` subcommand.
#[derive(Debug)]
pub struct GenerateNegativesArgs {
    /// Path to the search config TOML file.
    pub config: PathBuf,
    /// URL of the SGLang inference server.
    pub server_url: String,
    /// Path to the tactic pairs JSONL file.
    pub tactic_pairs: PathBuf,
    /// Path for the output Parquet file.
    pub output: PathBuf,
    /// Maximum number of theorems to process.
    pub num_theorems: Option<usize>,
    /// Minimum number of proof steps per theorem (filters short proofs).
    pub min_steps: Option<usize>,
    /// Number of LLM candidates to generate per proof step.
    pub candidates_per_step: usize,
    /// Target number of negatives per theorem before early stop.
    pub target_negatives: usize,
    /// Sampling temperature for generation.
    pub temperature: f64,
    /// Lean modules to import (e.g., `["Init", "Mathlib"]`).
    pub imports: Vec<String>,
    /// Number of theorems to process in parallel.
    pub concurrency: usize,
    /// Optional CLI override for number of Lean workers.
    pub num_workers: Option<usize>,
}

/// Per-theorem result from the generate-negatives pipeline.
#[allow(dead_code)]
struct NegGenOutcome {
    theorem_name: String,
    records: Vec<trajectory::TrajectoryRecord>,
    steps_walked: usize,
    total_steps: usize,
    completed: bool,
    positives: usize,
    negatives: usize,
    alternative_proofs: usize,
}

/// Infer the `open` namespace prefix from a fully-qualified theorem name.
///
/// Uses the last `.` separator: `"Polynomial.natDegree_cyclotomic'"` → `Some("Polynomial")`.
/// Returns `None` for simple names without a dot.
/// Build an `open ... in` prefix that opens all namespace segments of a theorem name.
///
/// For `CategoryTheory.ShortComplex.cycles_ext_iff` returns:
///   `"open CategoryTheory CategoryTheory.ShortComplex in "`
///
/// Opening each prefix individually ensures short names from any level
/// of the hierarchy resolve (e.g. `cancel_mono` from `CategoryTheory`).
fn infer_open_prefix(theorem_name: &str) -> Option<String> {
    let parts: Vec<&str> = theorem_name.split('.').collect();
    if parts.len() <= 1 {
        return None;
    }
    // Build all prefixes: ["A", "A.B", "A.B.C"] for "A.B.C.thm"
    let mut prefixes = Vec::new();
    for i in 1..parts.len() {
        prefixes.push(parts[..i].join("."));
    }
    Some(format!("open {} in ", prefixes.join(" ")))
}

/// Normalize a tactic string by collapsing whitespace for comparison.
fn normalize_tactic(tactic: &str) -> String {
    tactic.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Accumulates per-theorem search results for the final summary.
#[derive(Default)]
struct SearchAggregator {
    searched_count: u32,
    proved_count: u32,
    failed_count: u32,
    error_count: u32,
    total_nodes: u64,
    total_lean_ms: u64,
    total_gen_ms: u64,
    total_ebm_ms: u64,
    total_harvest_ms: u64,
    total_probe_lean_ms: u64,
    total_llm_lean_ms: u64,
    total_ebm_calls: u64,
    total_cache_hits: u32,
    total_cache_misses: u32,
    all_candidate_counts: Vec<usize>,
    lean_latencies_us: Vec<u64>,
    gen_latencies_us: Vec<u64>,
    ebm_latencies_us: Vec<u64>,
    /// Ring buffer of recent completion timestamps for moving-average ETA.
    completion_times: VecDeque<Instant>,
}

impl SearchAggregator {
    const ETA_WINDOW: usize = 200;

    /// Record a theorem completion for moving-average ETA.
    fn record_completion(&mut self) {
        self.completion_times.push_back(Instant::now());
        if self.completion_times.len() > Self::ETA_WINDOW {
            self.completion_times.pop_front();
        }
    }

    /// Compute ETA string from recent throughput (moving window).
    fn format_eta(&self, remaining: u32) -> String {
        if self.completion_times.len() < 2 {
            return "?".to_string();
        }
        let first = *self.completion_times.front().unwrap();
        let last = *self.completion_times.back().unwrap();
        let span = last.duration_since(first);
        let intervals = self.completion_times.len() - 1;
        if intervals == 0 || span.is_zero() {
            return "?".to_string();
        }
        let secs_per_thm = span.as_secs_f64() / intervals as f64;
        let eta_secs = (remaining as f64 * secs_per_thm) as u64;
        format_duration_short(eta_secs)
    }

    /// Print a compact progress block every N theorems during long runs.
    fn print_progress(&self, elapsed: Duration, records_written: usize) {
        let secs = elapsed.as_secs_f64();
        let searched = self.searched_count;
        let prove_pct = if searched > 0 {
            self.proved_count as f64 / searched as f64 * 100.0
        } else {
            0.0
        };

        let avg_nodes = if searched > 0 {
            self.total_nodes as f64 / searched as f64
        } else {
            0.0
        };

        // Timing breakdown as % of wall time
        let gen_s = self.total_gen_ms as f64 / 1000.0;
        let llm_lean_s = self.total_llm_lean_ms as f64 / 1000.0;
        let probe_lean_s = self.total_probe_lean_ms as f64 / 1000.0;
        let ebm_s = self.total_ebm_ms as f64 / 1000.0;
        let harvest_s = self.total_harvest_ms as f64 / 1000.0;

        fn pct(part: f64, total: f64) -> f64 {
            if total > 0.0 { part / total * 100.0 } else { 0.0 }
        }

        let cache_total = self.total_cache_hits + self.total_cache_misses;
        let cache_hit_pct = if cache_total > 0 {
            self.total_cache_hits as f64 / cache_total as f64 * 100.0
        } else {
            0.0
        };

        let thm_per_min = if secs > 0.0 { searched as f64 / secs * 60.0 } else { 0.0 };
        let traj_per_sec = if secs > 0.0 { records_written as f64 / secs } else { 0.0 };

        // Print via println inside pb.suspend() — must go to stdout
        // so indicatif can manage the output correctly.
        println!(
            "\n── Progress @ {} theorems ({:.0}s) ──\n\
             \x20 Proved {}/{} ({:.1}%), errors {}\n\
             \x20 Avg nodes/thm: {:.1}\n\
             \x20 Time: gen {:.1}% | lean(llm) {:.1}% | lean(probe) {:.1}% | ebm {:.1}% | harvest {:.1}%\n\
             \x20 Cache: {:.1}% hit ({}/{})\n\
             \x20 Throughput: {:.1} thm/min, {:.1} traj/s\n\
             \x20 EBM calls: {}",
            searched,
            secs,
            self.proved_count,
            searched,
            prove_pct,
            self.error_count,
            avg_nodes,
            pct(gen_s, secs),
            pct(llm_lean_s, secs),
            pct(probe_lean_s, secs),
            pct(ebm_s, secs),
            pct(harvest_s, secs),
            cache_hit_pct,
            self.total_cache_hits,
            cache_total,
            thm_per_min,
            traj_per_sec,
            self.total_ebm_calls,
        );

        // Also log structured stats for the log file
        tracing::info!(
            searched,
            proved = self.proved_count,
            errors = self.error_count,
            prove_pct = format!("{:.1}", prove_pct),
            thm_per_min = format!("{:.1}", thm_per_min),
            avg_nodes = format!("{:.1}", avg_nodes),
            gen_pct = format!("{:.1}", pct(gen_s, secs)),
            lean_llm_pct = format!("{:.1}", pct(llm_lean_s, secs)),
            lean_probe_pct = format!("{:.1}", pct(probe_lean_s, secs)),
            ebm_pct = format!("{:.1}", pct(ebm_s, secs)),
            cache_hit_pct = format!("{:.1}", cache_hit_pct),
            ebm_calls = self.total_ebm_calls,
            "progress_stats"
        );
    }
}

/// Format seconds as compact human-readable duration (e.g., "2h30m", "45m12s", "30s").
fn format_duration_short(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m{}s", secs / 60, secs % 60)
    } else {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        format!("{h}h{m:02}m")
    }
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
        concurrency,
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
    let total = theorems_to_search.len() as u32;
    let mut agg = SearchAggregator::default();

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({msg}) {prefix}")
            .expect("valid progress bar template")
            .progress_chars("=> "),
    );
    pb.set_prefix("proved=0");
    pb.set_message("?");
    pb.enable_steady_tick(Duration::from_secs(1));

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

    /// Process one completed search outcome into the aggregator.
    fn collect_search(
        outcome: SearchOutcome,
        agg: &mut SearchAggregator,
        writer: &mut TrajectoryWriter,
        pb: &ProgressBar,
        total: u32,
    ) {
        match outcome.result {
            Ok(result) => {
                agg.searched_count += 1;
                agg.total_nodes += result.nodes_expanded as u64;
                agg.total_lean_ms += result.stats.total_lean_time_ms;
                agg.total_gen_ms += result.stats.total_generate_time_ms;
                agg.total_ebm_ms += result.stats.total_ebm_time_ms;
                agg.total_harvest_ms += result.stats.total_harvest_time_ms;
                agg.total_probe_lean_ms += result.stats.total_probe_lean_time_ms;
                agg.total_llm_lean_ms += result.stats.total_llm_lean_time_ms;
                agg.total_ebm_calls += result.stats.ebm_score_calls as u64;
                agg.total_cache_hits += result.stats.cache_hits;
                agg.total_cache_misses += result.stats.cache_misses;
                agg.all_candidate_counts
                    .extend_from_slice(&result.stats.candidates_per_expansion);
                agg.lean_latencies_us
                    .extend_from_slice(&result.stats.lean_latencies_us);
                agg.gen_latencies_us
                    .extend_from_slice(&result.stats.gen_latencies_us);
                agg.ebm_latencies_us
                    .extend_from_slice(&result.stats.ebm_latencies_us);

                let proved = result.proved;
                let wall_ms = result.wall_time_ms;

                if proved {
                    agg.proved_count += 1;
                    tracing::debug!(
                        theorem = outcome.name,
                        tactics = ?result.proof_tactics,
                        nodes = result.nodes_expanded,
                        time_ms = wall_ms,
                        "Proved"
                    );
                } else {
                    agg.failed_count += 1;
                }

                // Per-theorem structured profile log
                tracing::debug!(
                    theorem = outcome.name,
                    proved,
                    wall_ms,
                    lean_ms = result.stats.total_lean_time_ms,
                    gen_ms = result.stats.total_generate_time_ms,
                    ebm_ms = result.stats.total_ebm_time_ms,
                    harvest_ms = result.stats.total_harvest_time_ms,
                    probe_lean_ms = result.stats.total_probe_lean_time_ms,
                    llm_lean_ms = result.stats.total_llm_lean_time_ms,
                    cache_hits = result.stats.cache_hits,
                    cache_misses = result.stats.cache_misses,
                    nodes = result.nodes_expanded,
                    "theorem_profile"
                );

                let labeled = TrajectoryWriter::from_search_result(&result);
                writer.record_all(labeled);
            }
            Err(e) => {
                agg.error_count += 1;
                tracing::warn!(theorem = outcome.name, error = %e, "Search error, skipping");
            }
        }
        agg.record_completion();
        pb.inc(1);
        if agg.error_count > 0 {
            pb.set_prefix(format!("proved={} err={}", agg.proved_count, agg.error_count));
        } else {
            pb.set_prefix(format!("proved={}", agg.proved_count));
        }
        let done = agg.searched_count + agg.error_count;
        let remaining = total.saturating_sub(done);
        pb.set_message(agg.format_eta(remaining));
    }

    for task in &theorems_to_search {
        if interrupted.load(Ordering::Relaxed) {
            break;
        }

        // Drain completed tasks before waiting for a permit
        while let Some(join_result) = join_set.try_join_next() {
            match join_result {
                Ok(outcome) => {
                    collect_search(outcome, &mut agg, &mut writer, &pb, total);
                    // Periodic auto-save
                    const AUTOSAVE_INTERVAL: u32 = 50;
                    if agg.searched_count % AUTOSAVE_INTERVAL == 0 && agg.searched_count > 0 {
                        writer.flush_partial()?;
                        tracing::info!(searched = agg.searched_count, "Auto-saved checkpoint");
                    }
                    // Periodic progress stats
                    const PROGRESS_INTERVAL: u32 = 500;
                    if agg.searched_count % PROGRESS_INTERVAL == 0 && agg.searched_count > 0 {
                        pb.suspend(|| agg.print_progress(start.elapsed(), writer.len()));
                    }
                }
                Err(e) => tracing::error!(error = %e, "Search task panicked"),
            }
        }

        let permit = concurrency_sem.clone().acquire_owned().await.unwrap();
        if interrupted.load(Ordering::Relaxed) {
            break;
        }

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

    // Final drain: collect remaining in-flight tasks
    while let Some(join_result) = join_set.join_next().await {
        match join_result {
            Ok(outcome) => {
                collect_search(outcome, &mut agg, &mut writer, &pb, total);
                const AUTOSAVE_INTERVAL: u32 = 50;
                if agg.searched_count % AUTOSAVE_INTERVAL == 0 && agg.searched_count > 0 {
                    writer.flush_partial()?;
                    tracing::info!(searched = agg.searched_count, "Auto-saved checkpoint");
                }
                const PROGRESS_INTERVAL: u32 = 500;
                if agg.searched_count % PROGRESS_INTERVAL == 0 && agg.searched_count > 0 {
                    pb.suspend(|| agg.print_progress(start.elapsed(), writer.len()));
                }
            }
            Err(e) => tracing::error!(error = %e, "Search task panicked"),
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

    let searched = agg.searched_count;
    let prove_pct = if searched > 0 {
        agg.proved_count as f64 / searched as f64 * 100.0
    } else {
        0.0
    };
    let fail_pct = if searched > 0 {
        agg.failed_count as f64 / searched as f64 * 100.0
    } else {
        0.0
    };
    let total_attempted = searched + agg.error_count;
    let avg_nodes = if searched > 0 {
        agg.total_nodes as f64 / searched as f64
    } else {
        0.0
    };
    let avg_time = if searched > 0 {
        elapsed_secs / searched as f64
    } else {
        0.0
    };

    // Timing breakdown
    let gen_s = agg.total_gen_ms as f64 / 1000.0;
    let llm_lean_s = agg.total_llm_lean_ms as f64 / 1000.0;
    let probe_lean_s = agg.total_probe_lean_ms as f64 / 1000.0;
    let ebm_s = agg.total_ebm_ms as f64 / 1000.0;
    let harvest_s = agg.total_harvest_ms as f64 / 1000.0;
    let accounted_s = gen_s + llm_lean_s + probe_lean_s + ebm_s + harvest_s;
    let overhead_s = (elapsed_secs - accounted_s).max(0.0);

    fn pct_of(part: f64, total: f64) -> f64 {
        if total > 0.0 { part / total * 100.0 } else { 0.0 }
    }

    // Latency percentiles
    use crate::results::percentiles;
    let (lean_p50, lean_p95, lean_p99) = percentiles(&mut agg.lean_latencies_us);
    let lean_calls = agg.lean_latencies_us.len();
    let (gen_p50, gen_p95, gen_p99) = percentiles(&mut agg.gen_latencies_us);
    let gen_calls = agg.gen_latencies_us.len();
    let (ebm_p50, ebm_p95, ebm_p99) = percentiles(&mut agg.ebm_latencies_us);
    let ebm_calls = agg.ebm_latencies_us.len();

    println!();
    println!("══════════════════════════════════════════");
    println!(" burn-qed Search Results{partial_note}");
    println!("──────────────────────────────────────────");
    println!(
        " Theorems:  {} attempted, {} proved ({:.1}%), {} failed ({:.1}%){}",
        total_attempted,
        agg.proved_count,
        prove_pct,
        agg.failed_count,
        fail_pct,
        if agg.error_count > 0 {
            format!(", {} errors", agg.error_count)
        } else {
            String::new()
        }
    );
    println!("──────────────────────────────────────────");
    println!(" TIMING BREAKDOWN (cumulative)");
    println!("   LLM generation:  {:>7.1}s  ({:>5.1}%)", gen_s, pct_of(gen_s, elapsed_secs));
    println!("   Lean (LLM):      {:>7.1}s  ({:>5.1}%)", llm_lean_s, pct_of(llm_lean_s, elapsed_secs));
    println!("   Lean (probes):   {:>7.1}s  ({:>5.1}%)", probe_lean_s, pct_of(probe_lean_s, elapsed_secs));
    println!("   EBM scoring:     {:>7.1}s  ({:>5.1}%)", ebm_s, pct_of(ebm_s, elapsed_secs));
    println!("   Harvest siblings:{:>7.1}s  ({:>5.1}%)", harvest_s, pct_of(harvest_s, elapsed_secs));
    println!("   Overhead/queue:  {:>7.1}s  ({:>5.1}%)", overhead_s, pct_of(overhead_s, elapsed_secs));
    println!("   Total wall time: {:>7.1}s", elapsed_secs);
    println!("──────────────────────────────────────────");
    println!(" LATENCY PERCENTILES");
    if lean_calls > 0 {
        println!(
            "   Lean/tactic:    p50={:>4}us  p95={:>5}us  p99={:>5}us  ({} calls)",
            lean_p50, lean_p95, lean_p99, lean_calls
        );
    }
    if gen_calls > 0 {
        println!(
            "   LLM gen/batch:  p50={:>4}us  p95={:>5}us  p99={:>5}us  ({} calls)",
            gen_p50, gen_p95, gen_p99, gen_calls
        );
    }
    if ebm_calls > 0 {
        println!(
            "   EBM score:      p50={:>4}us  p95={:>5}us  p99={:>5}us  ({} calls)",
            ebm_p50, ebm_p95, ebm_p99, ebm_calls
        );
    }
    // Cache stats
    let total_cache = agg.total_cache_hits + agg.total_cache_misses;
    if total_cache > 0 {
        let hit_rate = agg.total_cache_hits as f64 / total_cache as f64 * 100.0;
        println!("──────────────────────────────────────────");
        println!(
            " CACHE: {} hits / {} misses ({:.1}% hit rate)",
            agg.total_cache_hits, agg.total_cache_misses, hit_rate
        );
    }
    println!("──────────────────────────────────────────");
    println!(" THROUGHPUT");
    if elapsed_secs > 0.0 && searched > 0 {
        println!("   Theorems/min:    {:.1}", searched as f64 / elapsed_secs * 60.0);
    }
    println!("   Avg nodes/thm:   {avg_nodes:.1}");
    println!("   Avg time/thm:    {avg_time:.1}s");
    if !agg.all_candidate_counts.is_empty() {
        let n = agg.all_candidate_counts.len();
        let sum: usize = agg.all_candidate_counts.iter().sum();
        let min = agg.all_candidate_counts.iter().min().unwrap();
        let max = agg.all_candidate_counts.iter().max().unwrap();
        let avg = sum as f64 / n as f64;
        println!("   Candidates/exp:  avg={avg:.1} min={min} max={max} ({n} expansions)");
    }
    println!("──────────────────────────────────────────");
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

    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
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
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Probe — probe-only tactic search (no LLM)
// ---------------------------------------------------------------------------

/// Arguments for the `probe` subcommand.
#[derive(Debug)]
pub struct ProbeArgs {
    /// Path to the search config TOML file.
    pub config: PathBuf,
    /// Path to the theorem index JSON file.
    pub theorems: PathBuf,
    /// Path for the output JSON file (easy/hard/stats).
    pub output: PathBuf,
    /// Override the number of Lean workers.
    pub num_workers: Option<usize>,
    /// Number of theorems to search in parallel.
    pub concurrency: usize,
    /// Maximum number of theorems to process (truncates the index).
    pub max_theorems: Option<usize>,
    /// Lean modules to import (e.g., `["Init", "Mathlib"]`).
    pub imports: Option<Vec<String>>,
    /// Override max_nodes for probe search (default: 100).
    pub max_nodes: Option<u32>,
    /// Optional path to write hard theorems as TheoremIndex JSON.
    pub hard_theorems: Option<PathBuf>,
}

/// Result entry for an easily-proved theorem.
#[derive(serde::Serialize)]
struct ProbeEasyEntry {
    name: String,
    statement: String,
    depth: u32,
    tactics: Vec<String>,
}

/// Result entry for a hard (unproved by probes) theorem.
#[derive(serde::Serialize)]
struct ProbeHardEntry {
    name: String,
    statement: String,
}

/// Overall probe output structure.
#[derive(serde::Serialize)]
struct ProbeOutput {
    easy: Vec<ProbeEasyEntry>,
    hard: Vec<ProbeHardEntry>,
    stats: ProbeStats,
}

/// Summary statistics for the probe run.
#[derive(serde::Serialize)]
struct ProbeStats {
    total: usize,
    easy: usize,
    hard: usize,
    errors: usize,
    elapsed_secs: f64,
}

/// Outcome of a single theorem probe task, returned from spawned tasks.
struct ProbeOutcome {
    name: String,
    statement: String,
    result: Result<trajectory::SearchResult, search::SearchError>,
}

/// Run probe-only tactic search over a batch of theorems.
///
/// Uses `NullPolicyProvider` so only the engine's built-in probe tactics
/// (simp, ring, omega, etc.) are tried. No SGLang server required.
pub async fn run_probe(args: ProbeArgs) -> anyhow::Result<()> {
    let start = Instant::now();
    let concurrency = args.concurrency.max(1);

    // 1. Load search config from TOML
    let toml = load_search_toml(&args.config)?;
    let mut search_config = toml.search;

    // Override max_nodes for probe search (default: 100)
    if let Some(max) = args.max_nodes {
        search_config.max_nodes = max;
    } else {
        search_config.max_nodes = 100;
    }
    tracing::info!(max_nodes = search_config.max_nodes, "Probe search config");

    // 2. Build Lean pool (no SGLang needed)
    let lean_config = build_lean_pool_config(
        &toml.lean_pool,
        args.num_workers,
        args.imports.as_deref(),
    )?;
    tracing::info!(
        num_workers = lean_config.num_workers,
        "Starting Lean worker pool"
    );
    let pool = Arc::new(LeanPool::new(lean_config).await?);

    // 3. Load theorem index
    let mut index = TheoremIndex::from_json(&args.theorems)?;
    if let Some(max) = args.max_theorems {
        if max < index.len() {
            tracing::info!(total = index.len(), max, "Truncating theorem index");
            index.theorems.truncate(max);
        }
    }
    let total = index.len();
    tracing::info!(count = total, "Loaded theorems for probe search");

    // 4. Create engine with NullPolicyProvider
    let engine = SearchEngine::new(search_config);
    let null_policy: Arc<search::NullPolicyProvider> = Arc::new(search::NullPolicyProvider);

    // 5. Progress bar
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) easy={prefix} {msg}")
            .expect("valid progress bar template")
            .progress_chars("=> "),
    );
    pb.set_prefix("0");
    pb.enable_steady_tick(Duration::from_secs(1));

    // CTRL-C handler
    let interrupted = Arc::new(AtomicBool::new(false));
    let sig_flag = interrupted.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        sig_flag.store(true, Ordering::Relaxed);
        tracing::warn!("Interrupted by CTRL-C, finishing in-flight probes");
    });

    // 6. Concurrent probe search
    let concurrency_sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let mut join_set: tokio::task::JoinSet<ProbeOutcome> = tokio::task::JoinSet::new();

    let mut easy = Vec::new();
    let mut hard = Vec::new();
    let mut error_count = 0usize;
    let mut easy_count = 0u32;

    for task in &index.theorems {
        if interrupted.load(Ordering::Relaxed) {
            break;
        }

        // Drain completed tasks before waiting for a permit
        while let Some(join_result) = join_set.try_join_next() {
            match join_result {
                Ok(outcome) => {
                    match outcome.result {
                        Ok(result) => {
                            if result.proved {
                                easy_count += 1;
                                easy.push(ProbeEasyEntry {
                                    name: outcome.name,
                                    statement: outcome.statement,
                                    depth: result.max_depth_reached,
                                    tactics: result.proof_tactics,
                                });
                            } else {
                                hard.push(ProbeHardEntry {
                                    name: outcome.name,
                                    statement: outcome.statement,
                                });
                            }
                        }
                        Err(e) => {
                            error_count += 1;
                            tracing::debug!(
                                theorem = outcome.name,
                                error = %e,
                                "Probe failed, counting as hard"
                            );
                            hard.push(ProbeHardEntry {
                                name: outcome.name,
                                statement: outcome.statement,
                            });
                        }
                    }
                    pb.inc(1);
                    pb.set_prefix(format!("{easy_count}"));
                }
                Err(e) => {
                    error_count += 1;
                    tracing::error!(error = %e, "Probe task panicked");
                    pb.inc(1);
                }
            }
        }

        let permit = concurrency_sem.clone().acquire_owned().await.unwrap();
        if interrupted.load(Ordering::Relaxed) {
            break;
        }

        pb.set_message(task.name.clone());

        let pool = Arc::clone(&pool);
        let policy = Arc::clone(&null_policy);
        let engine = engine.clone();
        let name = task.name.clone();
        let statement = task.statement.clone();
        let interrupted = Arc::clone(&interrupted);

        join_set.spawn(async move {
            let _permit = permit;
            if interrupted.load(Ordering::Relaxed) {
                return ProbeOutcome {
                    name: name.clone(),
                    statement,
                    result: Err(search::SearchError::ProofStart("interrupted".into())),
                };
            }
            let result = engine
                .search_one(&pool, &*policy, None, &name, &statement)
                .await;
            ProbeOutcome {
                name,
                statement,
                result,
            }
        });
    }

    // Final drain
    while let Some(join_result) = join_set.join_next().await {
        match join_result {
            Ok(outcome) => {
                match outcome.result {
                    Ok(result) => {
                        if result.proved {
                            easy_count += 1;
                            easy.push(ProbeEasyEntry {
                                name: outcome.name,
                                statement: outcome.statement,
                                depth: result.max_depth_reached,
                                tactics: result.proof_tactics,
                            });
                        } else {
                            hard.push(ProbeHardEntry {
                                name: outcome.name,
                                statement: outcome.statement,
                            });
                        }
                    }
                    Err(e) => {
                        error_count += 1;
                        tracing::debug!(
                            theorem = outcome.name,
                            error = %e,
                            "Probe failed, counting as hard"
                        );
                        hard.push(ProbeHardEntry {
                            name: outcome.name,
                            statement: outcome.statement,
                        });
                    }
                }
                pb.inc(1);
                pb.set_prefix(format!("{easy_count}"));
            }
            Err(e) => {
                error_count += 1;
                tracing::error!(error = %e, "Probe task panicked");
                pb.inc(1);
            }
        }
    }

    pb.finish_with_message("done");
    pool.shutdown().await;

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    // 7. Write output JSON
    let output = ProbeOutput {
        stats: ProbeStats {
            total,
            easy: easy.len(),
            hard: hard.len(),
            errors: error_count,
            elapsed_secs,
        },
        easy,
        hard,
    };

    let json = serde_json::to_string_pretty(&output)?;
    std::fs::write(&args.output, &json)?;
    tracing::info!(path = %args.output.display(), "Wrote probe results");

    // 8. Optionally write hard theorems as TheoremIndex JSON
    if let Some(ref hard_path) = args.hard_theorems {
        // Build a TheoremIndex-compatible JSON: {"theorems": [{name, statement}, ...]}
        let hard_tasks: Vec<serde_json::Value> = output
            .hard
            .iter()
            .map(|h| {
                serde_json::json!({
                    "name": h.name,
                    "statement": h.statement,
                })
            })
            .collect();
        let hard_json = serde_json::to_string_pretty(&serde_json::json!({
            "theorems": hard_tasks,
        }))?;
        std::fs::write(hard_path, &hard_json)?;
        tracing::info!(
            path = %hard_path.display(),
            count = hard_tasks.len(),
            "Wrote hard theorems index"
        );
    }

    // 9. Print summary
    let partial_note = if interrupted.load(Ordering::Relaxed) {
        " (Partial — interrupted by CTRL-C)"
    } else {
        ""
    };

    println!();
    println!("══════════════════════════════════════════");
    println!(" burn-qed Probe Results{partial_note}");
    println!("──────────────────────────────────────────");
    println!(
        " Total:  {} theorems ({:.1}s)",
        output.stats.total, elapsed_secs
    );
    println!(
        " Easy:   {} ({:.1}%)",
        output.stats.easy,
        if output.stats.total > 0 {
            output.stats.easy as f64 / output.stats.total as f64 * 100.0
        } else {
            0.0
        }
    );
    println!(
        " Hard:   {} ({:.1}%)",
        output.stats.hard,
        if output.stats.total > 0 {
            output.stats.hard as f64 / output.stats.total as f64 * 100.0
        } else {
            0.0
        }
    );
    if output.stats.errors > 0 {
        println!(" Errors: {}", output.stats.errors);
    }
    println!("──────────────────────────────────────────");
    println!(" Output:  {}", args.output.display());
    if let Some(ref hard_path) = args.hard_theorems {
        println!(" Hard index: {}", hard_path.display());
    }
    println!(
        " Throughput: {:.1} theorems/min",
        if elapsed_secs > 0.0 {
            output.stats.total as f64 / elapsed_secs * 60.0
        } else {
            0.0
        }
    );
    println!("══════════════════════════════════════════");

    Ok(())
}

// ---------------------------------------------------------------------------
// Export proof paths from trajectory Parquet to tactic-pairs JSONL
// ---------------------------------------------------------------------------

/// Arguments for the `export-proof-paths` subcommand.
#[derive(Debug)]
pub struct ExportProofPathsArgs {
    /// Input trajectory Parquet files.
    pub trajectories: Vec<PathBuf>,
    /// Output tactic-pairs JSONL file.
    pub output: PathBuf,
    /// Minimum number of proof steps per theorem (filters short proofs).
    pub min_steps: Option<usize>,
}

/// Extract proof paths from trajectory Parquet files and write tactic-pairs JSONL.
///
/// For each proved theorem, extracts the Positive-labeled records (the proof path),
/// reconstructs the step sequence sorted by depth, and writes each step as a
/// `{"state", "tactic", "theorem", "depth"}` JSON line compatible with
/// `generate-negatives --tactic-pairs`.
///
/// When multiple trajectory files are provided, theorems are deduplicated by name
/// (first occurrence wins — use the file with the best data first).
pub fn run_export_proof_paths(args: ExportProofPathsArgs) -> anyhow::Result<()> {
    use std::io::Write;

    let mut seen_theorems: HashSet<String> = HashSet::new();
    let mut total_theorems = 0usize;
    let mut total_steps = 0usize;

    let file = std::fs::File::create(&args.output)?;
    let mut writer = std::io::BufWriter::new(file);

    for path in &args.trajectories {
        let records = TrajectoryReader::read_all(path)?;
        tracing::info!(
            path = %path.display(),
            records = records.len(),
            "Loaded trajectory file"
        );

        // Group records by theorem name
        let mut by_theorem: HashMap<String, Vec<&TrajectoryRecord>> = HashMap::new();
        for record in &records {
            by_theorem
                .entry(record.theorem_name.clone())
                .or_default()
                .push(record);
        }

        for (theorem_name, theorem_records) in &by_theorem {
            // Skip if already seen from an earlier file
            if seen_theorems.contains(theorem_name) {
                continue;
            }

            // Only process proved theorems (must have at least one proof-complete record)
            let has_proof = theorem_records
                .iter()
                .any(|r| r.is_proof_complete);
            if !has_proof {
                continue;
            }

            // Extract proof path: Positive-labeled records sorted by depth
            let mut proof_path: Vec<&TrajectoryRecord> = theorem_records
                .iter()
                .filter(|r| r.label == TrajectoryLabel::Positive)
                .copied()
                .collect();
            proof_path.sort_by_key(|r| r.depth_from_root);

            // Filter out root node (empty tactic); keep terminal (empty state_pp
            // is fine — we use the parent's state_pp for the JSONL "state" field)
            let steps: Vec<&TrajectoryRecord> = proof_path
                .iter()
                .filter(|r| !r.tactic_applied.is_empty())
                .copied()
                .collect();

            // Apply min_steps filter
            if let Some(min) = args.min_steps {
                if steps.len() < min {
                    continue;
                }
            }

            if steps.is_empty() {
                continue;
            }

            seen_theorems.insert(theorem_name.clone());
            total_theorems += 1;

            // Write the proof-path state *before* each tactic is applied.
            // For a proof path: root -[tac0]-> s1 -[tac1]-> s2 -[tac2]-> QED
            // We need: (root_state, tac0, depth=0), (s1_state, tac1, depth=1), ...
            //
            // The Parquet records store each node's state_pp as the state *after*
            // the tactic was applied. So to get the state *before* tactic[i],
            // we need the parent's state_pp.
            //
            // Build a state_id -> state_pp map from all records.
            let id_to_state: HashMap<u64, &str> = theorem_records
                .iter()
                .map(|r| (r.state_id, r.state_pp.as_str()))
                .collect();

            for step in &steps {
                // The state before this tactic = parent's state_pp
                let before_state = step
                    .parent_state_id
                    .and_then(|pid| id_to_state.get(&pid))
                    .copied()
                    .unwrap_or("");

                if before_state.is_empty() {
                    continue;
                }

                // depth is the parent's depth (depth_from_root - 1), matching
                // generate-negatives convention where depth=0 is the first tactic
                let depth = step.depth_from_root.saturating_sub(1);

                let json = serde_json::json!({
                    "state": before_state,
                    "tactic": step.tactic_applied,
                    "theorem": theorem_name,
                    "depth": depth,
                });
                writeln!(writer, "{}", json)?;
                total_steps += 1;
            }
        }
    }

    writer.flush()?;

    println!("--- Export Proof Paths ---");
    println!("Input files:    {}", args.trajectories.len());
    println!("Theorems:       {total_theorems}");
    println!("Steps:          {total_steps}");
    println!("Output:         {}", args.output.display());
    if total_theorems > 0 {
        println!(
            "Avg steps/thm:  {:.1}",
            total_steps as f64 / total_theorems as f64
        );
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
        concurrency,
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
                    "{{spinner:.green}} [budget={budget}] [{{bar:30.cyan/blue}}] {{pos}}/{{len}} ({{msg}}) proved={{prefix}}"
                ))
                .expect("valid progress bar template")
                .progress_chars("=> "),
        );
        pb.set_prefix("0");
        pb.enable_steady_tick(Duration::from_secs(1));
        let budget_start = Instant::now();

        // Interleaved spawn + collect: drain completed tasks before each spawn
        // so the progress bar updates in real time.
        let mut join_set = tokio::task::JoinSet::new();
        let pass_n = args.pass_n;

        let mut per_theorem = Vec::new();
        let mut solved = 0u32;
        let mut all_times: Vec<f64> = Vec::new();
        let mut total_nodes_sum: f64 = 0.0;

        /// Process one completed eval outcome into accumulators.
        fn collect_eval(
            outcome: EvalOutcome,
            per_theorem: &mut Vec<TheoremResult>,
            solved: &mut u32,
            all_times: &mut Vec<f64>,
            total_nodes_sum: &mut f64,
            cumulative_solved_set: &mut HashSet<String>,
            pb: &ProgressBar,
            budget_start: Instant,
            total: u32,
        ) {
            if outcome.best.proved {
                *solved += 1;
                cumulative_solved_set.insert(outcome.name);
            }
            all_times.extend(outcome.times);
            *total_nodes_sum += outcome.total_nodes;
            per_theorem.push(outcome.best);
            pb.inc(1);
            pb.set_prefix(format!("{solved}"));
            let done = pb.position() as usize;
            if done > 0 {
                let elapsed = budget_start.elapsed().as_secs_f64();
                let remaining = elapsed * (total as usize - done) as f64 / done as f64;
                let eta = if remaining < 60.0 {
                    format!("{:.0}s", remaining)
                } else if remaining < 3600.0 {
                    format!("{:.0}m", remaining / 60.0)
                } else {
                    format!("{:.1}h", remaining / 3600.0)
                };
                pb.set_message(eta);
            }
        }

        for task in &index.theorems {
            // Drain completed tasks before waiting for a permit
            while let Some(join_result) = join_set.try_join_next() {
                match join_result {
                    Ok(outcome) => collect_eval(
                        outcome, &mut per_theorem, &mut solved, &mut all_times,
                        &mut total_nodes_sum, &mut cumulative_solved_set,
                        &pb, budget_start, total,
                    ),
                    Err(e) => tracing::error!(error = %e, "Eval task panicked"),
                }
            }

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
                            tracing::debug!(theorem = name, error = %e, "Eval search failed");
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

        // Final drain: collect remaining in-flight tasks
        while let Some(join_result) = join_set.join_next().await {
            match join_result {
                Ok(outcome) => collect_eval(
                    outcome, &mut per_theorem, &mut solved, &mut all_times,
                    &mut total_nodes_sum, &mut cumulative_solved_set,
                    &pb, budget_start, total,
                ),
                Err(e) => tracing::error!(error = %e, "Eval task panicked"),
            }
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

/// Split records into train/val by theorem name.
///
/// Ensures all records for a given theorem end up in the same split
/// (no data leakage between train and val).
fn split_records_by_theorem(
    records: &[ProofStateRecord],
    val_fraction: f64,
) -> (Vec<ProofStateRecord>, Vec<ProofStateRecord>) {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    // Group by theorem name
    let mut by_theorem: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, r) in records.iter().enumerate() {
        by_theorem.entry(&r.theorem_name).or_default().push(i);
    }

    // Shuffle theorem names deterministically and split
    let mut theorem_names: Vec<&str> = by_theorem.keys().copied().collect();
    theorem_names.sort(); // deterministic base order
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    theorem_names.shuffle(&mut rng);

    let val_count = (theorem_names.len() as f64 * val_fraction).round() as usize;
    let val_theorems: HashSet<&str> = theorem_names[..val_count].iter().copied().collect();

    let mut train = Vec::new();
    let mut val = Vec::new();
    for r in records {
        if val_theorems.contains(r.theorem_name.as_str()) {
            val.push(r.clone());
        } else {
            train.push(r.clone());
        }
    }

    (train, val)
}

/// Train the Energy-Based Model from trajectory data.
pub async fn run_train_ebm(args: TrainEbmArgs) -> anyhow::Result<()> {
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
    let mut records = ebm::load_records_from_parquet(&args.trajectories)?;
    tracing::info!(records = records.len(), "Loaded trajectory records");

    // 2b. Merge tactic pair positives (if provided)
    if let Some(ref tactic_pairs_path) = args.tactic_pairs {
        let theorem_names: HashSet<String> = records.iter().map(|r| r.theorem_name.clone()).collect();
        let tp_records = ebm::load_tactic_pairs(tactic_pairs_path, Some(&theorem_names))?;
        let tp_theorems: HashSet<&str> = tp_records.iter().map(|r| r.theorem_name.as_str()).collect();
        tracing::info!(
            tactic_pair_records = tp_records.len(),
            tactic_pair_theorems = tp_theorems.len(),
            "Merged tactic pair records for augmented training"
        );
        records.extend(tp_records);
    }

    // Split records 90/10 by theorem for train/val
    let (train_records, val_records) = split_records_by_theorem(&records, 0.1);
    let num_train = train_records.len();
    let num_val = val_records.len();

    // 3. Collect unique states from records (before building samplers)
    let all_state_strings: Vec<String> = {
        let mut seen = HashSet::new();
        for r in train_records.iter().chain(val_records.iter()) {
            seen.insert(r.state_pp.clone());
        }
        seen.into_iter().collect()
    };
    let unique_states: HashSet<&str> = all_state_strings.iter().map(|s| s.as_str()).collect();
    tracing::info!(
        train_records = num_train,
        val_records = num_val,
        unique_states = unique_states.len(),
        "Trajectory data loaded (train/val split)"
    );

    // 4. Build embedding cache (warm from disk or start empty)
    let mut cache = if let Some(ref cache_path) = args.embeddings_cache {
        tracing::info!(path = %cache_path.display(), "Loading embedding cache from disk (warm start)");
        EmbeddingCache::load(cache_path)?
    } else {
        EmbeddingCache::new(hidden_size)
    };

    // 5. Connect to SGLang and precompute all unique embeddings concurrently
    let config = SglangConfig {
        server_url: args.server_url.clone(),
        temperature: 0.0,
        top_p: 1.0,
        max_tactic_tokens: 1,
        hidden_size,
    };
    let client = SglangClient::new(config).await?;
    let handle = InferenceHandle::new(client);

    let encode_concurrency = args.encode_concurrency;
    let encode_batch_size = args.encode_batch_size;

    let checkpoint_path = args.save_embeddings.as_deref();

    let (newly_encoded, encode_errors) = if encode_batch_size > 0 {
        // Batched: send N texts per HTTP request for GPU-optimal batching
        cache
            .precompute_batched_with_checkpoint(
                &unique_states,
                |texts: Vec<String>| {
                    let h = handle.clone();
                    async move {
                        let embeddings = h.encode_batch(&texts).await?;
                        Ok(embeddings
                            .into_iter()
                            .map(|r| r.map(|emb| emb.data))
                            .collect())
                    }
                },
                encode_batch_size,
                encode_concurrency,
                hidden_size,
                checkpoint_path,
                10_000,
            )
            .await
    } else {
        // Individual concurrent requests (fallback)
        cache
            .precompute_concurrent(
                &unique_states,
                |state: String| {
                    let h = handle.clone();
                    async move {
                        let emb = h.encode(&state).await?;
                        Ok(emb.data)
                    }
                },
                encode_concurrency,
                hidden_size,
                checkpoint_path,
                10_000,
            )
            .await
    };

    if encode_errors > 0 {
        tracing::warn!(
            encode_errors,
            newly_encoded,
            "Some states failed to encode"
        );
    }

    tracing::info!(
        newly_encoded,
        encode_errors,
        cached_entries = cache.len(),
        "Embedding precomputation complete"
    );

    // 6. Save embedding cache before training (so it's not lost if training is killed)
    if let Some(ref save_path) = args.save_embeddings {
        cache.save(save_path)?;
        tracing::info!(path = %save_path.display(), "Saved embedding cache before training");
    }

    // 7. Filter records to only those with cached embeddings
    let train_before = train_records.len();
    let val_before = val_records.len();
    let train_records: Vec<_> = train_records
        .into_iter()
        .filter(|r| cache.get(&r.state_pp).is_some())
        .collect();
    let val_records: Vec<_> = val_records
        .into_iter()
        .filter(|r| cache.get(&r.state_pp).is_some())
        .collect();
    let train_dropped = train_before - train_records.len();
    let val_dropped = val_before - val_records.len();
    if train_dropped > 0 || val_dropped > 0 {
        tracing::warn!(
            train_dropped,
            val_dropped,
            train_remaining = train_records.len(),
            val_remaining = val_records.len(),
            "Filtered out records with uncached embeddings"
        );
    }

    // 8. Build samplers from filtered records
    let sampler = match ContrastiveSampler::from_trajectory_records(train_records, args.k_negatives) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "Cannot train EBM — not enough contrastive data after filtering");
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
    let val_sampler = if !val_records.is_empty() {
        match ContrastiveSampler::from_trajectory_records(val_records, args.k_negatives) {
            Ok(s) => {
                tracing::info!(
                    val_records = num_val,
                    val_eligible = s.num_eligible_theorems(),
                    "Validation sampler initialized"
                );
                Some(s)
            }
            Err(e) => {
                tracing::warn!(error = %e, "No validation set — val split has insufficient data");
                None
            }
        }
    } else {
        None
    };
    tracing::info!(
        train_eligible = sampler.num_eligible_theorems(),
        "Samplers built from cache-filtered records"
    );

    // 9. Create cache-only encode_fn (no network calls during training)
    let encode_fn = |state: &str| -> anyhow::Result<Vec<f32>> {
        cache.get_or_err(state)
    };

    // 10. Build training config
    let output_dir_str = args.output_dir.to_string_lossy().to_string();
    let training_config = EBMTrainingConfig::new()
        .with_lr(args.lr)
        .with_total_steps(args.steps)
        .with_batch_size(args.batch_size)
        .with_k_negatives(args.k_negatives)
        .with_checkpoint_dir(output_dir_str.clone());

    // 11. Train
    let _trained = ebm::train(
        &training_config,
        model,
        &encode_fn,
        &sampler,
        val_sampler.as_ref(),
        &device,
    )?;

    tracing::info!(
        cached = cache.len(),
        dim = cache.dim(),
        "Embedding cache stats after training"
    );

    // 10. Save EnergyHeadConfig alongside checkpoint
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

// ---------------------------------------------------------------------------
// Generate-negatives pipeline
// ---------------------------------------------------------------------------

/// Built-in Lean 4 tactics tried at each proof step as additional candidates.
/// These are cheap (no LLM call) and frequently produce valid divergent states
/// where LLM-generated candidates mostly produce `TacticResult::Failed`.
const PROBE_TACTICS: &[&str] = &[
    "simp", "ring", "omega", "norm_num", "decide", "trivial", "rfl", "tauto",
    "linarith", "push_neg", "contradiction", "exfalso", "constructor", "left",
    "right", "ext", "simp_all", "intro _",
];

/// Process a single theorem: walk the ground-truth proof path, generate LLM
/// candidates at each step, and classify them as Positive or Negative.
async fn process_theorem(
    pool: &Arc<LeanPool>,
    inference: &InferenceHandle,
    theorem_name: &str,
    steps: &[ebm::TacticStep],
    candidates_per_step: usize,
    target_negatives: usize,
) -> NegGenOutcome {
    let mut outcome = NegGenOutcome {
        theorem_name: theorem_name.to_string(),
        records: Vec::new(),
        steps_walked: 0,
        total_steps: steps.len(),
        completed: false,
        positives: 0,
        negatives: 0,
        alternative_proofs: 0,
    };

    // Start proof by theorem name
    let mut handle = match pool.start_proof_by_name_owned(theorem_name).await {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!(theorem = theorem_name, error = %e, "Failed to start proof");
            return outcome;
        }
    };

    let total_steps = steps.len();
    let open_prefix = infer_open_prefix(theorem_name);

    // Record ALL ground-truth steps as Positive upfront from tactic pair data.
    // This ensures balanced depth distribution regardless of Pantograph replay
    // success — ground-truth states from LeanDojo are valid at all depths.
    for (step_idx, step) in steps.iter().enumerate() {
        let remaining_depth = (total_steps - step_idx) as i32;
        outcome.records.push(TrajectoryRecord {
            theorem_name: theorem_name.to_string(),
            state_id: 0, // placeholder — not from Pantograph
            state_pp: step.state.clone(),
            tactic_applied: step.tactic.clone(),
            parent_state_id: if step_idx == 0 { None } else { Some(0) },
            label: TrajectoryLabel::Positive,
            depth_from_root: step.depth,
            remaining_depth,
            llm_log_prob: 0.0,
            ebm_score: 0.0,
            is_proof_complete: false,
            timestamp_ms: 0,
        });
        outcome.positives += 1;
    }

    // Now walk through Pantograph to generate candidates and negatives.
    // Ground-truth replay may fail (namespace mismatch ~55%), but positives
    // are already recorded above. We only need Pantograph for candidate testing.
    //
    // "Zombie Walk": when ground-truth fails, we advance via a probe tactic
    // to explore deeper states. On zombie paths we skip LLM candidates (avoids
    // unreliable hard negatives) and label probe-reached states as Positive
    // context (valid representation learning, not value comparison).
    let mut current_state_id = handle.initial_state().state_id;
    let mut pantograph_state_pp: Option<String> = None;
    let mut is_zombie = false;

    for (step_idx, step) in steps.iter().enumerate() {
        outcome.steps_walked = step_idx + 1;

        // Use Pantograph's goal if available, otherwise fall back to tactic pair data
        let state_pp = pantograph_state_pp
            .clone()
            .unwrap_or_else(|| step.state.clone());
        let remaining_depth = (total_steps - step_idx) as i32;

        // Generate LLM candidates at this proof state.
        // Skip on zombie paths — avoids unreliable hard negatives and saves GPU time.
        let candidates = if !is_zombie {
            match inference
                .generate_candidates(&state_pp, candidates_per_step)
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    tracing::trace!(
                        theorem = theorem_name,
                        step = step_idx,
                        error = %e,
                        "Failed to generate candidates"
                    );
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        let gt_normalized = normalize_tactic(&step.tactic);

        if !candidates.is_empty() {
            tracing::trace!(
                theorem = theorem_name,
                step = step_idx,
                candidates = candidates.len(),
                gt_tactic = step.tactic,
                "Generated candidates for step"
            );
        }

        // Try each non-ground-truth candidate.
        // Wrap with open prefix so candidates using short names (from LLM training
        // on code with `open` directives) resolve in Pantograph's bare namespace.
        for candidate in &candidates {
            let candidate_normalized = normalize_tactic(&candidate.text);
            if candidate_normalized == gt_normalized {
                tracing::trace!(
                    theorem = theorem_name,
                    candidate = candidate.text,
                    "Skipping ground-truth match"
                );
                continue;
            }

            let tactic_to_try = if let Some(ref prefix) = open_prefix {
                format!("{}{}", prefix, &candidate.text)
            } else {
                candidate.text.clone()
            };

            tracing::trace!(
                theorem = theorem_name,
                step = step_idx,
                candidate = candidate.text,
                "Trying candidate tactic"
            );

            match handle
                .run_tactic(current_state_id, Some(0), &tactic_to_try)
                .await
            {
                Ok(TacticResult::ProofComplete { .. }) => {
                    // Alternative proof found — record as Positive
                    outcome.records.push(TrajectoryRecord {
                        theorem_name: theorem_name.to_string(),
                        state_id: current_state_id,
                        state_pp: state_pp.clone(),
                        tactic_applied: candidate.text.clone(),
                        parent_state_id: if step_idx == 0 {
                            None
                        } else {
                            Some(current_state_id.saturating_sub(1))
                        },
                        label: TrajectoryLabel::Positive,
                        depth_from_root: step.depth,
                        remaining_depth,
                        llm_log_prob: candidate.log_prob,
                        ebm_score: 0.0,
                        is_proof_complete: true,
                        timestamp_ms: 0,
                    });
                    outcome.positives += 1;
                    outcome.alternative_proofs += 1;
                }
                Ok(TacticResult::Success { goals, .. }) if goals.is_empty() => {
                    // Empty goals = proof complete (guard for edge case)
                    outcome.records.push(TrajectoryRecord {
                        theorem_name: theorem_name.to_string(),
                        state_id: current_state_id,
                        state_pp: state_pp.clone(),
                        tactic_applied: candidate.text.clone(),
                        parent_state_id: if step_idx == 0 {
                            None
                        } else {
                            Some(current_state_id.saturating_sub(1))
                        },
                        label: TrajectoryLabel::Positive,
                        depth_from_root: step.depth,
                        remaining_depth,
                        llm_log_prob: candidate.log_prob,
                        ebm_score: 0.0,
                        is_proof_complete: true,
                        timestamp_ms: 0,
                    });
                    outcome.positives += 1;
                    outcome.alternative_proofs += 1;
                }
                Ok(TacticResult::Success {
                    state_id: div_state_id,
                    goals,
                }) => {
                    // Divergent path — record the resulting state as Negative
                    let neg_state_pp = goals
                        .first()
                        .map(|g| g.raw.clone())
                        .unwrap_or_default();
                    outcome.records.push(TrajectoryRecord {
                        theorem_name: theorem_name.to_string(),
                        state_id: current_state_id,
                        state_pp: neg_state_pp.clone(),
                        tactic_applied: candidate.text.clone(),
                        parent_state_id: if step_idx == 0 {
                            None
                        } else {
                            Some(current_state_id.saturating_sub(1))
                        },
                        label: TrajectoryLabel::Negative,
                        depth_from_root: step.depth + 1,
                        remaining_depth: -1,
                        llm_log_prob: candidate.log_prob,
                        ebm_score: 0.0,
                        is_proof_complete: false,
                        timestamp_ms: 0,
                    });
                    outcome.negatives += 1;

                    // Walk remaining tactics from the model's full proof attempt.
                    // Buffer records and decide labels after: if the chain reaches
                    // ProofComplete, intermediate states are Positive (valid proof
                    // path); otherwise they stay Negative (divergent).
                    let all_tactics = policy::extract_all_tactics(&candidate.raw_text);
                    if all_tactics.len() > 1 {
                        tracing::trace!(
                            theorem = theorem_name,
                            step = step_idx,
                            total_tactics = all_tactics.len(),
                            "Walking multi-tactic chain"
                        );
                        let mut chain_state_id = div_state_id;
                        let mut chain_records: Vec<TrajectoryRecord> = Vec::new();
                        let mut chain_proved = false;

                        for (chain_idx, chain_tactic) in all_tactics[1..].iter().enumerate() {
                            let chain_tactic_wrapped = if let Some(ref prefix) = open_prefix {
                                format!("{}{}", prefix, chain_tactic)
                            } else {
                                chain_tactic.clone()
                            };
                            match handle
                                .run_tactic(chain_state_id, Some(0), &chain_tactic_wrapped)
                                .await
                            {
                                Ok(TacticResult::ProofComplete { .. }) => {
                                    chain_proved = true;
                                    outcome.alternative_proofs += 1;
                                    tracing::trace!(
                                        theorem = theorem_name,
                                        chain_len = all_tactics.len(),
                                        "Multi-tactic chain completed proof"
                                    );
                                    break;
                                }
                                Ok(TacticResult::Success {
                                    state_id: next_id,
                                    goals: next_goals,
                                }) if next_goals.is_empty() => {
                                    chain_proved = true;
                                    outcome.alternative_proofs += 1;
                                    break;
                                }
                                Ok(TacticResult::Success {
                                    state_id: next_id,
                                    goals: next_goals,
                                }) => {
                                    let next_pp = next_goals
                                        .first()
                                        .map(|g| g.raw.clone())
                                        .unwrap_or_default();
                                    chain_records.push(TrajectoryRecord {
                                        theorem_name: theorem_name.to_string(),
                                        state_id: chain_state_id,
                                        state_pp: next_pp,
                                        tactic_applied: chain_tactic.clone(),
                                        parent_state_id: Some(chain_state_id),
                                        label: TrajectoryLabel::Negative, // provisional
                                        depth_from_root: step.depth + 2 + chain_idx as u32,
                                        remaining_depth: -1,
                                        llm_log_prob: 0.0,
                                        ebm_score: 0.0,
                                        is_proof_complete: false,
                                        timestamp_ms: 0,
                                    });
                                    chain_state_id = next_id;
                                }
                                Ok(TacticResult::Failed { message }) => {
                                    tracing::trace!(
                                        theorem = theorem_name,
                                        chain_tactic = chain_tactic.as_str(),
                                        error = message,
                                        "Chain tactic failed, stopping walk"
                                    );
                                    break;
                                }
                                Err(e) => {
                                    tracing::trace!(
                                        theorem = theorem_name,
                                        chain_tactic = chain_tactic.as_str(),
                                        error = %e,
                                        "Chain tactic error, stopping walk"
                                    );
                                    break;
                                }
                            }
                        }

                        // Relabel chain records based on outcome
                        if chain_proved {
                            // Also relabel the first divergent record (already pushed above)
                            if let Some(first_div) = outcome.records.last_mut() {
                                if first_div.label == TrajectoryLabel::Negative
                                    && first_div.theorem_name == theorem_name
                                {
                                    first_div.label = TrajectoryLabel::Positive;
                                    outcome.negatives -= 1;
                                    outcome.positives += 1;
                                }
                            }
                            for mut rec in chain_records {
                                rec.label = TrajectoryLabel::Positive;
                                outcome.positives += 1;
                                outcome.records.push(rec);
                            }
                        } else {
                            for rec in chain_records {
                                outcome.negatives += 1;
                                outcome.records.push(rec);
                            }
                        }
                    }
                }
                Ok(TacticResult::Failed { message }) => {
                    tracing::trace!(
                        theorem = theorem_name,
                        candidate = candidate.text,
                        error = message,
                        "Candidate tactic failed"
                    );
                }
                Err(e) => {
                    tracing::trace!(
                        theorem = theorem_name,
                        candidate = candidate.text,
                        error = %e,
                        "Candidate tactic error"
                    );
                }
            }

            // Early stop if we have enough negatives
            if outcome.negatives >= target_negatives {
                break;
            }
        }

        // Try probe tactics — built-in Lean 4 tactics that commonly apply.
        // These are cheap (no LLM call) and often produce valid divergent
        // states, boosting negative yield where LLM candidates mostly fail.
        //
        // We also save the last successful probe state for "probe advancement":
        // if ground-truth replay fails, we advance via this probe state to
        // explore deeper levels and generate negatives at depth > 1.
        let mut probe_advance_state: Option<(u64, Option<String>)> = None;

        if outcome.negatives < target_negatives {
            let llm_tried: HashSet<String> = candidates
                .iter()
                .map(|c| normalize_tactic(&c.text))
                .collect();
            let mut probe_hits = 0usize;

            for probe in PROBE_TACTICS {
                let probe_normalized = normalize_tactic(probe);
                if probe_normalized == gt_normalized
                    || llm_tried.contains(&probe_normalized)
                {
                    continue;
                }

                let probe_wrapped = if let Some(ref prefix) = open_prefix {
                    format!("{}{}", prefix, probe)
                } else {
                    probe.to_string()
                };

                match handle
                    .run_tactic(current_state_id, Some(0), &probe_wrapped)
                    .await
                {
                    Ok(TacticResult::ProofComplete { .. }) => {
                        outcome.records.push(TrajectoryRecord {
                            theorem_name: theorem_name.to_string(),
                            state_id: current_state_id,
                            state_pp: state_pp.clone(),
                            tactic_applied: probe.to_string(),
                            parent_state_id: if step_idx == 0 {
                                None
                            } else {
                                Some(current_state_id.saturating_sub(1))
                            },
                            label: TrajectoryLabel::Positive,
                            depth_from_root: step.depth,
                            remaining_depth,
                            llm_log_prob: 0.0,
                            ebm_score: 0.0,
                            is_proof_complete: true,
                            timestamp_ms: 0,
                        });
                        outcome.positives += 1;
                        outcome.alternative_proofs += 1;
                        probe_hits += 1;
                    }
                    Ok(TacticResult::Success { goals, .. }) if goals.is_empty() => {
                        outcome.records.push(TrajectoryRecord {
                            theorem_name: theorem_name.to_string(),
                            state_id: current_state_id,
                            state_pp: state_pp.clone(),
                            tactic_applied: probe.to_string(),
                            parent_state_id: if step_idx == 0 {
                                None
                            } else {
                                Some(current_state_id.saturating_sub(1))
                            },
                            label: TrajectoryLabel::Positive,
                            depth_from_root: step.depth,
                            remaining_depth,
                            llm_log_prob: 0.0,
                            ebm_score: 0.0,
                            is_proof_complete: true,
                            timestamp_ms: 0,
                        });
                        outcome.positives += 1;
                        outcome.alternative_proofs += 1;
                        probe_hits += 1;
                    }
                    Ok(TacticResult::Success {
                        state_id: probe_sid,
                        goals,
                    }) => {
                        let result_pp = goals
                            .first()
                            .map(|g| g.raw.clone())
                            .unwrap_or_default();
                        // Save for potential probe advancement
                        probe_advance_state =
                            Some((probe_sid, Some(result_pp.clone())));
                        // Normal path: Negative (divergent state).
                        // Zombie path: Positive (valid context for
                        // representation learning, not value comparison).
                        let label = if is_zombie {
                            TrajectoryLabel::Positive
                        } else {
                            TrajectoryLabel::Negative
                        };
                        outcome.records.push(TrajectoryRecord {
                            theorem_name: theorem_name.to_string(),
                            state_id: current_state_id,
                            state_pp: result_pp,
                            tactic_applied: probe.to_string(),
                            parent_state_id: if step_idx == 0 {
                                None
                            } else {
                                Some(current_state_id.saturating_sub(1))
                            },
                            label,
                            depth_from_root: step.depth + 1,
                            remaining_depth: -1,
                            llm_log_prob: 0.0,
                            ebm_score: 0.0,
                            is_proof_complete: false,
                            timestamp_ms: 0,
                        });
                        if is_zombie {
                            outcome.positives += 1;
                        } else {
                            outcome.negatives += 1;
                        }
                        probe_hits += 1;
                    }
                    Ok(TacticResult::Failed { .. }) | Err(_) => {
                        // Expected — most probe tactics won't apply
                    }
                }

                if outcome.negatives >= target_negatives {
                    break;
                }
            }

            if probe_hits > 0 {
                tracing::trace!(
                    theorem = theorem_name,
                    step = step_idx,
                    probe_hits,
                    "Probe tactics produced records"
                );
            }
        }

        // Early stop across steps
        if outcome.negatives >= target_negatives {
            break;
        }

        // Advance to the next proof state.
        if !is_zombie {
            // Normal path: try ground-truth tactic
            let advance_result = match handle
                .run_tactic(current_state_id, Some(0), &step.tactic)
                .await
            {
                Ok(r) => Some(r),
                Err(_) if open_prefix.is_some() => {
                    // Retry with all namespace prefixes opened
                    let open_tactic = format!(
                        "{}{}",
                        open_prefix.as_ref().unwrap(),
                        &step.tactic
                    );
                    match handle
                        .run_tactic(current_state_id, Some(0), &open_tactic)
                        .await
                    {
                        Ok(r) => Some(r),
                        Err(e) => {
                            tracing::trace!(
                                theorem = theorem_name,
                                step = step_idx,
                                tactic = step.tactic,
                                error = %e,
                                "Ground-truth tactic failed even with open namespaces"
                            );
                            None
                        }
                    }
                }
                Err(e) => {
                    tracing::trace!(
                        theorem = theorem_name,
                        step = step_idx,
                        tactic = step.tactic,
                        error = %e,
                        "Ground-truth tactic failed (no namespace to retry)"
                    );
                    None
                }
            };

            match advance_result {
                Some(TacticResult::Success { state_id, goals }) => {
                    current_state_id = state_id;
                    pantograph_state_pp = goals.first().map(|g| g.raw.clone());
                }
                Some(TacticResult::ProofComplete { .. }) => {
                    outcome.completed = true;
                    break;
                }
                Some(TacticResult::Failed { .. }) | None => {
                    // Ground-truth failed — enter zombie walk if a probe
                    // succeeded earlier (use its state to explore deeper).
                    if let Some((adv_id, adv_pp)) = probe_advance_state.take()
                    {
                        tracing::trace!(
                            theorem = theorem_name,
                            step = step_idx,
                            "Entering zombie walk (ground-truth failed, advancing via probe)"
                        );
                        is_zombie = true;
                        current_state_id = adv_id;
                        pantograph_state_pp = adv_pp;
                    } else {
                        break;
                    }
                }
            }
        } else {
            // Zombie path: advance via the last successful probe state.
            // No ground-truth to try — we're off the proof path.
            if let Some((adv_id, adv_pp)) = probe_advance_state.take() {
                current_state_id = adv_id;
                pantograph_state_pp = adv_pp;
            } else {
                tracing::trace!(
                    theorem = theorem_name,
                    step = step_idx,
                    "Zombie walk ended (no probe could advance)"
                );
                break;
            }
        }
    }

    // If we walked all steps without early break, mark completed
    if outcome.steps_walked == total_steps && !outcome.completed {
        outcome.completed = true;
    }

    outcome
}

/// Run the generate-negatives pipeline: walk known-good proof paths and generate
/// LLM candidates at each step, classifying them as Positive or Negative.
pub async fn run_generate_negatives(args: GenerateNegativesArgs) -> anyhow::Result<()> {
    let start = Instant::now();
    let concurrency = args.concurrency.max(1);

    // 1. Load config and build Lean pool
    let toml = load_search_toml(&args.config)?;
    let imports: Vec<String> = args.imports.clone();
    let lean_config =
        build_lean_pool_config(&toml.lean_pool, args.num_workers, Some(&imports))?;
    tracing::info!(
        num_workers = lean_config.num_workers,
        "Starting Lean worker pool"
    );
    let pool = Arc::new(LeanPool::new(lean_config).await?);

    // 2. Connect to SGLang server
    tracing::info!(url = args.server_url, "Connecting to SGLang inference server");
    let sglang_config = SglangConfig {
        server_url: args.server_url.clone(),
        temperature: args.temperature,
        top_p: 0.95,
        max_tactic_tokens: 128,
        hidden_size: 4096,
    };
    let client = SglangClient::new(sglang_config).await?;
    let inference = InferenceHandle::new(client);

    // 3. Load tactic pairs grouped by theorem
    tracing::info!(
        path = %args.tactic_pairs.display(),
        "Loading tactic pairs"
    );
    let grouped =
        ebm::load_tactic_pairs_grouped(&args.tactic_pairs, args.num_theorems, args.min_steps)?;
    let total = grouped.len();
    tracing::info!(theorems = total, "Loaded theorems for negative generation");

    // 4. Set up concurrency + writer + CTRL-C handler
    let mut writer = TrajectoryWriter::new(args.output.clone());
    let concurrency_sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let interrupted = Arc::new(AtomicBool::new(false));
    let sig_flag = interrupted.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        sig_flag.store(true, Ordering::Relaxed);
        tracing::warn!("Interrupted by CTRL-C, finishing in-flight theorems");
    });

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .expect("valid progress bar template")
            .progress_chars("=> "),
    );
    pb.enable_steady_tick(Duration::from_secs(1));

    // 5. Spawn per-theorem tasks and collect results concurrently.
    // We interleave spawning with draining completed tasks so the progress
    // bar updates in real time and auto-save works during the run.
    let mut join_set = tokio::task::JoinSet::new();
    let candidates_per_step = args.candidates_per_step;
    let target_negatives = args.target_negatives;

    let mut total_positives: usize = 0;
    let mut total_negatives: usize = 0;
    let mut total_alt_proofs: usize = 0;
    let mut completed_count: usize = 0;
    let mut skipped_count: usize = 0;
    let mut processed_count: usize = 0;
    // depth → (positives, negatives)
    let mut depth_dist: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
    // Survival rate: theorems with at least one negative at depth > 2
    let mut deep_neg_theorems: usize = 0;

    /// Collect one completed outcome from the JoinSet into accumulators.
    fn collect_outcome(
        outcome: NegGenOutcome,
        total_positives: &mut usize,
        total_negatives: &mut usize,
        total_alt_proofs: &mut usize,
        completed_count: &mut usize,
        skipped_count: &mut usize,
        processed_count: &mut usize,
        depth_dist: &mut BTreeMap<u32, (usize, usize)>,
        deep_neg_theorems: &mut usize,
        writer: &mut TrajectoryWriter,
        pb: &ProgressBar,
    ) {
        *processed_count += 1;
        *total_positives += outcome.positives;
        *total_negatives += outcome.negatives;
        *total_alt_proofs += outcome.alternative_proofs;
        if outcome.completed {
            *completed_count += 1;
        } else if outcome.steps_walked == 0 {
            *skipped_count += 1;
        }
        let mut has_deep_neg = false;
        for rec in &outcome.records {
            let entry = depth_dist.entry(rec.depth_from_root).or_insert((0, 0));
            match rec.label {
                TrajectoryLabel::Positive => entry.0 += 1,
                TrajectoryLabel::Negative => {
                    entry.1 += 1;
                    if rec.depth_from_root > 2 {
                        has_deep_neg = true;
                    }
                }
                _ => {}
            }
        }
        if has_deep_neg {
            *deep_neg_theorems += 1;
        }
        if !outcome.records.is_empty() {
            writer.record_all(outcome.records);
        }
        pb.inc(1);
    }

    for (theorem_name, steps) in grouped {
        if interrupted.load(Ordering::Relaxed) {
            break;
        }

        // Drain any completed tasks while we wait for a permit
        while let Some(join_result) = join_set.try_join_next() {
            match join_result {
                Ok(outcome) => {
                    collect_outcome(
                        outcome,
                        &mut total_positives,
                        &mut total_negatives,
                        &mut total_alt_proofs,
                        &mut completed_count,
                        &mut skipped_count,
                        &mut processed_count,
                        &mut depth_dist,
                        &mut deep_neg_theorems,
                        &mut writer,
                        &pb,
                    );
                    if processed_count % 50 == 0 {
                        writer.flush_partial()?;
                        tracing::info!(processed = processed_count, "Auto-saved checkpoint");
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "Generate-negatives task panicked");
                }
            }
        }

        let permit = concurrency_sem.clone().acquire_owned().await.unwrap();
        if interrupted.load(Ordering::Relaxed) {
            break;
        }

        pb.set_message(theorem_name.clone());

        let pool = Arc::clone(&pool);
        let inference = inference.clone();
        let interrupted = Arc::clone(&interrupted);

        let step_count = steps.len();
        join_set.spawn(async move {
            let _permit = permit;
            if interrupted.load(Ordering::Relaxed) {
                return NegGenOutcome {
                    theorem_name,
                    records: Vec::new(),
                    steps_walked: 0,
                    total_steps: step_count,
                    completed: false,
                    positives: 0,
                    negatives: 0,
                    alternative_proofs: 0,
                };
            }

            // Per-theorem timeout (120s)
            match tokio::time::timeout(
                std::time::Duration::from_secs(120),
                process_theorem(
                    &pool,
                    &inference,
                    &theorem_name,
                    &steps,
                    candidates_per_step,
                    target_negatives,
                ),
            )
            .await
            {
                Ok(outcome) => outcome,
                Err(_) => {
                    tracing::warn!(theorem = theorem_name, "Timed out after 120s");
                    NegGenOutcome {
                        theorem_name,
                        records: Vec::new(),
                        steps_walked: 0,
                        total_steps: step_count,
                        completed: false,
                        positives: 0,
                        negatives: 0,
                        alternative_proofs: 0,
                    }
                }
            }
        });
    }

    // 6. Drain remaining completed tasks
    while let Some(join_result) = join_set.join_next().await {
        match join_result {
            Ok(outcome) => {
                collect_outcome(
                    outcome,
                    &mut total_positives,
                    &mut total_negatives,
                    &mut total_alt_proofs,
                    &mut completed_count,
                    &mut skipped_count,
                    &mut processed_count,
                    &mut depth_dist,
                    &mut deep_neg_theorems,
                    &mut writer,
                    &pb,
                );
                if processed_count % 50 == 0 && processed_count > 0 {
                    writer.flush_partial()?;
                    tracing::info!(processed = processed_count, "Auto-saved checkpoint");
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "Generate-negatives task panicked");
            }
        }
    }

    pb.finish_with_message("done");

    // 7. Write final Parquet
    let record_count = writer.len();
    writer.finish()?;

    // Shutdown pool
    pool.shutdown().await;

    // 8. Print summary
    let elapsed = start.elapsed();
    let partial_note = if interrupted.load(Ordering::Relaxed) {
        " (Partial — interrupted by CTRL-C)"
    } else {
        ""
    };

    println!();
    println!("══════════════════════════════════════════");
    println!(" Generate-Negatives Results{partial_note}");
    println!("──────────────────────────────────────────");
    println!(" Theorems attempted:  {processed_count}");
    println!(" Completed:           {completed_count}");
    println!(" Skipped (start fail):{skipped_count}");
    println!(" Positives:           {total_positives} (ground-truth + alternative)");
    println!(" Negatives:           {total_negatives}");
    println!(" Alternative proofs:  {total_alt_proofs}");
    println!(" Total records:       {record_count}");
    println!(" Output:              {}", args.output.display());
    println!(" Wall time:           {:.1}s", elapsed.as_secs_f64());
    println!(" Concurrency:         {concurrency}");
    println!(" Candidates/step:     {candidates_per_step} LLM + {} probe", PROBE_TACTICS.len());
    println!(" Target negs/theorem: {target_negatives}");
    println!("──────────────────────────────────────────");
    println!(" Depth Distribution:");
    println!(" {:>5}  {:>8}  {:>8}  {:>8}", "Depth", "Pos", "Neg", "Total");
    for (depth, (pos, neg)) in &depth_dist {
        println!(" {:>5}  {:>8}  {:>8}  {:>8}", depth, pos, neg, pos + neg);
    }
    println!("──────────────────────────────────────────");
    // Survival Rate: % of theorems with negatives at depth > 2
    let non_skipped = processed_count.saturating_sub(skipped_count);
    let survival_pct = if non_skipped > 0 {
        (deep_neg_theorems as f64 / non_skipped as f64) * 100.0
    } else {
        0.0
    };
    println!(
        " Survival Rate:  {deep_neg_theorems}/{non_skipped} ({survival_pct:.1}%) theorems with deep negatives (depth>2)"
    );
    if survival_pct >= 30.0 {
        println!("   -> SAFE: sufficient deep negative coverage");
    } else if survival_pct >= 10.0 {
        println!("   -> MARGINAL: consider namespace/import fix for deeper coverage");
    } else {
        println!("   -> WARNING: deep negative skew too severe — namespace/import fix needed");
    }
    println!("══════════════════════════════════════════");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_open_prefix() {
        assert_eq!(
            infer_open_prefix("Polynomial.natDegree_cyclotomic'"),
            Some("open Polynomial in ".to_string())
        );
        assert_eq!(
            infer_open_prefix("MeasureTheory.Measure.hahn"),
            Some("open MeasureTheory MeasureTheory.Measure in ".to_string())
        );
        assert_eq!(
            infer_open_prefix("CategoryTheory.ShortComplex.cycles_ext_iff"),
            Some("open CategoryTheory CategoryTheory.ShortComplex in ".to_string())
        );
        assert_eq!(infer_open_prefix("simple_name"), None);
        assert_eq!(infer_open_prefix(""), None);
    }

    #[test]
    fn test_normalize_tactic() {
        assert_eq!(normalize_tactic("intro  h"), "intro h");
        assert_eq!(normalize_tactic("  simp [add_comm]  "), "simp [add_comm]");
        assert_eq!(
            normalize_tactic("rw\t[Nat.add_comm]\n"),
            "rw [Nat.add_comm]"
        );
        assert_eq!(normalize_tactic("exact h"), "exact h");
    }

    #[test]
    fn test_probe_tactics_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for tactic in PROBE_TACTICS {
            assert!(
                seen.insert(normalize_tactic(tactic)),
                "Duplicate probe tactic: {tactic}"
            );
        }
        assert!(
            PROBE_TACTICS.len() >= 10,
            "Expected at least 10 probe tactics, got {}",
            PROBE_TACTICS.len()
        );
    }
}
