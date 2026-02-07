//! Proof search pipeline and evaluation utilities.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};

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
}

/// Arguments for the `eval` subcommand.
#[derive(Debug)]
pub struct EvalArgs {
    /// Path to the trajectory Parquet file.
    pub input: PathBuf,
}

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
    let policy = MutexPolicyProvider::new(generator);

    // 4. Load theorem index
    let index = TheoremIndex::from_json(&args.theorems)?;
    tracing::info!(count = index.len(), "Loaded theorems");

    // 4b. Dry-run: verify setup and exit early
    if args.dry_run {
        println!("Dry run — setup verified successfully");
        println!("  Model: {}", args.model_path.display());
        println!("  Theorems: {} loaded", index.len());
        println!("  Workers: {}", pool.num_workers());
        pool.shutdown().await;
        return Ok(());
    }

    // 5. Run search with progress bar
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

    for task in &index.theorems {
        if interrupted {
            break;
        }

        pb.set_message(task.name.clone());

        let search_fut = engine.search_one(&pool, &policy, None, &task.name, &task.statement);

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

    // 6. Write Parquet
    let record_count = writer.len();
    writer.finish()?;

    // 7. Shutdown pool
    pool.shutdown().await;

    // 8. Print enhanced summary
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
