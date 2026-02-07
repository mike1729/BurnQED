/// Proof search pipeline and evaluation utilities.

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
    let policy_config = PolicyConfig::new(args.model_path);
    let generator = TacticGenerator::load(&policy_config)?;
    let policy = MutexPolicyProvider::new(generator);

    // 4. Load theorem index
    let index = TheoremIndex::from_json(&args.theorems)?;
    tracing::info!(count = index.len(), "Loaded theorems");

    // 5. Run search with progress bar
    let engine = SearchEngine::new(toml.search);
    let mut writer = TrajectoryWriter::new(args.output.clone());
    let mut proved_count: u32 = 0;
    let total = index.len() as u32;

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .expect("valid progress bar template")
            .progress_chars("=> "),
    );

    for task in &index.theorems {
        pb.set_message(task.name.clone());

        match engine
            .search_one(&pool, &policy, None, &task.name, &task.statement)
            .await
        {
            Ok(result) => {
                if result.proved {
                    proved_count += 1;
                    tracing::info!(
                        theorem = task.name,
                        tactics = ?result.proof_tactics,
                        nodes = result.nodes_expanded,
                        time_ms = result.wall_time_ms,
                        "Proved"
                    );
                }
                let labeled = TrajectoryWriter::from_search_result(&result);
                writer.record_all(labeled);
            }
            Err(e) => {
                tracing::warn!(theorem = task.name, error = %e, "Search failed, skipping");
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

    // 8. Print summary
    let elapsed = start.elapsed();
    println!("\n--- Search Summary ---");
    println!("Proved: {proved_count}/{total}");
    println!("Records: {record_count}");
    println!("Output: {}", args.output.display());
    println!("Elapsed: {:.1}s", elapsed.as_secs_f64());

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
