mod config;
mod pipeline;
pub mod results;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use pipeline::{CompareArgs, EvalArgs, SearchArgs, SummaryArgs, TrainEbmArgs};

/// burn-qed: Lean 4 theorem prover with LLM policy and EBM value function.
#[derive(Parser)]
#[command(name = "burn-qed", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// CLI subcommands for proof search, evaluation, comparison, and EBM training.
#[derive(Subcommand)]
enum Command {
    /// Run proof search over a batch of theorems.
    Search {
        /// Path to search config TOML file.
        #[arg(long, default_value = "configs/search.toml")]
        config: PathBuf,
        /// Path to the HuggingFace model directory.
        #[arg(long)]
        model_path: PathBuf,
        /// Path to the theorem index JSON file.
        #[arg(long)]
        theorems: PathBuf,
        /// Path for the output trajectory Parquet file.
        #[arg(long)]
        output: PathBuf,
        /// Override the number of Lean workers.
        #[arg(long)]
        num_workers: Option<usize>,
        /// Load model and pool but don't search. Verifies environment setup.
        #[arg(long)]
        dry_run: bool,
        /// Path to EBM checkpoint directory for value-guided search.
        #[arg(long)]
        ebm_path: Option<PathBuf>,
        /// Resume from a partial trajectory file — skip already-searched theorems.
        #[arg(long)]
        resume_from: Option<PathBuf>,
        /// Override sampling temperature for tactic generation.
        #[arg(long)]
        temperature: Option<f64>,
        /// Number of theorems to search in parallel (default: 1 = sequential).
        #[arg(long, default_value_t = 1)]
        concurrency: usize,
        /// Maximum number of theorems to search (truncates the index).
        #[arg(long)]
        max_theorems: Option<usize>,
        /// Lean modules to import (e.g., "Init", "Mathlib"). Default: Init.
        #[arg(long, value_delimiter = ',')]
        imports: Option<Vec<String>>,
    },
    /// Print statistics from a trajectory Parquet file.
    Summary {
        /// Path to the trajectory Parquet file.
        #[arg(long)]
        input: PathBuf,
    },
    /// Evaluate a model at multiple search budgets.
    Eval {
        /// Path to search config TOML file.
        #[arg(long, default_value = "configs/search.toml")]
        config: PathBuf,
        /// Path to the HuggingFace model directory.
        #[arg(long)]
        model_path: PathBuf,
        /// Path to EBM checkpoint directory for value-guided search.
        #[arg(long)]
        ebm_path: Option<PathBuf>,
        /// Path to the theorem index JSON file.
        #[arg(long)]
        theorems: PathBuf,
        /// Comma-separated list of node budgets to evaluate at.
        #[arg(long, value_delimiter = ',', default_values_t = vec![100, 300, 600])]
        budgets: Vec<u32>,
        /// Number of attempts per theorem per budget (best-of-N).
        #[arg(long, default_value_t = 1)]
        pass_n: u32,
        /// Path to write JSON evaluation results.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Override the number of Lean workers.
        #[arg(long)]
        num_workers: Option<usize>,
        /// Number of theorems to search in parallel (default: 1 = sequential).
        #[arg(long, default_value_t = 1)]
        concurrency: usize,
        /// Maximum number of theorems to evaluate (truncates the index).
        #[arg(long)]
        max_theorems: Option<usize>,
        /// Lean modules to import (e.g., "Init", "Mathlib"). Default: Init.
        #[arg(long, value_delimiter = ',')]
        imports: Option<Vec<String>>,
    },
    /// Compare evaluation results across iterations.
    Compare {
        /// Paths to evaluation result JSON files.
        #[arg(long, required = true, num_args = 1..)]
        results: Vec<PathBuf>,
    },
    /// Train the Energy-Based Model from trajectory data.
    TrainEbm {
        /// Path(s) to trajectory Parquet files.
        #[arg(long, required = true, num_args = 1..)]
        trajectories: Vec<PathBuf>,
        /// Directory for saving checkpoints.
        #[arg(long, default_value = "checkpoints/ebm")]
        output_dir: PathBuf,
        /// Path to the HuggingFace LLM model directory (for encoding).
        #[arg(long)]
        llm_path: PathBuf,
        /// Resume training from a checkpoint directory.
        #[arg(long)]
        resume_from: Option<PathBuf>,
        /// Total training steps.
        #[arg(long, default_value_t = 50_000)]
        steps: usize,
        /// Learning rate.
        #[arg(long, default_value_t = 1e-4)]
        lr: f64,
        /// Batch size.
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        /// Number of negative samples per positive.
        #[arg(long, default_value_t = 4)]
        k_negatives: usize,
        /// Path to precomputed embedding cache (Parquet). If omitted, precomputes from LLM.
        #[arg(long)]
        embeddings_cache: Option<PathBuf>,
        /// Save precomputed embeddings to this path for reuse.
        #[arg(long)]
        save_embeddings: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Search {
            config,
            model_path,
            theorems,
            output,
            num_workers,
            dry_run,
            ebm_path,
            resume_from,
            temperature,
            concurrency,
            max_theorems,
            imports,
        } => {
            pipeline::run_search(SearchArgs {
                config,
                model_path,
                theorems,
                output,
                num_workers,
                dry_run,
                ebm_path,
                resume_from,
                temperature,
                concurrency,
                max_theorems,
                imports,
            })
            .await
        }
        Command::Summary { input } => pipeline::run_summary(SummaryArgs { input }),
        Command::Eval {
            config,
            model_path,
            ebm_path,
            theorems,
            budgets,
            pass_n,
            output,
            num_workers,
            concurrency,
            max_theorems,
            imports,
        } => {
            pipeline::run_eval(EvalArgs {
                config,
                model_path,
                ebm_path,
                theorems,
                budgets,
                pass_n,
                output,
                num_workers,
                concurrency,
                max_theorems,
                imports,
            })
            .await
        }
        Command::Compare { results } => pipeline::run_compare(CompareArgs { results }),
        Command::TrainEbm {
            trajectories,
            output_dir,
            llm_path,
            resume_from,
            steps,
            lr,
            batch_size,
            k_negatives,
            embeddings_cache,
            save_embeddings,
        } => pipeline::run_train_ebm(TrainEbmArgs {
            trajectories,
            output_dir,
            llm_path,
            resume_from,
            steps,
            lr,
            batch_size,
            k_negatives,
            embeddings_cache,
            save_embeddings,
        }),
    }
}
