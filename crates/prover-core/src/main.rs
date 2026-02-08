mod config;
mod pipeline;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use pipeline::{EvalArgs, SearchArgs, TrainEbmArgs};

/// burn-qed: Lean 4 theorem prover with LLM policy and EBM value function.
#[derive(Parser)]
#[command(name = "burn-qed", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

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
    },
    /// Print statistics from a trajectory Parquet file.
    Eval {
        /// Path to the trajectory Parquet file.
        #[arg(long)]
        input: PathBuf,
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
        } => {
            pipeline::run_search(SearchArgs {
                config,
                model_path,
                theorems,
                output,
                num_workers,
                dry_run,
                ebm_path,
            })
            .await
        }
        Command::Eval { input } => pipeline::run_eval(EvalArgs { input }),
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
