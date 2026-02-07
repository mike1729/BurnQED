mod config;
mod pipeline;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use pipeline::{EvalArgs, SearchArgs};

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
    },
    /// Print statistics from a trajectory Parquet file.
    Eval {
        /// Path to the trajectory Parquet file.
        #[arg(long)]
        input: PathBuf,
    },
    /// Train the Energy-Based Model (Phase 4 — not yet implemented).
    TrainEbm,
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
        } => {
            pipeline::run_search(SearchArgs {
                config,
                model_path,
                theorems,
                output,
                num_workers,
                dry_run,
            })
            .await
        }
        Command::Eval { input } => pipeline::run_eval(EvalArgs { input }),
        Command::TrainEbm => {
            println!("train-ebm: not yet implemented (Phase 4)");
            Ok(())
        }
    }
}
