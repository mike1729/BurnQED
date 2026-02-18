mod config;
mod pipeline;
pub mod results;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use pipeline::{CompareArgs, EvalArgs, ExportProofPathsArgs, GenerateNegativesArgs, ProbeArgs, SearchArgs, SummaryArgs, TrainEbmArgs};

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
        /// URL of the SGLang inference server (e.g., http://localhost:30000).
        #[arg(long)]
        server_url: String,
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
        /// Override maximum tokens per generated tactic.
        #[arg(long)]
        max_tactic_tokens: Option<usize>,
        /// Override number of candidate tactics per expansion.
        #[arg(long)]
        num_candidates: Option<usize>,
        /// Number of theorems to search in parallel.
        #[arg(long, default_value_t = 8)]
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
        /// Output as JSON instead of human-readable text.
        #[arg(long)]
        json: bool,
    },
    /// Export proof paths from trajectory Parquet files to tactic-pairs JSONL.
    ExportProofPaths {
        /// Path(s) to trajectory Parquet files.
        #[arg(long, required = true, num_args = 1..)]
        trajectories: Vec<PathBuf>,
        /// Output tactic-pairs JSONL file.
        #[arg(long)]
        output: PathBuf,
        /// Minimum number of proof steps per theorem (filters out short proofs).
        #[arg(long)]
        min_steps: Option<usize>,
    },
    /// Evaluate a model at multiple search budgets.
    Eval {
        /// Path to search config TOML file.
        #[arg(long, default_value = "configs/search.toml")]
        config: PathBuf,
        /// URL of the SGLang inference server (e.g., http://localhost:30000).
        #[arg(long)]
        server_url: String,
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
        /// Number of theorems to search in parallel.
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
        /// Maximum number of theorems to evaluate (truncates the index).
        #[arg(long)]
        max_theorems: Option<usize>,
        /// Override maximum tokens per generated tactic.
        #[arg(long)]
        max_tactic_tokens: Option<usize>,
        /// Override number of candidate tactics per expansion.
        #[arg(long)]
        num_candidates: Option<usize>,
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
    /// Generate high-quality contrastive negatives by walking known-good proof paths.
    GenerateNegatives {
        /// Path to tactic pairs JSONL file (from LeanDojo traces).
        #[arg(long)]
        tactic_pairs: PathBuf,
        /// URL of the SGLang inference server (e.g., http://localhost:30000).
        #[arg(long)]
        server_url: String,
        /// Path for the output trajectory Parquet file.
        #[arg(long)]
        output: PathBuf,
        /// Maximum number of theorems to process.
        #[arg(long)]
        num_theorems: Option<usize>,
        /// Minimum number of proof steps per theorem (filters out short proofs).
        #[arg(long)]
        min_steps: Option<usize>,
        /// Number of LLM candidates to generate per proof step.
        #[arg(long, default_value_t = 8)]
        candidates_per_step: usize,
        /// Target number of negatives per theorem before early stop.
        #[arg(long, default_value_t = 15)]
        target_negatives: usize,
        /// Sampling temperature for tactic generation.
        #[arg(long, default_value_t = 1.0)]
        temperature: f64,
        /// Lean modules to import (e.g., "Init", "Mathlib").
        #[arg(long, value_delimiter = ',', default_values_t = vec!["Init".to_string(), "Mathlib".to_string()])]
        imports: Vec<String>,
        /// Number of theorems to process in parallel.
        #[arg(long, default_value_t = 6)]
        concurrency: usize,
        /// Override the number of Lean workers.
        #[arg(long)]
        num_workers: Option<usize>,
        /// Path to search config TOML file.
        #[arg(long, default_value = "configs/search.toml")]
        config: PathBuf,
    },
    /// Probe-only tactic search: filter easy theorems using built-in tactics (no LLM).
    Probe {
        /// Path to search config TOML file.
        #[arg(long, default_value = "configs/search.toml")]
        config: PathBuf,
        /// Path to the theorem index JSON file.
        #[arg(long)]
        theorems: PathBuf,
        /// Path for the output JSON file (easy/hard/stats).
        #[arg(long)]
        output: PathBuf,
        /// Override the number of Lean workers.
        #[arg(long)]
        num_workers: Option<usize>,
        /// Number of theorems to search in parallel.
        #[arg(long, default_value_t = 32)]
        concurrency: usize,
        /// Maximum number of theorems to process (truncates the index).
        #[arg(long)]
        max_theorems: Option<usize>,
        /// Lean modules to import (e.g., "Init", "Mathlib"). Default: Init.
        #[arg(long, value_delimiter = ',')]
        imports: Option<Vec<String>>,
        /// Override max_nodes for probe search (default: 100).
        #[arg(long)]
        max_nodes: Option<u32>,
        /// Write hard theorems as TheoremIndex JSON for use with --theorems in search.
        #[arg(long)]
        hard_theorems: Option<PathBuf>,
    },
    /// Train the Energy-Based Model from trajectory data.
    TrainEbm {
        /// Path(s) to trajectory Parquet files.
        #[arg(long, required = true, num_args = 1..)]
        trajectories: Vec<PathBuf>,
        /// Directory for saving checkpoints.
        #[arg(long, default_value = "checkpoints/ebm")]
        output_dir: PathBuf,
        /// URL of the SGLang inference server (for embedding precomputation).
        #[arg(long)]
        server_url: String,
        /// Hidden size of the LLM (embedding dimension). Default: 4096 for DeepSeek-Prover-V2-7B.
        #[arg(long, default_value_t = 4096)]
        hidden_size: usize,
        /// Resume training from a checkpoint directory.
        #[arg(long)]
        resume_from: Option<PathBuf>,
        /// Total training steps.
        #[arg(long, default_value_t = 50_000)]
        steps: usize,
        /// Learning rate.
        #[arg(long, default_value_t = 3e-5)]
        lr: f64,
        /// Batch size.
        #[arg(long, default_value_t = 256)]
        batch_size: usize,
        /// Number of negative samples per positive.
        #[arg(long, default_value_t = 7)]
        k_negatives: usize,
        /// Path to precomputed embedding cache (Parquet). If omitted, precomputes from SGLang.
        #[arg(long)]
        embeddings_cache: Option<PathBuf>,
        /// Save precomputed embeddings to this path for reuse.
        #[arg(long)]
        save_embeddings: Option<PathBuf>,
        /// Path to tactic pairs JSONL file for augmenting training data.
        #[arg(long)]
        tactic_pairs: Option<PathBuf>,
        /// Number of concurrent encode requests during embedding precomputation.
        #[arg(long, default_value_t = 32)]
        encode_concurrency: usize,
        /// Batch size for batched encode requests. Each batch is a single HTTP request
        /// to SGLang, enabling GPU-optimal batching. Set to 0 for individual requests.
        #[arg(long, default_value_t = 8)]
        encode_batch_size: usize,
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
            server_url,
            theorems,
            output,
            num_workers,
            dry_run,
            ebm_path,
            resume_from,
            temperature,
            max_tactic_tokens,
            num_candidates,
            concurrency,
            max_theorems,
            imports,
        } => {
            pipeline::run_search(SearchArgs {
                config,
                server_url,
                theorems,
                output,
                num_workers,
                dry_run,
                ebm_path,
                resume_from,
                temperature,
                max_tactic_tokens,
                num_candidates,
                concurrency,
                max_theorems,
                imports,
            })
            .await
        }
        Command::Summary { input, json } => pipeline::run_summary(SummaryArgs { input, json }),
        Command::ExportProofPaths {
            trajectories,
            output,
            min_steps,
        } => pipeline::run_export_proof_paths(ExportProofPathsArgs {
            trajectories,
            output,
            min_steps,
        }),
        Command::Eval {
            config,
            server_url,
            ebm_path,
            theorems,
            budgets,
            pass_n,
            output,
            num_workers,
            concurrency,
            max_theorems,
            max_tactic_tokens,
            num_candidates,
            imports,
        } => {
            pipeline::run_eval(EvalArgs {
                config,
                server_url,
                ebm_path,
                theorems,
                budgets,
                pass_n,
                output,
                num_workers,
                concurrency,
                max_theorems,
                max_tactic_tokens,
                num_candidates,
                imports,
            })
            .await
        }
        Command::Compare { results } => pipeline::run_compare(CompareArgs { results }),
        Command::GenerateNegatives {
            tactic_pairs,
            server_url,
            output,
            num_theorems,
            min_steps,
            candidates_per_step,
            target_negatives,
            temperature,
            imports,
            concurrency,
            num_workers,
            config,
        } => {
            pipeline::run_generate_negatives(GenerateNegativesArgs {
                config,
                server_url,
                tactic_pairs,
                output,
                num_theorems,
                min_steps,
                candidates_per_step,
                target_negatives,
                temperature,
                imports,
                concurrency,
                num_workers,
            })
            .await
        }
        Command::Probe {
            config,
            theorems,
            output,
            num_workers,
            concurrency,
            max_theorems,
            imports,
            max_nodes,
            hard_theorems,
        } => {
            pipeline::run_probe(ProbeArgs {
                config,
                theorems,
                output,
                num_workers,
                concurrency,
                max_theorems,
                imports,
                max_nodes,
                hard_theorems,
            })
            .await
        }
        Command::TrainEbm {
            trajectories,
            output_dir,
            server_url,
            hidden_size,
            resume_from,
            steps,
            lr,
            batch_size,
            k_negatives,
            embeddings_cache,
            save_embeddings,
            tactic_pairs,
            encode_concurrency,
            encode_batch_size,
        } => {
            pipeline::run_train_ebm(TrainEbmArgs {
                trajectories,
                output_dir,
                server_url,
                hidden_size,
                resume_from,
                steps,
                lr,
                batch_size,
                k_negatives,
                embeddings_cache,
                save_embeddings,
                tactic_pairs,
                encode_concurrency,
                encode_batch_size,
            })
            .await
        }
    }
}
