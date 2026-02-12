```
 ██████╗ ██╗   ██╗██████╗ ███╗   ██╗       ██████╗ ███████╗██████╗
 ██╔══██╗██║   ██║██╔══██╗████╗  ██║      ██╔═══██╗██╔════╝██╔══██╗
 ██████╔╝██║   ██║██████╔╝██╔██╗ ██║█████╗██║   ██║█████╗  ██║  ██║
 ██╔══██╗██║   ██║██╔══██╗██║╚██╗██║╚════╝██║▄▄ ██║██╔══╝  ██║  ██║
 ██████╔╝╚██████╔╝██║  ██║██║ ╚████║      ╚██████╔╝███████╗██████╔╝
 ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝       ╚══▀▀═╝ ╚══════╝╚═════╝
```

**Lean 4 theorem prover combining LLM policy with Energy-Based Model value function, trained via expert iteration.**

A single DeepSeek-Prover-V2-7B backbone serves both autoregressive tactic generation (policy) and mean-pooled state embeddings (value), AlphaZero-style. A small EBM head (~5M params) is the only component trained in Rust. LLM fine-tuning runs in Python with LoRA/PEFT.

## Architecture

```
DeepSeek-Prover-V2-7B (candle, frozen)
├── Policy head: autoregressive tactic generation (LM head)
└── Mean-pool hidden states → detached Vec<f32>
                                    │
                                    ▼
                    Energy Head (burn-rs, trainable)
                    SpectralNorm MLP: 4096 → 512 → 256 → 1
                    Output: scalar energy (lower = more provable)

Lean 4 REPL Pool (tokio, Pantograph JSON protocol)
└── Verifies tactics against proof states, returns new goals
```

Best-first search expands nodes by combined LLM log-probability and EBM energy score, verified against Lean 4 via Pantograph.

## Project Structure

```
BurnQED/
├── crates/
│   ├── lean-repl/       # Async Pantograph client, worker pool, ProofHandle pattern
│   ├── policy/          # LLM tactic generation (candle), tokenizer, encode_only()
│   ├── search/          # Best-first proof search engine, trait-based
│   ├── trajectory/      # Parquet I/O for search trajectories
│   ├── ebm/             # Energy-Based Model: SpectralNorm MLP, training, inference
│   ├── prover-core/     # CLI binary tying everything together
│   └── burn-contrib/    # Upstream burn-rs PR modules (stub)
├── python/
│   ├── training/        # LoRA fine-tuning (train_llm.py, merge_lora.py)
│   └── data/            # Mathlib tracing, tactic pair extraction
├── scripts/             # Setup, baseline, iteration, resume orchestration
├── configs/             # search.toml, models.toml
├── data/                # test_theorems.json, theorem indices
├── vendor/Pantograph/   # Git submodule (Lean 4 REPL)
└── docs/                # Full plan, phase instructions
```

## Prerequisites

- **Rust** (stable, edition 2021)
- **Lean 4** via [elan](https://github.com/leanprover/elan)
- **Python 3.10+** with `torch`, `transformers`, `peft`, `accelerate` (for LLM fine-tuning)
- **GPU** recommended for LLM inference and training; CPU works for development and EBM-only training
- **DeepSeek-Prover-V2-7B** weights ([HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B))

## Quick Start

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/<you>/BurnQED.git
cd BurnQED

# Build Pantograph (Lean 4 REPL)
./scripts/setup_pantograph.sh

# Build the prover
cargo build --release -p prover-core

# Smoke test (~2-3 min on A100: LLM search → train EBM → search with EBM → compare)
./scripts/lean_start.sh /path/to/deepseek-prover-v2-7b
```

For cloud GPU setup (installs Rust, elan, Python deps, builds everything):

```bash
bash scripts/setup_cloud.sh
```

## CLI Reference

All commands are subcommands of `cargo run --release -p prover-core --`:

### `search` — Run proof search

```bash
cargo run --release -p prover-core -- search \
    --model-path models/deepseek-prover-v2-7b \
    --theorems data/test_theorems.json \
    --output trajectories/run.parquet \
    --ebm-path checkpoints/ebm \       # optional: EBM value guidance
    --resume-from partial.parquet \     # optional: skip already-proved theorems
    --temperature 0.8 \                 # optional: sampling temperature
    --concurrency 4 \                   # optional: parallel theorem search
    --num-workers 64                    # optional: Lean worker pool size
```

Use `--dry-run` to validate environment setup without searching.

### `eval` — Multi-budget evaluation

```bash
cargo run --release -p prover-core -- eval \
    --model-path models/deepseek-prover-v2-7b \
    --theorems data/minif2f_test.json \
    --budgets 100,300,600 \
    --pass-n 8 \
    --output eval_results/iter_1.json \
    --ebm-path checkpoints/ebm/iter_1
```

### `train-ebm` — Train EBM from trajectories

```bash
cargo run --release -p prover-core -- train-ebm \
    --trajectories trajectories/iter_0.parquet \
    --llm-path models/deepseek-prover-v2-7b \
    --output-dir checkpoints/ebm/iter_1 \
    --steps 50000 \
    --save-embeddings checkpoints/ebm/iter_1/embeddings.parquet
```

### `summary` — Trajectory statistics

```bash
cargo run --release -p prover-core -- summary --input trajectories/run.parquet
```

### `compare` — Cross-iteration comparison

```bash
cargo run --release -p prover-core -- compare \
    --results eval_results/iter_0.json eval_results/iter_1.json eval_results/iter_2.json
```

## Expert Iteration Workflow

```
For iteration i = 0, 1, 2, 3, 4:

  ┌──────────────────────────────────────┐
  │ 1. LLM Fine-tuning (Python, GPU)     │
  │    Input: tactic pairs + trajectories │
  │    Output: safetensors checkpoint     │
  ├──────────────────────────────────────┤
  │ 2. EBM Training (Rust/burn, GPU)     │
  │    Input: trajectory Parquet files    │
  │    Output: burn-rs checkpoint         │  ← skip for iter 0
  ├──────────────────────────────────────┤
  │ 3. Search / Trajectory Collection    │
  │    Input: LLM + EBM checkpoints      │  (Rust, multi-GPU)
  │    Output: trajectories/iter_i.parquet│
  ├──────────────────────────────────────┤
  │ 4. Evaluation                        │
  │    Input: LLM + EBM + miniF2F        │
  │    Output: solve rates at budgets    │
  └──────────────────────────────────────┘
```

Run the full experiment:

```bash
# 1. Prepare Mathlib training data
./scripts/prepare_data.sh

# 2. Run baseline + 5 iterations
NUM_WORKERS=64 ./scripts/run_all_iterations.sh
```

Individual iteration: `./scripts/run_iteration.sh 0`

## Configuration

### `configs/search.toml`

```toml
[search]
max_nodes = 600           # Node budget per theorem
max_depth = 50            # Max proof depth
beam_width = 8            # Top-k tactics before Lean verification
alpha = 0.5               # LLM log-prob weight
beta = 0.5                # EBM score weight
timeout_per_theorem = 600 # seconds

[lean_pool]
num_workers = 64
max_requests_per_worker = 1000
max_lifetime_secs = 1800  # Worker recycling (memory leak mitigation)
tactic_timeout_secs = 30
```

### `configs/models.toml`

```toml
[encoder]
mode = "shared"           # Use policy backbone for EBM embeddings
shared_hidden_dim = 4096  # DeepSeek-Prover-V2-7B hidden size

[energy_head]
d_hidden1 = 512
d_hidden2 = 256
dropout = 0.1
n_power_iterations = 5

[llm]
model_name = "deepseek-ai/DeepSeek-Prover-V2-7B"
max_seq_len = 2048
num_candidates = 32
temperature = 0.8
```

## Testing

```bash
# Unit tests (no external dependencies)
cargo test --workspace

# Integration tests (require Pantograph, run single-threaded)
cargo test -p lean-repl -- --ignored --test-threads=1    # ~60-90s
cargo test -p search -- --ignored --test-threads=1       # ~60s
cargo test -p prover-core -- --ignored --test-threads=1  # ~15s

# LLM integration tests (require model weights)
MODEL_PATH=/path/to/model cargo test -p policy -- --ignored --test-threads=1
```

## Key Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `candle-core` / `candle-nn` | 0.8 | LLM inference (DeepSeek-Prover-V2-7B) |
| `burn` | 0.16 | EBM training and inference (NdArray backend) |
| `tokio` | 1 | Async runtime for Lean worker pool |
| `arrow` / `parquet` | 53 | Trajectory data I/O |
| `tokenizers` | 0.21 | HuggingFace tokenizer bindings |
| `clap` | 4 | CLI argument parsing |

## License

[Apache 2.0](LICENSE)

---

For the full architecture plan, cost analysis, and implementation details, see [`docs/burn-qed_plan.md`](docs/burn-qed_plan.md).
