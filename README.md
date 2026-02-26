```
 ██████╗ ██╗   ██╗██████╗ ███╗   ██╗       ██████╗ ███████╗██████╗
 ██╔══██╗██║   ██║██╔══██╗████╗  ██║      ██╔═══██╗██╔════╝██╔══██╗
 ██████╔╝██║   ██║██████╔╝██╔██╗ ██║█████╗██║   ██║█████╗  ██║  ██║
 ██╔══██╗██║   ██║██╔══██╗██║╚██╗██║╚════╝██║▄▄ ██║██╔══╝  ██║  ██║
 ██████╔╝╚██████╔╝██║  ██║██║ ╚████║      ╚██████╔╝███████╗██████╔╝
 ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝       ╚══▀▀═╝ ╚══════╝╚═════╝
```

**Lean 4 theorem prover combining LLM policy with Energy-Based Model value function, trained via expert iteration.**

A single DeepSeek-Prover-V2-7B backbone serves both autoregressive tactic generation (policy) and mean-pooled state embeddings (value), AlphaZero-style. A small EBM head (~11M params) is the only component trained in Rust. LLM fine-tuning runs in Python with LoRA/PEFT.

## Results

### miniF2F v1 (budget=600, 244 theorems)

| Iteration | LLM + EBM | LLM only | EBM lift |
|-----------|-----------|----------|----------|
| iter 0 (base) | -- | 76/244 (31.1%) | -- |
| iter 2 | 90/244 (36.9%) | 72/244 (29.5%) | +7.4pp |
| iter 4 | **235/244 (96.3%)** | 83/244 (34.0%) | **+62.3pp** |

**Iter 4 node distribution** (proved theorems, budget=600):

| Nodes | With EBM | Without EBM |
|-------|----------|-------------|
| Probe (0) | 49 (21%) | 49 (59%) |
| 1 node | 12 (5%) | 16 (19%) |
| 2-5 | 170 (72%) | 9 (11%) |
| 6-50 | 4 (2%) | 9 (11%) |
| **Total proved** | **235** | **83** |

EBM guides search to proofs in 2-5 node expansions (72% of proved theorems). Mean 2.7 nodes, 9.4s per theorem.

### Benchmarks

| Benchmark | Source | Theorems | Description |
|-----------|--------|----------|-------------|
| miniF2F v1 | yangky11/miniF2F | 244 test + 244 valid | Standard math competition (AMC, AIME, etc.) |
| miniF2F v2s | roozbeh-yz/miniF2F_v2 | 244 test + 244 valid | Harder variant, different problems |
| IMO-Steps lemmas | roozbeh-yz/IMO-Steps | 1,328 steps | Incremental proof steps from 13 IMO problems |
| IMO-Steps theorems | roozbeh-yz/IMO-Steps | 21 theorems | Full IMO problem proofs |

## Architecture

```
DeepSeek-Prover-V2-7B (custom inference server, sgl.Engine)
├── Policy head: autoregressive tactic generation (LM head)
└── Mean-pool hidden states → Vec<f32> via /encode endpoint
                                    │
                                    ▼
                    Energy Head (burn-rs, trainable)
                    SpectralNorm MLP: 4096 → 2048 → 1024 → 512 → 1
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
│   ├── policy/          # SGLang HTTP client for tactic generation + hidden-state extraction
│   ├── search/          # Best-first proof search engine, trait-based
│   ├── trajectory/      # Parquet I/O for search trajectories
│   ├── ebm/             # Energy-Based Model: SpectralNorm MLP, training, inference
│   ├── prover-core/     # CLI binary tying everything together
│   └── burn-contrib/    # Upstream burn-rs PR modules (stub)
├── python/
│   ├── inference_server.py  # Custom sgl.Engine server with in-process mean-pooling
│   ├── encode_embeddings.py # Direct PyTorch encoding (bypasses SGLang batch bug)
│   ├── training/            # LoRA fine-tuning (train_llm.py, export_llm.py)
│   └── data/                # Dataset downloads, benchmark conversion
├── scripts/             # Pipeline orchestration scripts
├── configs/             # search.toml, models.toml
├── data/                # HF datasets, miniF2F JSONs, SFT training data
├── vendor/Pantograph/   # Git submodule (Lean 4 REPL, Mathlib v4.26.0)
└── docs/                # Architecture plan, experiment guide, known issues
```

## Prerequisites

- **Rust** (stable, edition 2021)
- **Lean 4** via [elan](https://github.com/leanprover/elan)
- **Python 3.10+** with `sglang`, `torch`, `transformers`, `peft`, `accelerate`
- **GPU** required for LLM inference; EBM training supports both CUDA and CPU
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
./scripts/smoke_test.sh
```

For cloud GPU setup (installs Rust, elan, Python deps, builds everything):

```bash
bash scripts/setup_runpod.sh   # RunPod (RTX 4090) — recommended
bash scripts/setup_lambda.sh   # Lambda Labs (A100)
```

## Expert Iteration Pipeline

Each iteration has two phases: **train** then **search**.

```
For iteration i = 0, 1, 2, ...:

  run_iteration_train.sh i
  ┌───────────────────────────────────────────┐
  │ Step 0: Pre-training eval (train subset)  │
  │ Step 1: LLM LoRA fine-tuning (Python)     │
  │ Step 1b: Export merged safetensors        │
  │ Step 2: Restart inference server          │
  │ Step 3: Post-training eval (train subset) │
  │ Step 4: EBM training (encode + train)     │  ← skip for iter 0
  │ Step 5: miniF2F evaluation + ablation     │
  └───────────────────────────────────────────┘

  run_iteration_search.sh i
  ┌───────────────────────────────────────────┐
  │ Step 2: Proof search (trajectory collect) │
  │ Step 3: Trajectory summary                │
  └───────────────────────────────────────────┘
```

Both scripts support `START_STEP=N` to skip earlier steps.

```bash
# Full pipeline
./scripts/run_iteration_train.sh 1
./scripts/run_iteration_search.sh 1

# Skip to EBM training + eval
START_STEP=4 ./scripts/run_iteration_train.sh 1

# Skip to miniF2F eval only
START_STEP=5 ./scripts/run_iteration_train.sh 1
```

### EBM Training

EBM training uses pre-computed embeddings (direct PyTorch encoding, bypassing SGLang's broken batch hidden states). The `run_ebm_train.sh` script handles the full cycle: stop server, encode, train, restart server.

```bash
# Standalone EBM training
./scripts/run_ebm_train.sh 2

# Environment variables
EBM_STEPS=50000 EBM_LR=3e-5 LOSS_TYPE=info_nce ./scripts/run_ebm_train.sh 2
```

### Inference Server

Custom server wrapping `sgl.Engine` with in-process mean-pooling for encoding:

```bash
./scripts/start_inference_server.sh models/llm/iter_1
PORT=30000 TP=1 ./scripts/start_inference_server.sh
```

## CLI Reference

All commands are subcommands of `cargo run --release -p prover-core --`:

### `search` — Run proof search

```bash
cargo run --release -p prover-core -- search \
    --config configs/search.toml \
    --server-url http://localhost:30000 \
    --theorems data/iter1_search_theorems.json \
    --output trajectories/iter_1.parquet \
    --ebm-path checkpoints/ebm/iter_1 \  # optional: EBM value guidance
    --concurrency 8 \
    --num-workers 8 \
    --imports Mathlib
```

### `eval` — Multi-budget evaluation

```bash
cargo run --release -p prover-core -- eval \
    --config configs/search.toml \
    --server-url http://localhost:30000 \
    --theorems data/minif2f_test.json \
    --budgets 600 \
    --output eval_results/iter_1.json \
    --ebm-path checkpoints/ebm/iter_1 \
    --num-candidates 16 \
    --imports Mathlib
```

### `train-ebm` — Train EBM from trajectories

```bash
cargo run --release -p prover-core -- train-ebm \
    --trajectories trajectories/iter_0.parquet trajectories/iter_1.parquet \
    --server-url http://localhost:30000 \
    --output-dir checkpoints/ebm/iter_2 \
    --steps 50000 \
    --embeddings-cache checkpoints/ebm/iter_2/embeddings.parquet \
    --loss-type info_nce
```

### `summary` / `compare`

```bash
cargo run --release -p prover-core -- summary --input trajectories/iter_1.parquet
cargo run --release -p prover-core -- compare \
    --results eval_results/iter_0.json eval_results/iter_1.json
```

## Configuration

### `configs/search.toml`

```toml
[search]
max_nodes = 100               # Node budget per theorem
max_depth = 50                # Max proof depth
num_candidates = 8            # Tactics generated per node
alpha = 0.5                   # LLM log-prob weight
beta = 0.5                    # EBM score weight
timeout_per_theorem = 120     # seconds
harvest_siblings = true       # Mine sibling states after proof found
batch_generate_size = 32      # Nodes expanded per generation batch
batch_encode_size = 8         # Max states per EBM encode batch

[lean_pool]
num_workers = 8
max_requests_per_worker = 1000
max_lifetime_secs = 1800      # Worker recycling
tactic_timeout_secs = 30
```

## Testing

```bash
# Unit tests (no external dependencies)
cargo test --workspace

# Integration tests (require Pantograph, run single-threaded)
cargo test -p lean-repl -- --ignored --test-threads=1    # ~60-90s
cargo test -p search -- --ignored --test-threads=1       # ~60s
cargo test -p prover-core -- --ignored --test-threads=1  # ~15s

# Policy integration tests (require running SGLang server)
cargo test -p policy -- --ignored --test-threads=1
```

## Key Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `burn` | 0.16 | EBM training and inference |
| `reqwest` | 0.12 | HTTP client for inference server |
| `tokio` | 1 | Async runtime for Lean worker pool |
| `arrow` / `parquet` | 53 | Trajectory data I/O |
| `clap` | 4 | CLI argument parsing |

## Known Issues

- **SGLang batch hidden states**: `return_hidden_states=True` is broken in batch mode (SGLang #8066). Workaround: `python/encode_embeddings.py` uses direct PyTorch encoding for training data. Search-time encoding uses sequential calls (one at a time), which is correct. See `docs/encoding_bug.md`.
- **Dataset version gaps**: Training datasets (Lean Workbook v4.8, Goedel v4.9, NuminaMath v4.15) lag behind our Pantograph v4.26. Mathlib lemma renames may break some tactics. Pantograph validation (tasks 0.3d-f) measures actual compatibility. See `docs/datasets.md`.

## License

[Apache 2.0](LICENSE)

---

For the full architecture plan, see [`docs/burn-qed_plan.md`](docs/burn-qed_plan.md).
