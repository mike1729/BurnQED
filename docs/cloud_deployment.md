# BurnQED Cloud Deployment Guide

## Provider Recommendations

For a 7B model with QLoRA, an A100 is overkill. An RTX 4090 (24GB) handles both
QLoRA training (~12GB) and fp16 inference (~14GB) at a fraction of the cost.

| Provider | GPU | Cost/hr | VRAM | Best For |
|----------|-----|---------|------|----------|
| **RunPod** (recommended) | RTX 4090 | $0.32–0.44 | 24GB | 7B QLoRA + inference |
| **TensorDock** | RTX 4090 | $0.30–0.40 | 24GB | Long training jobs |
| **Lambda Labs** | A100 40GB | $1.29–1.50 | 40GB | If you need >24GB VRAM |
| **Vast.ai** | RTX 3090 | $0.15–0.25 | 24GB | Cheapest; less reliable |

**Recommended**: RunPod RTX 4090 with a Network Volume. ~3× cheaper than A100
with nearly identical performance for 7B models. Use spot instances for search
(auto-resumes), on-demand for fine-tuning.

**Important**: Ensure the instance has 8+ vCPUs and 32GB+ RAM for Lean workers.

**RunPod note**: Community Cloud pods do NOT support persistent Network Volumes.
You must use Secure Cloud (spot or on-demand) to attach a Network Volume.

## Hardware Requirements

### LLM Fine-Tuning (QLoRA)

- **GPU**: 1x RTX 4090 24GB (recommended) or 1x A100 40GB
  - 7B model in 4-bit: ~4GB VRAM
  - Activations + optimizer states: ~8-12GB with gradient checkpointing
  - Fits comfortably on 24GB
- **RAM**: 32GB+ (model loading + data)
- **Disk**: 50GB (model weights + checkpoints)
- **Time**: ~2-3 hours for 1500 steps on 246K examples

### Proof Search (Lean + LLM Inference)

- **GPU**: 1x RTX 4090 24GB (fp16 inference ~14GB)
- **CPU**: 8+ cores (for Lean worker pool, 6 concurrent Pantograph processes)
- **RAM**: 32GB+ (Lean processes use ~2GB each)
- **Disk**: 30GB (model weights + Lean/Mathlib toolchain)
- **Time**: ~2-4 hours per 2000 theorems (100 nodes/theorem budget)
- Note: 4090 has lower memory bandwidth (1 TB/s vs A100's 2 TB/s), so
  autoregressive generation is ~1.5× slower per token, but the cost savings
  more than compensate

### EBM Training

- **GPU**: 1x any GPU with 8GB+ VRAM (only ~5M params)
  - Even a T4 or RTX 3090 works
  - Most time is spent on embedding precomputation (LLM forward passes)
- **RAM**: 16GB+
- **Time**: ~30 min for 50K steps with precomputed embeddings

### Combined (One Iteration)

- **Minimum**: 1x RTX 4090 24GB, 8 CPU cores, 32GB RAM, 100GB disk
- **Recommended**: 1x RTX 4090 24GB, 16 CPU cores, 64GB RAM, 200GB disk
- **Time per iteration**: ~6-10 hours (4090), ~4-7 hours (A100)

## Instance Setup Checklist

### 1. Base System (Ubuntu 22.04 + CUDA)

Most cloud providers offer Ubuntu 22.04 images with CUDA pre-installed. Verify:

```bash
nvidia-smi            # Should show GPU(s) and CUDA version
nvcc --version        # CUDA compiler (12.x recommended)
```

### 2. Automated Setup

```bash
# Clone repo and run setup
git clone https://github.com/<you>/BurnQED.git
cd BurnQED

# RunPod (RTX 4090) — recommended
bash scripts/setup_runpod.sh

# Lambda Labs (A100)
bash scripts/setup_lambda.sh
```

This installs: Rust, elan (Lean 4), Python deps, builds Pantograph and prover-core.

### 3. Manual Setup (if automated fails)

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Lean 4 (elan)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
export PATH="$HOME/.elan/bin:$PATH"

# Python
python3 -m venv .venv && source .venv/bin/activate
pip install -r python/requirements.txt

# Submodules + Pantograph
git submodule update --init --recursive
bash scripts/setup_pantograph.sh

# Build
cargo build --release -p prover-core
```

### 4. Model Weights

```bash
# Option A: HuggingFace Hub (recommended)
pip install huggingface-hub
huggingface-cli download deepseek-ai/DeepSeek-Prover-V2-7B \
    --local-dir models/deepseek-prover-v2-7b

# Option B: From a persistent volume (faster on re-provision)
ln -s /mnt/persistent/models/deepseek-prover-v2-7b models/deepseek-prover-v2-7b
```

### 5. Training Data

```bash
# Option A: Download pre-traced data (default, ~5 min)
python python/data/trace_mathlib.py --output-dir data/

# Option B: Local LeanDojo trace (hours, requires 100GB+ disk)
python python/data/trace_mathlib.py --trace --output-dir data/

# Format for training
python python/data/prepare_tactic_pairs.py \
    --input data/tactic_pairs/train.jsonl \
    --output data/tactic_pairs/train_formatted.jsonl
python python/data/prepare_tactic_pairs.py \
    --input data/tactic_pairs/val.jsonl \
    --output data/tactic_pairs/val_formatted.jsonl
```

## Docker / Snapshot Workflow

### Creating a Snapshot

After completing setup, create a snapshot/image for fast re-provisioning:

```bash
# Verify everything works
cargo run --release -p prover-core -- search --dry-run \
    --model-path models/deepseek-prover-v2-7b \
    --theorems data/test_theorems.json \
    --output /tmp/test.parquet

# Clean up temporary files
cargo clean
rm -rf target/debug  # Keep release build
```

Then create a snapshot via your cloud provider's console/API.

### Docker (Alternative)

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    build-essential git curl wget pkg-config \
    libssl-dev libclang-dev cmake \
    python3 python3-pip python3-venv

# Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Lean 4
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
ENV PATH="/root/.elan/bin:${PATH}"

WORKDIR /app
COPY . .

RUN pip install -r python/requirements.txt
RUN git submodule update --init --recursive
RUN cd vendor/Pantograph && lake build
RUN cargo build --release -p prover-core
```

Build and push:
```bash
docker build -t burnqed:latest .
# Push to your registry
```

## Persistent Storage

### What to Persist

Mount a persistent volume at `/mnt/persistent/` and symlink:

```bash
# Model weights (~15GB, download once)
/mnt/persistent/models/deepseek-prover-v2-7b/

# Checkpoints (grow each iteration, ~20GB per iter)
/mnt/persistent/checkpoints/

# Trajectories (~5-10GB per iteration)
/mnt/persistent/trajectories/

# Evaluation results (~1MB per iteration)
/mnt/persistent/eval_results/
```

Setup symlinks:
```bash
ln -s /mnt/persistent/models models
ln -s /mnt/persistent/checkpoints checkpoints
ln -s /mnt/persistent/trajectories trajectories
ln -s /mnt/persistent/eval_results eval_results
```

### Volume Sizing

| Component | Size Per Iteration | 5 Iterations |
|-----------|-------------------|--------------|
| LLM checkpoints (LoRA) | ~500MB | 2.5GB |
| LLM merged (safetensors) | ~15GB | 75GB |
| EBM checkpoints | ~50MB | 250MB |
| Trajectories | ~5GB | 25GB |
| Embedding cache | ~2GB | 2GB (recomputed) |
| **Total** | ~22GB | **~105GB** |

**Recommended volume size**: 200GB (allows headroom for logs, temp files).

To save space, delete merged safetensors from previous iterations (keep LoRA adapters for re-merge):
```bash
# Keep only latest merged model
rm -rf models/llm/iter_{0..3}/
```

## Cost Estimate

### Per Iteration

| Step | GPU Hours | 4090 on-demand ($0.59) | 4090 spot ($0.35) | A100 ($1.29) |
|------|-----------|------------------------|-------------------|--------------|
| LLM fine-tune (iter 0: 1500 steps) | 2.5-3h | $1.48-1.77 | $0.90-1.05 | $3.75-4.50 |
| LLM fine-tune (iter N: 800 steps) | 1-1.5h | $0.59-0.89 | $0.35-0.53 | $1.29-1.95 |
| LLM export (merge + save) | 0.2h | $0.12 | $0.07 | $0.26 |
| EBM training (2000 steps) | 1h | $0.59 | $0.35 | $1.29 |
| Search (2000 theorems) | 3-5h | $1.77-2.95 | $1.05-1.75 | $3.87-6.45 |
| Evaluation (miniF2F, budget 600) | 1-2h | $0.59-1.18 | $0.35-0.70 | $1.29-2.58 |
| **Iteration 0 total** | ~10h | **~$5.90** | **~$3.50** | **~$13** |
| **Iteration N total** | ~8h | **~$4.70** | **~$2.80** | **~$10** |

### Full 5-Iteration Run

| Component | 4090 on-demand | 4090 spot | A100 |
|-----------|----------------|-----------|------|
| Iteration 0 | ~$5.90 | ~$3.50 | ~$13 |
| Iterations 1-4 | ~$19 | ~$11 | ~$40 |
| Persistent storage (200GB, 1 week) | ~$3-5 | ~$3-5 | ~$5-10 |
| **Total** | **~$28-30** | **~$18-20** | **~$58-63** |

### Budget Tips

- **Recommended**: RunPod Secure Cloud on-demand 4090 ($0.59/hr) — reliable, persistent volumes, ~2× cheaper than A100
- Spot instances ($0.32-0.44/hr) are cheaper but get interrupted frequently on RunPod; only viable for search (auto-resumes via `--resume-from`), not for fine-tuning
- Pre-compute embeddings once (`--save-embeddings`), reuse for EBM training
- Stop the pod when idle — persistent Network Volume costs ~$0.10/GB/month (storage only)

## Monitoring

### tmux Session Layout

```bash
tmux new-session -s burnqed

# Window 0: Main iteration
./scripts/run_iteration.sh 0

# Window 1: GPU monitoring
watch -n 1 nvidia-smi

# Window 2: Log tailing
tail -f logs/iter_0.log

# Window 3: Disk usage
watch -n 60 'df -h /mnt/persistent && du -sh checkpoints/ trajectories/ models/'
```

### Key Metrics to Watch

- **GPU utilization**: Should be >90% during LLM fine-tune/inference, low during Lean verification
- **GPU memory**: Fine-tuning peaks at ~30-40GB; search at ~15-20GB
- **Disk usage**: Trajectories grow ~5GB/iteration; delete old merged models if tight
- **Lean worker health**: `RUST_LOG=info` shows per-theorem timing; >60s/theorem may indicate Lean issues

### Alerts

```bash
# Simple disk space alert (add to crontab)
*/10 * * * * df /mnt/persistent | awk 'NR==2{if($5+0>90) print "DISK ALERT: "$5" used"}' | mail -s "BurnQED Disk Alert" you@email.com
```

## Spot Instance Resilience

BurnQED is designed for graceful recovery from spot instance interruptions:

### Auto-Save During Search

The search command auto-saves partial results every 50 theorems via `flush_partial()`. On SIGTERM (spot preemption), the partial Parquet file is already on disk.

### Resume Protocol

```bash
# After re-provisioning:
cd BurnQED

# Check what was saved
ls -la trajectories/
cargo run --release -p prover-core -- summary --input trajectories/iter_0.parquet

# Resume search from partial results
./scripts/resume_search.sh 0

# Or manually:
cargo run --release -p prover-core -- search \
    --model-path models/llm/iter_0 \
    --theorems data/theorem_index.json \
    --output trajectories/iter_0.parquet \
    --resume-from trajectories/iter_0.parquet
```

### SIGTERM Handling

The Rust search loop handles SIGTERM (sent by cloud providers before spot termination):
1. Sets a cancellation flag via `tokio::signal::ctrl_c()`
2. Current theorem search completes (or times out)
3. Partial results are flushed to Parquet
4. Process exits cleanly

**Tip**: Most cloud providers give 30-120 seconds warning before termination. A single theorem search typically takes 10-30 seconds, so the current theorem usually completes.

### What Needs Re-Running

| Component | Resumable? | Notes |
|-----------|-----------|-------|
| LLM fine-tuning | Partial | HuggingFace Trainer saves checkpoints every 500 steps; resume manually |
| LLM export | No | Fast (~5 min), just re-run |
| EBM training | Yes | `--resume-from` loads checkpoint |
| Search | Yes | `--resume-from` skips done theorems |
| Evaluation | No | Fast per-budget; re-run entirely |
