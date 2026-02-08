# BurnQED Cloud Deployment Guide

## Provider Recommendations

| Provider | GPU Options | Spot Price (A100 80GB) | On-Demand | Notes |
|----------|------------|----------------------|-----------|-------|
| **Lambda Labs** | A100 80GB, H100 | ~$1.10/hr | $1.99/hr | Best for ML; pre-installed CUDA/PyTorch |
| **RunPod** | A100, H100, A6000 | ~$1.20/hr | $1.64/hr | Serverless + pod options; community templates |
| **Vast.ai** | Mixed (A100, 3090, etc.) | ~$0.80/hr | Varies | Cheapest spot; less reliable; good for search |
| **AWS (p4d)** | A100 40GB x8 | ~$8/hr | $32/hr | Most reliable; overkill for single-GPU tasks |
| **GCP (a2-highgpu)** | A100 40/80GB | ~$3/hr | $5/hr | Good Kubernetes integration |

**Recommended setup**: Lambda Labs or RunPod for simplicity. Use spot/interruptible instances for search (which can resume), on-demand for LLM fine-tuning (harder to checkpoint mid-epoch).

## Hardware Requirements

### LLM Fine-Tuning (QLoRA)

- **GPU**: 1x A100 80GB (preferred) or 1x A100 40GB
  - 7B model in 4-bit: ~4GB VRAM
  - Activations + optimizer states: ~20-30GB
  - With gradient checkpointing: fits on 40GB
- **RAM**: 64GB+ (model loading + data)
- **Disk**: 50GB (model weights + checkpoints)
- **Time**: ~2-4 hours per epoch on full Mathlib tactic pairs (~500K examples)

For multi-GPU fine-tuning (faster):
- 4x A100 80GB with `accelerate` FSDP or DeepSpeed ZeRO-3
- ~30 min per epoch

### Proof Search (Lean + LLM Inference)

- **GPU**: 1x A100 40GB (LLM inference in fp16/bf16)
- **CPU**: 8+ cores (for Lean worker pool, 4-8 concurrent Pantograph processes)
- **RAM**: 32GB+ (Lean processes use ~2GB each)
- **Disk**: 30GB (model weights + Lean/Mathlib toolchain)
- **Time**: ~1-2 hours per 1000 theorems (600 nodes/theorem budget)

### EBM Training

- **GPU**: 1x any GPU with 8GB+ VRAM (only ~5M params)
  - Even a T4 or RTX 3090 works
  - Most time is spent on embedding precomputation (LLM forward passes)
- **RAM**: 16GB+
- **Time**: ~30 min for 50K steps with precomputed embeddings

### Combined (One Iteration)

- **Minimum**: 1x A100 40GB, 8 CPU cores, 64GB RAM, 100GB disk
- **Recommended**: 1x A100 80GB, 16 CPU cores, 128GB RAM, 200GB disk
- **Time per iteration**: ~6-10 hours

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
bash scripts/setup_cloud.sh
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
# Option A: Full Mathlib trace (hours, requires 100GB+ disk)
python python/data/trace_mathlib.py --output-dir data/

# Option B: Pre-traced fallback (faster)
python python/data/trace_mathlib.py --fallback --output-dir data/

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

### Per Iteration (A100 80GB spot @ $1.10/hr)

| Step | GPU Hours | Cost |
|------|-----------|------|
| LLM fine-tune (iter 0: 3 epochs) | 6-8h | $7-9 |
| LLM fine-tune (iter N: 1 epoch) | 2-3h | $2-3 |
| LLM export (merge + save) | 0.5h | $0.55 |
| EBM training (50K steps) | 0.5h | $0.55 |
| Search (1000 theorems) | 2-4h | $2-4 |
| Evaluation (miniF2F, 3 budgets) | 1-2h | $1-2 |
| **Iteration 0 total** | ~12h | **~$13** |
| **Iteration N total** | ~8h | **~$9** |

### Full 5-Iteration Run

| Component | Cost |
|-----------|------|
| Iteration 0 | ~$13 |
| Iterations 1-4 | ~$36 |
| Persistent storage (200GB, 1 week) | ~$5-10 |
| **Total** | **~$55-60** |

### Budget Tips

- Use spot instances for search (auto-resumes via `--resume-from`)
- Use on-demand for fine-tuning (harder to resume mid-training)
- Pre-compute embeddings once (`--save-embeddings`), reuse for EBM training
- A smaller model (1.3B) reduces costs 5x but with lower prove rates

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
