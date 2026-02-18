#!/bin/bash
# Bootstrap a RunPod GPU instance for BurnQED.
#
# Optimized for RTX 4090 (24GB VRAM) — the sweet spot for 7B QLoRA.
# Also works on RTX 3090, A100, or any CUDA GPU with >= 24GB VRAM.
#
# Usage:
#   # On a fresh RunPod pod (PyTorch template recommended):
#   git clone https://github.com/<you>/BurnQED.git && cd BurnQED
#   bash scripts/setup_runpod.sh
#
# RunPod Network Volume:
#   Create a Network Volume in the RunPod dashboard and attach it to your pod.
#   It mounts at /workspace/ (or /runpod-volume/ in some templates).
#   The script auto-detects /workspace/ as PERSIST_DIR if present.
#
#   Pre-place model weights at /workspace/models/deepseek-prover-v2-7b/
#   to avoid re-downloading on pod restarts.
#
# Spot Instances:
#   Safe to use — the pipeline auto-checkpoints:
#     - LoRA saves every 500 training steps
#     - Trajectory auto-saves every 50 theorems (flush_partial)
#     - --resume-from flag for search resumption
#   Attach a Network Volume so checkpoints survive pod termination.
#
# See also: scripts/setup_lambda.sh for Lambda Labs (A100) instances.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Auto-detect RunPod Network Volume
if [ -z "${PERSIST_DIR:-}" ]; then
    if [ -d "/workspace" ] && [ -w "/workspace" ]; then
        PERSIST_DIR="/workspace"
    fi
fi
PERSIST_DIR="${PERSIST_DIR:-}"

# Detect GPU type for VRAM-aware configuration
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")

echo "================================================================"
echo "  BurnQED Setup — RunPod"
echo "================================================================"
echo "  GPU:        ${GPU_NAME}"
echo "  VRAM:       ${GPU_MEM_MB} MB"
if [ -n "$PERSIST_DIR" ]; then
    echo "  Storage:    ${PERSIST_DIR} (persistent)"
else
    echo "  Storage:    local (no Network Volume detected)"
fi
echo "================================================================"

# Warn if VRAM is tight
if [ "$GPU_MEM_MB" -gt 0 ] && [ "$GPU_MEM_MB" -lt 20000 ]; then
    echo ""
    echo "  WARNING: ${GPU_MEM_MB}MB VRAM detected. Minimum 24GB recommended."
    echo "  QLoRA training needs ~12GB, SGLang inference needs ~14GB."
    echo ""
fi

# ── Step 0: Persistent storage symlinks ─────────────────────────────────
if [ -n "$PERSIST_DIR" ]; then
    echo ""
    echo "=== Step 0: Persistent storage symlinks ==="
    echo "  Persist dir: ${PERSIST_DIR}"

    if [ ! -d "$PERSIST_DIR" ]; then
        echo "Creating persistent directory: ${PERSIST_DIR}"
        mkdir -p "$PERSIST_DIR"
    fi

    PERSIST_DIRS=(models trajectories checkpoints baselines eval_results data logs)

    for dir in "${PERSIST_DIRS[@]}"; do
        persist_path="${PERSIST_DIR}/${dir}"
        repo_path="${REPO_ROOT}/${dir}"

        mkdir -p "$persist_path"

        if [ -L "$repo_path" ]; then
            current_target=$(readlink -f "$repo_path")
            expected_target=$(readlink -f "$persist_path")
            if [ "$current_target" = "$expected_target" ]; then
                echo "  ${dir}/ → already linked"
            else
                echo "  ${dir}/ → re-linking (was: ${current_target})"
                rm "$repo_path"
                ln -s "$persist_path" "$repo_path"
            fi
        elif [ -d "$repo_path" ]; then
            if [ -n "$(ls -A "$repo_path" 2>/dev/null)" ]; then
                echo "  ${dir}/ → migrating contents to persistent storage"
                cp -a "$repo_path"/* "$persist_path"/ 2>/dev/null || true
            fi
            rm -rf "$repo_path"
            ln -s "$persist_path" "$repo_path"
            echo "  ${dir}/ → linked"
        else
            ln -s "$persist_path" "$repo_path"
            echo "  ${dir}/ → linked (new)"
        fi
    done

    # Symlink HuggingFace cache to workspace (avoids 13GB+ on overlay)
    HF_CACHE_PERSIST="${PERSIST_DIR}/.cache/huggingface"
    HF_CACHE_LOCAL="$HOME/.cache/huggingface"
    mkdir -p "$(dirname "$HF_CACHE_PERSIST")"
    if [ -L "$HF_CACHE_LOCAL" ]; then
        echo "  ~/.cache/huggingface → already linked"
    elif [ -d "$HF_CACHE_LOCAL" ]; then
        echo "  ~/.cache/huggingface → migrating to persistent storage"
        mkdir -p "$HF_CACHE_PERSIST"
        cp -a "$HF_CACHE_LOCAL"/* "$HF_CACHE_PERSIST"/ 2>/dev/null || true
        rm -rf "$HF_CACHE_LOCAL"
        ln -s "$HF_CACHE_PERSIST" "$HF_CACHE_LOCAL"
        echo "  ~/.cache/huggingface → linked"
    else
        mkdir -p "$HF_CACHE_PERSIST"
        ln -s "$HF_CACHE_PERSIST" "$HF_CACHE_LOCAL"
        echo "  ~/.cache/huggingface → linked (new)"
    fi

    MODEL_CHECK="${PERSIST_DIR}/models/deepseek-prover-v2-7b"
    if [ -d "$MODEL_CHECK" ] && [ -n "$(ls -A "$MODEL_CHECK"/*.safetensors 2>/dev/null)" ]; then
        echo ""
        echo "  Model weights found at ${MODEL_CHECK}"
    else
        echo ""
        echo "  Model weights not found at ${MODEL_CHECK}"
        echo "  They'll be downloaded in Step 8."
    fi
else
    echo ""
    echo "  (No PERSIST_DIR set — using local storage)"
    echo "  Tip: Attach a RunPod Network Volume for persistent storage"
fi

# ── Step 1: System packages ───────────────────────────────────────────────
echo ""
echo "=== Step 1: System packages ==="
if command -v apt-get &>/dev/null; then
    # RunPod templates have most packages, but ensure build deps are present
    apt-get update -qq 2>/dev/null || sudo apt-get update -qq
    apt-get install -y -qq \
        build-essential git curl wget pkg-config \
        libssl-dev libclang-dev cmake libnuma1 \
        python3 python3-pip python3-venv \
        2>/dev/null || \
    sudo apt-get install -y -qq \
        build-essential git curl wget pkg-config \
        libssl-dev libclang-dev cmake libnuma1 \
        python3 python3-pip python3-venv
fi

# ── Step 2: Rust ──────────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Rust ==="
MIN_CARGO_VERSION="1.85"

needs_rust_install() {
    if ! command -v cargo &>/dev/null; then
        return 0
    fi
    local ver
    ver=$(cargo --version | grep -oP '\d+\.\d+')
    if [ "$(printf '%s\n' "$MIN_CARGO_VERSION" "$ver" | sort -V | head -1)" != "$MIN_CARGO_VERSION" ]; then
        echo "Cargo ${ver} is too old (need >= ${MIN_CARGO_VERSION})"
        return 0
    fi
    return 1
}

if needs_rust_install; then
    echo "Installing/upgrading Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
else
    echo "Rust already installed: $(cargo --version)"
fi

# ── Step 3: Lean 4 (elan) ────────────────────────────────────────────────
echo ""
echo "=== Step 3: Lean 4 (elan) ==="
if ! command -v elan &>/dev/null; then
    echo "Installing elan..."
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
    export PATH="$HOME/.elan/bin:$PATH"
else
    echo "elan already installed: $(elan --version)"
fi

# ── Step 4: Git submodules ────────────────────────────────────────────────
echo ""
echo "=== Step 4: Git submodules ==="
git submodule update --init --recursive

# ── Step 5: Build Pantograph ──────────────────────────────────────────────
echo ""
echo "=== Step 5: Build Pantograph ==="
bash "${REPO_ROOT}/scripts/setup_pantograph.sh"

# ── Step 6: Python environment ────────────────────────────────────────────
echo ""
echo "=== Step 6: Python environment ==="

# RunPod templates often have torch pre-installed (conda or system).
# Prefer using the existing environment to avoid re-downloading torch.
if python3 -c "import torch; print(f'PyTorch {torch.__version__} (CUDA {torch.version.cuda})')" 2>/dev/null; then
    echo "Using existing Python environment with PyTorch"
    pip install --upgrade pip
    pip install -r "${REPO_ROOT}/python/requirements.txt"
    pip install "sglang[all]"
else
    echo "No PyTorch found — creating venv and installing from scratch"
    if [ ! -d "${REPO_ROOT}/.venv" ]; then
        python3 -m venv "${REPO_ROOT}/.venv"
    fi
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
    pip install --upgrade pip
    pip install -r "${REPO_ROOT}/python/requirements.txt"
    pip install "sglang[all]"
fi

# Configure accelerate for single-GPU
echo "Configuring accelerate for single GPU..."
accelerate config default 2>/dev/null || python -m accelerate config default 2>/dev/null || true

# Purge pip cache to save overlay disk space (~5GB)
echo "Purging pip cache..."
pip cache purge 2>/dev/null || true

# ── Step 7: Build prover-core (release) ───────────────────────────────────
echo ""
echo "=== Step 7: Build prover-core ==="
cargo build --release -p prover-core

# ── Step 8: Verify model weights ──────────────────────────────────────────
echo ""
echo "=== Step 8: Model weights ==="
MODEL_DIR="${REPO_ROOT}/models/deepseek-prover-v2-7b"
if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR"/*.safetensors 2>/dev/null)" ]; then
    echo "Model weights found at ${MODEL_DIR}"
else
    echo "Model weights not found at ${MODEL_DIR}"
    echo "Transfer them from another instance using:"
    echo "  bash scripts/migrate_to_runpod.sh <source_host> <this_host>"
fi

# ── Step 9: Shell environment ─────────────────────────────────────────────
echo ""
echo "=== Step 9: Persist PATH in .bashrc ==="
BASHRC="$HOME/.bashrc"
for line in 'source "$HOME/.cargo/env"' 'export PATH="$HOME/.elan/bin:$PATH"'; do
    if ! grep -qF "$line" "$BASHRC" 2>/dev/null; then
        echo "$line" >> "$BASHRC"
        echo "  Added: $line"
    else
        echo "  Already present: $line"
    fi
done

# ── Step 10: Smoke test ──────────────────────────────────────────────────
echo ""
echo "=== Step 10: Smoke test ==="
echo "To run a quick end-to-end smoke test:"
echo "  ./scripts/smoke_test.sh"

echo ""
echo "================================================================"
echo "  RunPod setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Start inference server: ./scripts/start_inference_server.sh"
echo "    2. Prepare training data: ./scripts/prepare_data.sh"
echo "    3. Run baseline evaluation: ./scripts/run_baseline.sh"
echo "    4. Run first iteration: ./scripts/run_iteration.sh 0"
echo ""
echo "  Spot instance tips:"
echo "    - Checkpoints auto-save (LoRA: every 500 steps, search: every 50 theorems)"
echo "    - Resume interrupted search: ./scripts/resume_search.sh N"
echo "    - Use Network Volume to persist data across pod restarts"
echo "================================================================"
