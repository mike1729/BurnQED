#!/bin/bash
# Bootstrap a Lambda Labs GPU instance for BurnQED.
#
# Installs: Rust, elan (Lean 4), Python deps, builds Pantograph and prover-core.
# Tested on: Lambda A100 40GB (Ubuntu 22.04, CUDA 12.x pre-installed).
#
# Usage:
#   git clone https://github.com/<you>/BurnQED.git && cd BurnQED
#   bash scripts/setup_lambda.sh
#
#   # With persistent storage:
#   PERSIST_DIR=/home/ubuntu/burnqed-data bash scripts/setup_lambda.sh
#
# See also: scripts/setup_runpod.sh for RunPod (RTX 4090) instances.
#
# Persistent Storage:
#   Set PERSIST_DIR to a mounted persistent volume (e.g., NFS, EBS).
#   The script will create symlinks from the repo to $PERSIST_DIR for:
#     models/, trajectories/, checkpoints/, baselines/, eval_results/, data/, logs/
#   Model weights should be pre-placed at $PERSIST_DIR/models/deepseek-prover-v2-7b/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PERSIST_DIR="${PERSIST_DIR:-}"

echo "================================================================"
echo "  BurnQED Setup — Lambda Labs"
echo "================================================================"

# ── Step 0: Persistent storage symlinks ─────────────────────────────────
if [ -n "$PERSIST_DIR" ]; then
    echo ""
    echo "=== Step 0: Persistent storage symlinks ==="
    echo "  Persist dir: ${PERSIST_DIR}"

    if [ ! -d "$PERSIST_DIR" ]; then
        echo "Creating persistent directory: ${PERSIST_DIR}"
        mkdir -p "$PERSIST_DIR"
    fi

    # Directories to symlink to persistent storage
    PERSIST_DIRS=(models trajectories checkpoints baselines eval_results data logs)

    for dir in "${PERSIST_DIRS[@]}"; do
        persist_path="${PERSIST_DIR}/${dir}"
        repo_path="${REPO_ROOT}/${dir}"

        # Create the persistent directory if it doesn't exist
        mkdir -p "$persist_path"

        if [ -L "$repo_path" ]; then
            # Already a symlink — verify it points to the right place
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
            # Real directory exists — move contents to persistent, then symlink
            if [ -n "$(ls -A "$repo_path" 2>/dev/null)" ]; then
                echo "  ${dir}/ → migrating contents to persistent storage"
                cp -a "$repo_path"/* "$persist_path"/ 2>/dev/null || true
            fi
            rm -rf "$repo_path"
            ln -s "$persist_path" "$repo_path"
            echo "  ${dir}/ → linked"
        else
            # Nothing exists yet — just create the symlink
            ln -s "$persist_path" "$repo_path"
            echo "  ${dir}/ → linked (new)"
        fi
    done

    # Check if model weights are pre-placed
    MODEL_CHECK="${PERSIST_DIR}/models/deepseek-prover-v2-7b"
    if [ -d "$MODEL_CHECK" ] && [ -n "$(ls -A "$MODEL_CHECK"/*.safetensors 2>/dev/null)" ]; then
        echo ""
        echo "  Model weights found at ${MODEL_CHECK}"
    else
        echo ""
        echo "  WARNING: Model weights not found at ${MODEL_CHECK}"
        echo "  Place them there before running baseline/iteration scripts."
    fi
else
    echo ""
    echo "  (No PERSIST_DIR set — using local storage)"
    echo "  Tip: Set PERSIST_DIR=/path/to/mount for persistent storage"
fi

# ── Step 1: System packages ───────────────────────────────────────────────
echo ""
echo "=== Step 1: System packages ==="
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        build-essential git curl wget pkg-config \
        libssl-dev libclang-dev cmake \
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
if [ ! -d "${REPO_ROOT}/.venv" ]; then
    python3 -m venv "${REPO_ROOT}/.venv"
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
pip install --upgrade pip
pip install -r "${REPO_ROOT}/python/requirements.txt"
pip install "sglang[all]"

# Configure accelerate for single-GPU (needed for LLM fine-tuning)
echo "Configuring accelerate for single GPU..."
accelerate config default 2>/dev/null || python -m accelerate config default 2>/dev/null || true

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
echo "  Lambda setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Download model weights (if not done above)"
echo "    2. Start SGLang server: ./scripts/start_sglang.sh"
echo "    3. Prepare training data: ./scripts/prepare_data.sh"
echo "    4. Run baseline evaluation: ./scripts/run_baseline.sh"
echo "    5. Run first iteration: ./scripts/run_iteration.sh 0"
echo "================================================================"
