#!/bin/bash
# Bootstrap a cloud GPU instance for BurnQED.
#
# Installs: Rust, elan (Lean 4), Python deps, builds Pantograph and prover-core.
# Tested on: Ubuntu 22.04 with CUDA 12.x pre-installed.
#
# Usage:
#   # On a fresh cloud instance:
#   git clone https://github.com/<you>/BurnQED.git && cd BurnQED
#   bash scripts/setup_cloud.sh
#
#   # Or via SSH:
#   ssh gpu-instance 'cd BurnQED && bash scripts/setup_cloud.sh'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================"
echo "  BurnQED Cloud Setup"
echo "================================================================"

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
if ! command -v rustc &>/dev/null; then
    echo "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
else
    echo "Rust already installed: $(rustc --version)"
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

# ── Step 7: Build prover-core (release) ───────────────────────────────────
echo ""
echo "=== Step 7: Build prover-core ==="
cargo build --release -p prover-core

# ── Step 8: Download model weights ────────────────────────────────────────
echo ""
echo "=== Step 8: Model weights ==="
MODEL_DIR="${REPO_ROOT}/models/deepseek-prover-v2-7b"
if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR"/*.safetensors 2>/dev/null)" ]; then
    echo "Model weights found at ${MODEL_DIR}"
else
    echo "Model weights not found. Download them with:"
    echo ""
    echo "  # Option A: HuggingFace CLI"
    echo "  pip install huggingface-hub"
    echo "  huggingface-cli download deepseek-ai/DeepSeek-Prover-V2-7B --local-dir ${MODEL_DIR}"
    echo ""
    echo "  # Option B: git lfs"
    echo "  git clone https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B ${MODEL_DIR}"
    echo ""
    echo "Skipping smoke test until model weights are available."
    echo ""
    echo "================================================================"
    echo "  Setup complete (model weights needed for smoke test)"
    echo "================================================================"
    exit 0
fi

# ── Step 9: Smoke test ────────────────────────────────────────────────────
echo ""
echo "=== Step 9: Smoke test ==="
echo "Running search on 2 easy theorems..."

# Create a minimal test file
SMOKE_THEOREMS=$(mktemp /tmp/smoke_theorems.XXXXXX.json)
python3 -c "
import json
with open('${REPO_ROOT}/data/test_theorems.json') as f:
    data = json.load(f)
# Take first 2 theorems
subset = {'theorems': data['theorems'][:2]}
with open('${SMOKE_THEOREMS}', 'w') as f:
    json.dump(subset, f)
"

SMOKE_OUTPUT=$(mktemp /tmp/smoke_output.XXXXXX.parquet)

cargo run --release -p prover-core -- search \
    --model-path "$MODEL_DIR" \
    --theorems "$SMOKE_THEOREMS" \
    --output "$SMOKE_OUTPUT" && \
    echo "Smoke test PASSED" || \
    echo "Smoke test FAILED (check Lean/model setup)"

rm -f "$SMOKE_THEOREMS" "$SMOKE_OUTPUT"

echo ""
echo "================================================================"
echo "  Cloud setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Download model weights (if not done above)"
echo "    2. Prepare training data: python python/data/trace_mathlib.py"
echo "    3. Run first iteration: ./scripts/run_iteration.sh 0"
echo "================================================================"
