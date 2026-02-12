#!/bin/bash
# Quick end-to-end smoke test for the full pipeline (~2-3 min on A100).
#
# Validates that all components work together:
#   1. LLM-only search on 3 easy theorems (small node budget)
#   2. Train EBM from trajectory (500 steps)
#   3. Search with EBM
#   4. Compare solve rates
#
# Usage:
#   ./scripts/lean_start.sh [model_path]
#
# Prerequisites:
#   - Pantograph built (./scripts/setup_pantograph.sh)
#   - cargo build --release -p prover-core
#   - Model weights available

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${1:-${REPO_ROOT}/models/deepseek-prover-v2-7b}"
WORK_DIR="${REPO_ROOT}/lean_start_output"

# Auto-detect CUDA
CUDA_FEATURES=$(command -v nvidia-smi &>/dev/null && echo "--features cuda" || echo "")
PROVER="cargo run --release -p prover-core ${CUDA_FEATURES} --"

echo "================================================================"
echo "  BurnQED Smoke Test"
echo "================================================================"
echo "  Model: ${MODEL_PATH}"
echo "  Output: ${WORK_DIR}"
echo "================================================================"

mkdir -p "$WORK_DIR"

# ── Step 0: Create minimal inputs ────────────────────────────────────────
# 3 easy theorems: True, False→False, ∀ n, n = n
SMOKE_THEOREMS="${WORK_DIR}/smoke_theorems.json"
python3 -c "
import json
with open('${REPO_ROOT}/data/test_theorems.json') as f:
    data = json.load(f)
subset = {'theorems': data['theorems'][:3]}
with open('${SMOKE_THEOREMS}', 'w') as f:
    json.dump(subset, f, indent=2)
print(f'Using {len(subset[\"theorems\"])} theorems for smoke test')
"

# Small search config: 50 nodes, 4 candidates (instead of 600 nodes / 32 candidates)
SMOKE_CONFIG="${WORK_DIR}/smoke_search.toml"
cat > "$SMOKE_CONFIG" << 'TOML'
[search]
max_nodes = 50
max_depth = 20
num_candidates = 4
beam_width = 4
alpha = 0.5
beta = 0.5
timeout_per_theorem = 60

[lean_pool]
num_workers = 4
max_requests_per_worker = 1000
max_lifetime_secs = 1800
tactic_timeout_secs = 30
TOML

# ── Step 1: LLM-only search ──────────────────────────────────────────────
echo ""
echo "=== Step 1: LLM-only Search (3 theorems, 50 nodes) ==="
LLM_TRAJ="${WORK_DIR}/llm_only.parquet"

$PROVER search \
    --config "$SMOKE_CONFIG" \
    --model-path "$MODEL_PATH" \
    --theorems "$SMOKE_THEOREMS" \
    --output "$LLM_TRAJ"

echo ""
echo "LLM-only trajectory summary:"
$PROVER summary --input "$LLM_TRAJ"

# ── Step 2: Train EBM ────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Train EBM (500 steps) ==="
EBM_DIR="${WORK_DIR}/ebm_checkpoint"

$PROVER train-ebm \
    --trajectories "$LLM_TRAJ" \
    --llm-path "$MODEL_PATH" \
    --output-dir "$EBM_DIR" \
    --steps 500

# ── Step 3: Search with EBM ──────────────────────────────────────────────
echo ""
echo "=== Step 3: Search with EBM (3 theorems, 50 nodes) ==="
EBM_TRAJ="${WORK_DIR}/with_ebm.parquet"

$PROVER search \
    --config "$SMOKE_CONFIG" \
    --model-path "$MODEL_PATH" \
    --ebm-path "$EBM_DIR" \
    --theorems "$SMOKE_THEOREMS" \
    --output "$EBM_TRAJ"

echo ""
echo "EBM-guided trajectory summary:"
$PROVER summary --input "$EBM_TRAJ"

# ── Step 4: Compare ──────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Comparison: LLM-only vs LLM+EBM"
echo "================================================================"

echo ""
echo "LLM-only:"
$PROVER summary --input "$LLM_TRAJ"

echo ""
echo "LLM+EBM:"
$PROVER summary --input "$EBM_TRAJ"

echo ""
echo "================================================================"
echo "  Smoke test complete!"
echo "  All outputs saved to: ${WORK_DIR}"
echo "================================================================"
