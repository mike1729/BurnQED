#!/bin/bash
# Quick end-to-end validation on a small theorem set.
#
# Tests the full pipeline without requiring a full Mathlib trace:
#   1. LLM-only search on a small set
#   2. Train EBM from trajectory
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
PROVER="cargo run --release -p prover-core --"

echo "================================================================"
echo "  BurnQED Quick Start Validation"
echo "================================================================"
echo "  Model: ${MODEL_PATH}"
echo "  Output: ${WORK_DIR}"
echo "================================================================"

mkdir -p "$WORK_DIR"

# ── Step 0: Choose theorem set ────────────────────────────────────────────
THEOREMS="${REPO_ROOT}/data/test_theorems.json"
if [ ! -f "$THEOREMS" ]; then
    echo "ERROR: test_theorems.json not found at ${THEOREMS}"
    exit 1
fi

THEOREM_COUNT=$(python3 -c "import json; print(len(json.load(open('${THEOREMS}'))['theorems']))")
echo "Using ${THEOREM_COUNT} theorems from ${THEOREMS}"

# If a full theorem_index.json exists, use a 500-theorem subset instead
FULL_INDEX="${REPO_ROOT}/data/theorem_index.json"
if [ -f "$FULL_INDEX" ]; then
    echo "Found full theorem index; creating 500-theorem subset..."
    python3 -c "
import json, random
random.seed(42)
with open('${FULL_INDEX}') as f:
    data = json.load(f)
theorems = data['theorems']
subset = random.sample(theorems, min(500, len(theorems)))
with open('${WORK_DIR}/subset_theorems.json', 'w') as f:
    json.dump({'theorems': subset}, f, indent=2)
print(f'Subset: {len(subset)} theorems')
"
    THEOREMS="${WORK_DIR}/subset_theorems.json"
fi

# ── Step 1: LLM-only search ──────────────────────────────────────────────
echo ""
echo "=== Step 1: LLM-only Search ==="
LLM_TRAJ="${WORK_DIR}/llm_only.parquet"

$PROVER search \
    --model-path "$MODEL_PATH" \
    --theorems "$THEOREMS" \
    --output "$LLM_TRAJ"

echo ""
echo "LLM-only trajectory summary:"
$PROVER summary --input "$LLM_TRAJ"

# ── Step 2: Train EBM ────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Train EBM from LLM-only Trajectory ==="
EBM_DIR="${WORK_DIR}/ebm_checkpoint"

$PROVER train-ebm \
    --trajectories "$LLM_TRAJ" \
    --llm-path "$MODEL_PATH" \
    --output-dir "$EBM_DIR" \
    --steps 5000

# ── Step 3: Search with EBM ──────────────────────────────────────────────
echo ""
echo "=== Step 3: Search with EBM Value Guidance ==="
EBM_TRAJ="${WORK_DIR}/with_ebm.parquet"

$PROVER search \
    --model-path "$MODEL_PATH" \
    --ebm-path "$EBM_DIR" \
    --theorems "$THEOREMS" \
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
echo "  Quick start validation complete!"
echo "  All outputs saved to: ${WORK_DIR}"
echo "================================================================"
