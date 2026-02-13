#!/bin/bash
# End-to-end smoke test for the full pipeline (~3-5 min on A100).
#
# Validates that all components work together:
#   1. LLM-only search on all test theorems (light node budget)
#   2. Train EBM from trajectory (200 steps, skipped if insufficient data)
#   3. Search with EBM (skipped if EBM training was skipped)
#   4. Compare solve rates
#
# Usage:
#   SGLANG_URL=http://localhost:30000 ./scripts/lean_start.sh
#
# Prerequisites:
#   - SGLang server running (./scripts/start_sglang.sh)
#   - Pantograph built (./scripts/setup_pantograph.sh)
#   - cargo build --release -p prover-core

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
WORK_DIR="${REPO_ROOT}/lean_start_output"

PROVER="cargo run --release -p prover-core --"

echo "================================================================"
echo "  BurnQED Smoke Test"
echo "================================================================"
echo "  SGLang: ${SGLANG_URL}"
echo "  Output: ${WORK_DIR}"
echo "================================================================"

mkdir -p "$WORK_DIR"

# ── Step 0: Create inputs ──────────────────────────────────────────────
# Use all test theorems (16 theorems of varying difficulty)
SMOKE_THEOREMS="${REPO_ROOT}/data/test_theorems.json"
echo "Using all theorems from ${SMOKE_THEOREMS}"

# Light search config for smoke test: 100 nodes, 4 candidates, 120s timeout
SMOKE_CONFIG="${WORK_DIR}/smoke_search.toml"
cat > "$SMOKE_CONFIG" << 'TOML'
[search]
max_nodes = 100
max_depth = 25
num_candidates = 4
beam_width = 8
alpha = 0.5
beta = 0.5
timeout_per_theorem = 120

[lean_pool]
num_workers = 6
max_requests_per_worker = 1000
max_lifetime_secs = 1800
tactic_timeout_secs = 30
TOML

# ── Step 1: LLM-only search ──────────────────────────────────────────────
echo ""
echo "=== Step 1: LLM-only Search (16 theorems, 100 nodes) ==="
LLM_TRAJ="${WORK_DIR}/llm_only.parquet"

$PROVER search \
    --config "$SMOKE_CONFIG" \
    --server-url "$SGLANG_URL" \
    --theorems "$SMOKE_THEOREMS" \
    --output "$LLM_TRAJ"

echo ""
echo "LLM-only trajectory summary:"
$PROVER summary --input "$LLM_TRAJ"

# ── Step 2: Train EBM ────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Train EBM (200 steps) ==="
EBM_DIR="${WORK_DIR}/ebm_checkpoint"

$PROVER train-ebm \
    --trajectories "$LLM_TRAJ" \
    --server-url "$SGLANG_URL" \
    --hidden-size "${HIDDEN_SIZE:-4096}" \
    --output-dir "$EBM_DIR" \
    --steps 200

# ── Step 3: Search with EBM (if trained) ────────────────────────────────
EBM_TRAJ="${WORK_DIR}/with_ebm.parquet"

if [ -f "${EBM_DIR}/final.mpk" ]; then
    echo ""
    echo "=== Step 3: Search with EBM (16 theorems, 100 nodes) ==="

    $PROVER search \
        --config "$SMOKE_CONFIG" \
        --server-url "$SGLANG_URL" \
        --ebm-path "$EBM_DIR" \
        --theorems "$SMOKE_THEOREMS" \
        --output "$EBM_TRAJ"

    echo ""
    echo "EBM-guided trajectory summary:"
    $PROVER summary --input "$EBM_TRAJ"
else
    echo ""
    echo "=== Step 3: Skipped (EBM not trained) ==="
fi

# ── Step 4: Compare ──────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Results"
echo "================================================================"

echo ""
echo "LLM-only:"
$PROVER summary --input "$LLM_TRAJ"

if [ -f "$EBM_TRAJ" ]; then
    echo ""
    echo "LLM+EBM:"
    $PROVER summary --input "$EBM_TRAJ"
fi

echo ""
echo "================================================================"
echo "  Smoke test complete!"
echo "  All outputs saved to: ${WORK_DIR}"
echo "================================================================"
