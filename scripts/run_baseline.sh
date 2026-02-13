#!/bin/bash
# Run baseline evaluation of the raw (unfine-tuned) DeepSeek-Prover-V2-7B model.
#
# Phase B of the experiment execution plan:
#   B1. Pipeline validation on test_theorems.json (quick)
#   B2. miniF2F benchmark evaluation at budgets 100, 300, 600
#   B3. Full theorem_index search for training data collection
#   B3b. Train baseline EBM on raw model trajectories
#   B4. Record results summary
#
# Usage:
#   ./scripts/run_baseline.sh [model_path]
#   NUM_WORKERS=30 ./scripts/run_baseline.sh
#   CONCURRENCY=16 ./scripts/run_baseline.sh
#
# Prerequisites:
#   - Rust prover-core built (cargo build --release -p prover-core)
#   - Pantograph built (./scripts/setup_pantograph.sh)
#   - Model weights available
#   - Data prepared (test_theorems.json, minif2f_test.json, theorem_index.json)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${1:-${REPO_ROOT}/models/deepseek-prover-v2-7b}"
CONCURRENCY="${CONCURRENCY:-6}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_THEOREMS="${MAX_THEOREMS:-2000}"
EBM_STEPS="${EBM_STEPS:-10000}"

# Auto-detect CUDA
CUDA_FEATURES=$(command -v nvidia-smi &>/dev/null && echo "--features cuda" || echo "")
PROVER="cargo run --release -p prover-core ${CUDA_FEATURES} --"
BASELINES_DIR="${REPO_ROOT}/baselines"
TRAJ_DIR="${REPO_ROOT}/trajectories"

mkdir -p "$BASELINES_DIR" "$TRAJ_DIR" "${REPO_ROOT}/logs"

echo "================================================================"
echo "  Phase B: Baseline Raw Model Evaluation"
echo "================================================================"
echo "  Model:        ${MODEL_PATH}"
echo "  Workers:      ${NUM_WORKERS}"
echo "  Concurrency:  ${CONCURRENCY}"
echo "  Max theorems: ${MAX_THEOREMS}"
echo "  EBM steps:    ${EBM_STEPS}"
echo "  Output dir:   ${BASELINES_DIR}"
echo "================================================================"

# ── B1. Pipeline validation on test_theorems.json ────────────────────────
echo ""
echo "=== B1: Pipeline Validation (test_theorems.json) ==="
TEST_THEOREMS="${REPO_ROOT}/data/test_theorems.json"

if [ ! -f "$TEST_THEOREMS" ]; then
    echo "ERROR: ${TEST_THEOREMS} not found"
    exit 1
fi

$PROVER search \
    --model-path "$MODEL_PATH" \
    --theorems "$TEST_THEOREMS" \
    --output "${BASELINES_DIR}/raw_test_theorems.parquet" \
    --num-workers "$NUM_WORKERS" \
    --concurrency "$CONCURRENCY"

echo ""
echo "Pipeline validation summary:"
$PROVER summary --input "${BASELINES_DIR}/raw_test_theorems.parquet"

# ── B2. miniF2F benchmark evaluation ────────────────────────────────────
echo ""
echo "=== B2: miniF2F Evaluation (budgets 100, 300, 600) ==="
MINIF2F="${REPO_ROOT}/data/minif2f_test.json"

if [ -f "$MINIF2F" ]; then
    $PROVER eval \
        --model-path "$MODEL_PATH" \
        --theorems "$MINIF2F" \
        --budgets 100,300,600 \
        --output "${BASELINES_DIR}/raw_minif2f.json" \
        --num-workers "$NUM_WORKERS" \
        --concurrency "$CONCURRENCY" \
        --max-theorems "$MAX_THEOREMS" \
        --imports Mathlib
else
    echo "Warning: ${MINIF2F} not found, skipping miniF2F evaluation."
    echo "Run: python python/data/trace_mathlib.py --output-dir data/"
fi

# ── B3. Full theorem_index search ───────────────────────────────────────
echo ""
echo "=== B3: Full Theorem Index Search (training data collection) ==="
THEOREM_INDEX="${REPO_ROOT}/data/theorem_index.json"

if [ -f "$THEOREM_INDEX" ]; then
    $PROVER search \
        --model-path "$MODEL_PATH" \
        --theorems "$THEOREM_INDEX" \
        --output "${TRAJ_DIR}/baseline_raw.parquet" \
        --num-workers "$NUM_WORKERS" \
        --concurrency "$CONCURRENCY" \
        --max-theorems "$MAX_THEOREMS" \
        --imports Mathlib

    echo ""
    echo "Full theorem search summary:"
    $PROVER summary --input "${TRAJ_DIR}/baseline_raw.parquet"
else
    echo "Warning: ${THEOREM_INDEX} not found, skipping full search."
    echo "Run: python python/data/trace_mathlib.py --output-dir data/"
fi

# ── B3b. Train baseline EBM ──────────────────────────────────────────────
echo ""
echo "=== B3b: Baseline EBM Training ==="
BASELINE_EBM_DIR="${REPO_ROOT}/checkpoints/ebm/baseline"
mkdir -p "$BASELINE_EBM_DIR"

if [ -f "${TRAJ_DIR}/baseline_raw.parquet" ]; then
    $PROVER train-ebm \
        --trajectories "${TRAJ_DIR}/baseline_raw.parquet" \
        --llm-path "$MODEL_PATH" \
        --output-dir "$BASELINE_EBM_DIR" \
        --steps "$EBM_STEPS" \
        --save-embeddings "${BASELINE_EBM_DIR}/embeddings.parquet"

    echo "Baseline EBM saved to: ${BASELINE_EBM_DIR}"
else
    echo "Warning: baseline_raw.parquet not found, skipping baseline EBM training."
fi

# ── B4. Results summary ─────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Phase B Complete — Baseline Results"
echo "================================================================"
echo ""

if [ -f "${BASELINES_DIR}/raw_test_theorems.parquet" ]; then
    echo "  B1 (pipeline validation):"
    $PROVER summary --input "${BASELINES_DIR}/raw_test_theorems.parquet"
    echo ""
fi

if [ -f "${BASELINES_DIR}/raw_minif2f.json" ]; then
    echo "  B2 (miniF2F baseline):"
    echo "    Results: ${BASELINES_DIR}/raw_minif2f.json"
    python3 -c "
import json
with open('${BASELINES_DIR}/raw_minif2f.json') as f:
    r = json.load(f)
for br in r.get('budget_results', []):
    print(f\"    Budget {br['budget']}: {br['solved']}/{br['total']} ({br['rate']*100:.1f}%)\")
print(f\"    Cumulative: {r['cumulative_solved']}/{r['total_theorems']} ({r['cumulative_rate']*100:.1f}%)\")
" 2>/dev/null || echo "    (could not parse results JSON)"
    echo ""
fi

if [ -f "${TRAJ_DIR}/baseline_raw.parquet" ]; then
    echo "  B3 (training data):"
    echo "    Trajectory: ${TRAJ_DIR}/baseline_raw.parquet"
    echo ""
fi

if [ -d "$BASELINE_EBM_DIR" ] && [ -f "${BASELINE_EBM_DIR}/final.mpk" ]; then
    echo "  B3b (baseline EBM):"
    echo "    Checkpoint: ${BASELINE_EBM_DIR}/"
    echo ""
fi

echo "  All baseline artifacts saved to: ${BASELINES_DIR}/"
echo "================================================================"
