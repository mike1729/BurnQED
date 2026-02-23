#!/bin/bash
# Proof search, eval, and summary for an expert iteration.
#
# Run this after run_iteration_train.sh (which handles LLM training, encoding,
# and EBM training). The server is auto-managed — if it's running with the
# wrong model (or not running), it will be restarted.
#
# Usage:
#   ./scripts/run_iteration_search.sh <iteration_number>
#   ./scripts/run_iteration_search.sh 0
#   NUM_WORKERS=30 ./scripts/run_iteration_search.sh 1
#   START_STEP=3 ./scripts/run_iteration_search.sh 2   # skip search, start at summary
#
# Steps: 2=proof search, 3=summary
# Note: miniF2F evaluation moved to run_iteration_train.sh (Step 5)

set -euo pipefail
export PYTHONUNBUFFERED=1

ITER=${1:?"Usage: ./scripts/run_iteration_search.sh <iteration_number>"}
PREV=$((ITER - 1))

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$(dirname "$0")/_lib.sh"
LLM_DIR="${REPO_ROOT}/models/llm/iter_${ITER}"
EBM_DIR="${REPO_ROOT}/checkpoints/ebm/iter_${ITER}"
TRAJ_DIR="${REPO_ROOT}/trajectories"
EVAL_DIR="${REPO_ROOT}/eval_results"
SEARCH_THEOREMS="${REPO_ROOT}/data/iter${ITER}_search_theorems.json"
THEOREM_INDEX="${THEOREM_INDEX:-${REPO_ROOT}/data/theorem_index.json}"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
ensure_server "$SGLANG_URL" "$LLM_DIR"
CONCURRENCY="${CONCURRENCY:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_THEOREMS="${MAX_THEOREMS:-2000}"
START_STEP="${START_STEP:-2}"
SEARCH_CONFIG="${REPO_ROOT}/configs/search.toml"

PROVER="cargo run --release -p prover-core $CARGO_FEATURES --"

mkdir -p "$TRAJ_DIR" "$EVAL_DIR" "$EBM_DIR"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  Expert Iteration ${ITER} — Search & Eval"
echo "================================================================"
echo "  LLM model:      ${LLM_DIR}"
echo "  EBM output:     ${EBM_DIR}"
echo "  Trajectory dir:  ${TRAJ_DIR}"
echo "  Search theorems: ${SEARCH_THEOREMS}"
echo "  Eval theorems:   ${THEOREM_INDEX}"
echo "  Config:          ${SEARCH_CONFIG}"
echo "  SGLang:          ${SGLANG_URL}"
echo "  Workers:         ${NUM_WORKERS}"
echo "  Concurrency:     ${CONCURRENCY}"
echo "  Max theorems:    ${MAX_THEOREMS}"
echo "  Start step:      ${START_STEP} (2=search, 3=summary)"
echo "================================================================"

# ── Step 2: Proof Search ──────────────────────────────────────────────────

# EBM flag needed by both search and eval — resolve before skip check
EBM_FLAG=""
if [ "$ITER" -gt 0 ] && [ -d "$EBM_DIR" ] && [ -f "${EBM_DIR}/final.mpk" ]; then
    EBM_FLAG="--ebm-path ${EBM_DIR}"
fi

TRAJ_OUTPUT="${TRAJ_DIR}/iter_${ITER}.parquet"

if [ "$START_STEP" -gt 2 ]; then
    echo ""
    echo "=== Step 2: Proof Search [SKIPPED — START_STEP=${START_STEP}] ==="
else
    echo ""
    echo "=== Step 2: Proof Search ==="

    if [ ! -f "$SEARCH_THEOREMS" ]; then
        echo "ERROR: Search theorem file not found: ${SEARCH_THEOREMS}"
        echo "  Generate it before running search (e.g., filter theorem_index.json for this iteration)."
        exit 1
    fi

    STEP_LOG="${LOG_DIR}/iter_${ITER}_step_2_search.log"
    echo "  Logging to: ${STEP_LOG}"

    # shellcheck disable=SC2086
    run_logged "$STEP_LOG" $PROVER search \
        --config $SEARCH_CONFIG \
        --server-url $SGLANG_URL \
        $EBM_FLAG \
        --theorems $SEARCH_THEOREMS \
        --output $TRAJ_OUTPUT \
        --num-workers $NUM_WORKERS \
        --concurrency $CONCURRENCY \
        --max-theorems $MAX_THEOREMS \
        --imports Mathlib

    # ── Step 2b: Noise injection search (iteration 0 only) ────────────────
    if [ "$ITER" -eq 0 ]; then
        echo ""
        echo "=== Step 2b: Noise Injection Search (temperature=1.2) ==="

        NOISY_OUTPUT="${TRAJ_DIR}/iter_0_noisy.parquet"
        STEP_LOG="${LOG_DIR}/iter_0_step_2b_noisy.log"
        echo "  Logging to: ${STEP_LOG}"

        # shellcheck disable=SC2086
        run_logged "$STEP_LOG" $PROVER search \
            --config $SEARCH_CONFIG \
            --server-url $SGLANG_URL \
            --temperature 1.2 \
            --theorems $SEARCH_THEOREMS \
            --output $NOISY_OUTPUT \
            --num-workers $NUM_WORKERS \
            --concurrency $CONCURRENCY \
            --max-theorems $MAX_THEOREMS \
            --imports Mathlib
    fi
fi

# ── Step 3: Summary ──────────────────────────────────────────────────────
echo ""
echo "=== Step 3: Trajectory Summary ==="
$PROVER summary --input "$TRAJ_OUTPUT"

echo ""
echo "================================================================"
echo "  Iteration ${ITER} search complete!"
echo "  Trajectory:  ${TRAJ_OUTPUT}"
echo "  LLM:         ${LLM_DIR}"
if [ "$ITER" -gt 0 ]; then
    echo "  EBM:         ${EBM_DIR}"
fi
echo "================================================================"
