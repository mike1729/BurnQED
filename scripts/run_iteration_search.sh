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
#   START_STEP=3 ./scripts/run_iteration_search.sh 2   # skip search, start at eval
#   START_STEP=4 ./scripts/run_iteration_search.sh 2   # skip search+eval, start at summary
#
# Steps: 2=proof search, 3=evaluation, 4=summary

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
MINIF2F="${REPO_ROOT}/data/minif2f_test.json"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
ensure_server "$SGLANG_URL" "$LLM_DIR"
CONCURRENCY="${CONCURRENCY:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_THEOREMS="${MAX_THEOREMS:-2000}"
EVAL_MAX_THEOREMS="${EVAL_MAX_THEOREMS:-500}"
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
echo "  Max theorems:    ${MAX_THEOREMS} (eval: ${EVAL_MAX_THEOREMS})"
echo "  Start step:      ${START_STEP} (2=search, 3=eval, 4=summary)"
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

# ── Step 3: Evaluation ────────────────────────────────────────────────────
if [ "$START_STEP" -gt 3 ]; then
    echo ""
    echo "=== Step 3: Evaluation [SKIPPED — START_STEP=${START_STEP}] ==="
else
    echo ""
    echo "=== Step 3: Evaluation ==="

    if [ -f "$MINIF2F" ]; then
        EVAL_THEOREMS="$MINIF2F"
    else
        echo "Warning: miniF2F file not found at ${MINIF2F}, using theorem_index.json"
        EVAL_THEOREMS="$THEOREM_INDEX"
    fi

    # Eval WITH EBM (if available)
    STEP_LOG="${LOG_DIR}/iter_${ITER}_step_3_eval.log"
    echo "  Logging to: ${STEP_LOG}"

    # shellcheck disable=SC2086
    run_logged "$STEP_LOG" $PROVER eval \
        --config $SEARCH_CONFIG \
        --server-url $SGLANG_URL \
        $EBM_FLAG \
        --theorems $EVAL_THEOREMS \
        --budgets 600 \
        --output ${EVAL_DIR}/iter_${ITER}.json \
        --num-workers $NUM_WORKERS \
        --concurrency $CONCURRENCY \
        --max-theorems $EVAL_MAX_THEOREMS \
        --num-candidates 16 \
        --imports Mathlib

    # ── Step 3b: EBM Ablation (iter > 0 — eval WITHOUT EBM) ──────────────
    if [ "$ITER" -gt 0 ] && [ -n "$EBM_FLAG" ]; then
        echo ""
        echo "=== Step 3b: EBM Ablation (eval WITHOUT EBM) ==="
        STEP_LOG="${LOG_DIR}/iter_${ITER}_step_3b_ablation.log"
        echo "  Logging to: ${STEP_LOG}"

        # shellcheck disable=SC2086
        run_logged "$STEP_LOG" $PROVER eval \
            --config $SEARCH_CONFIG \
            --server-url $SGLANG_URL \
            --theorems $EVAL_THEOREMS \
            --budgets 600 \
            --output ${EVAL_DIR}/iter_${ITER}_no_ebm.json \
            --num-workers $NUM_WORKERS \
            --concurrency $CONCURRENCY \
            --max-theorems $EVAL_MAX_THEOREMS \
            --num-candidates 16 \
            --imports Mathlib
    fi
fi

# ── Step 4: Summary ──────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Trajectory Summary ==="
$PROVER summary --input "$TRAJ_OUTPUT"

# Compare with previous iteration if available
if [ "$ITER" -gt 0 ]; then
    PREV_EVAL="${EVAL_DIR}/iter_${PREV}.json"
    CURR_EVAL="${EVAL_DIR}/iter_${ITER}.json"
    if [ -f "$PREV_EVAL" ] && [ -f "$CURR_EVAL" ]; then
        echo ""
        echo "=== Cross-Iteration Comparison ==="
        $PROVER compare --results "$PREV_EVAL" "$CURR_EVAL"
    fi

    # EBM ablation comparison
    NO_EBM_EVAL="${EVAL_DIR}/iter_${ITER}_no_ebm.json"
    if [ -f "$NO_EBM_EVAL" ] && [ -f "$CURR_EVAL" ]; then
        echo ""
        echo "=== EBM Ablation Comparison ==="
        $PROVER compare --results "$NO_EBM_EVAL" "$CURR_EVAL"
    fi
fi

echo ""
echo "================================================================"
echo "  Iteration ${ITER} search & eval complete!"
echo "  Trajectory:  ${TRAJ_OUTPUT}"
echo "  Eval:        ${EVAL_DIR}/iter_${ITER}.json"
if [ "$ITER" -gt 0 ] && [ -f "${EVAL_DIR}/iter_${ITER}_no_ebm.json" ]; then
    echo "  Ablation:    ${EVAL_DIR}/iter_${ITER}_no_ebm.json"
fi
echo "  LLM:         ${LLM_DIR}"
if [ "$ITER" -gt 0 ]; then
    echo "  EBM:         ${EBM_DIR}"
fi
echo "================================================================"
