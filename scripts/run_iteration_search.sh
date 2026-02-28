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
LLM_DIR="${MERGED_MODEL_DIR}/iter_${ITER}"
EBM_DIR="${EBM_CKPT_DIR}/iter_${ITER}"
SEARCH_THEOREMS="${BENCH_DIR}/iter${ITER}_search_theorems.json"
THEOREM_INDEX="${THEOREM_INDEX:-${BENCH_DIR}/theorem_index.json}"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
ENCODE_URL="${ENCODE_URL:-http://localhost:30001}"
ensure_server "$SGLANG_URL" "$LLM_DIR"
CONCURRENCY="${CONCURRENCY:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_THEOREMS="${MAX_THEOREMS:-}"
START_STEP="${START_STEP:-2}"
SEARCH_CONFIG="${SEARCH_CONFIG:-${REPO_ROOT}/configs/search_minif2f.toml}"

PROVER="cargo run --release -p prover-core $CARGO_FEATURES --"

mkdir -p "$TRAJ_DIR" "$EVAL_DIR" "$EBM_DIR" "$LOG_DIR"

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
echo "  Max theorems:    ${MAX_THEOREMS:-all}"
echo "  Start step:      ${START_STEP} (2=search, 3=summary)"
echo "================================================================"

# ── Step 2: Proof Search ──────────────────────────────────────────────────

# EBM flag needed by both search and eval — resolve before skip check
EBM_FLAG=""
ENCODE_FLAG=""
if [ "$ITER" -gt 0 ] && [ -d "$EBM_DIR" ] && [ -f "${EBM_DIR}/final/model.mpk" ]; then
    EBM_FLAG="--ebm-path ${EBM_DIR}"
    ENCODE_FLAG="--encode-url ${ENCODE_URL}"
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

    MAX_FLAG=""
    if [ -n "$MAX_THEOREMS" ]; then
        MAX_FLAG="--max-theorems $MAX_THEOREMS"
    fi

    # Auto-resume from partial trajectory if it exists.
    # The prover only merges when --resume-from differs from --output,
    # so we copy the old file to a .resume temp path.
    RESUME_FLAG=""
    RESUME_TMP="${TRAJ_OUTPUT%.parquet}.resume.parquet"
    if [ -f "$TRAJ_OUTPUT" ] && [ -s "$TRAJ_OUTPUT" ]; then
        DONE_COUNT=$(python3 -c "import pyarrow.parquet as pq; t = pq.read_table('$TRAJ_OUTPUT', columns=['theorem_name']); print(len(set(t.column('theorem_name').to_pylist())))" 2>/dev/null || echo "0")
        if [ "$DONE_COUNT" -gt 0 ]; then
            cp "$TRAJ_OUTPUT" "$RESUME_TMP"
            echo "  Resuming from partial trajectory: ${TRAJ_OUTPUT} (${DONE_COUNT} theorems done)"
            RESUME_FLAG="--resume-from $RESUME_TMP"
        fi
    fi

    # shellcheck disable=SC2086
    run_logged "$STEP_LOG" $PROVER search \
        --config $SEARCH_CONFIG \
        --server-url $SGLANG_URL \
        $EBM_FLAG \
        $ENCODE_FLAG \
        --theorems $SEARCH_THEOREMS \
        --output $TRAJ_OUTPUT \
        --concurrency $CONCURRENCY \
        $RESUME_FLAG \
        $MAX_FLAG \
        --imports Mathlib

    # Clean up resume temp file
    rm -f "$RESUME_TMP"

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
            --concurrency $CONCURRENCY \
            $MAX_FLAG \
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
