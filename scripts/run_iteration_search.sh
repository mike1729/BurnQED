#!/bin/bash
# Steps 2-5 of an expert iteration: EBM training, proof search, eval, summary.
#
# Run this after run_iteration_train.sh and restarting SGLang with the new model.
#
# Usage:
#   ./scripts/run_iteration_search.sh <iteration_number>
#   ./scripts/run_iteration_search.sh 0
#   NUM_WORKERS=30 ./scripts/run_iteration_search.sh 1

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
THEOREM_INDEX="${THEOREM_INDEX:-${REPO_ROOT}/data/theorem_index.json}"
MINIF2F="${REPO_ROOT}/data/minif2f_test.json"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
ensure_sglang "$SGLANG_URL"
CONCURRENCY="${CONCURRENCY:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_THEOREMS="${MAX_THEOREMS:-2000}"
EVAL_MAX_THEOREMS="${EVAL_MAX_THEOREMS:-500}"
EBM_STEPS="${EBM_STEPS:-50000}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-64}"
ENCODE_CONCURRENCY="${ENCODE_CONCURRENCY:-2}"
EBM_RESUME="${EBM_RESUME:-auto}"
SEARCH_CONFIG="${REPO_ROOT}/configs/search.toml"

PROVER="cargo run --release -p prover-core $CARGO_FEATURES --"

mkdir -p "$TRAJ_DIR" "$EVAL_DIR" "$EBM_DIR"
mkdir -p "${REPO_ROOT}/logs"

echo "================================================================"
echo "  Expert Iteration ${ITER} — Search & Eval"
echo "================================================================"
echo "  LLM model:      ${LLM_DIR}"
echo "  EBM output:     ${EBM_DIR}"
echo "  Trajectory dir:  ${TRAJ_DIR}"
echo "  Theorem index:   ${THEOREM_INDEX}"
echo "  Config:          ${SEARCH_CONFIG}"
echo "  SGLang:          ${SGLANG_URL}"
echo "  Workers:         ${NUM_WORKERS}"
echo "  Concurrency:     ${CONCURRENCY}"
echo "  Max theorems:    ${MAX_THEOREMS} (eval: ${EVAL_MAX_THEOREMS})"
echo "  EBM steps:       ${EBM_STEPS}"
echo "  EBM resume:      ${EBM_RESUME}"
echo "  Encode:          batch_size=${ENCODE_BATCH_SIZE} concurrency=${ENCODE_CONCURRENCY}"
echo "================================================================"

# ── Step 2: EBM Training (skip iteration 0) ───────────────────────────────
if [ "$ITER" -gt 0 ]; then
    echo ""
    echo "=== Step 2a: Preparing EBM Training Data ==="

    # Collect trajectory files from all previous iterations.
    # Includes: iter_N.parquet, iter_N_noisy.parquet, iter_N_harvest.parquet,
    #           iter_N_negatives.parquet, etc.
    # Excludes: *_test.parquet, *_debug.parquet, *_smoke.parquet (tiny artifacts)
    TRAJ_FILES=()
    for i in $(seq 0 "$PREV"); do
        pattern="${TRAJ_DIR}/iter_${i}*.parquet"
        # shellcheck disable=SC2086
        for f in $pattern; do
            base=$(basename "$f")
            case "$base" in
                *_test.parquet|*_debug.parquet|*_smoke.parquet) continue ;;
            esac
            [ -f "$f" ] && TRAJ_FILES+=("$f")
        done
    done

    if [ ${#TRAJ_FILES[@]} -eq 0 ]; then
        echo "ERROR: No trajectory files found for iterations 0..${PREV}"
        exit 1
    fi

    # Data summary: report files, total size, and record stats
    echo "  Trajectory files (${#TRAJ_FILES[@]}):"
    TOTAL_SIZE=0
    for f in "${TRAJ_FILES[@]}"; do
        fsize=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
        fsize_mb=$((fsize / 1048576))
        TOTAL_SIZE=$((TOTAL_SIZE + fsize))
        echo "    $(basename "$f")  (${fsize_mb}MB)"
    done
    echo "  Total: $((TOTAL_SIZE / 1048576))MB across ${#TRAJ_FILES[@]} files"

    # Quick record count via Python (fast — reads only Parquet metadata + label column)
    python3 -c "
import pyarrow.parquet as pq
from collections import Counter
import sys
files = sys.argv[1:]
total, labels = 0, Counter()
states = set()
for f in files:
    t = pq.read_table(f, columns=['label', 'state_pp'])
    total += len(t)
    labels.update(t.column('label').to_pylist())
    states.update(t.column('state_pp').to_pylist())
print(f'  Records: {total:,} ({labels.get(\"positive\",0):,} pos, {labels.get(\"negative\",0):,} neg)')
print(f'  Unique states to encode: {len(states):,}')
print(f'  EBM network: 11M params (4096→2048→1024→512→1)')
steps = int('${EBM_STEPS}')
bs = 128
draws = steps * bs
print(f'  Training: {steps} steps × {bs} batch = {draws:,} samples')
print(f'  Effective epochs: ~{draws / max(len(states), 1):.1f}×')
" "${TRAJ_FILES[@]}"

    # EBM resume logic
    RESUME_FLAG=""
    PREV_EBM="${REPO_ROOT}/checkpoints/ebm/iter_${PREV}"
    if [ "$EBM_RESUME" = "none" ]; then
        RESUME_FLAG=""
        echo "  Resume: disabled (EBM_RESUME=none)"
    elif [ "$EBM_RESUME" = "auto" ] && [ -d "$PREV_EBM" ] && [ -f "${PREV_EBM}/final.mpk" ]; then
        RESUME_FLAG="--resume-from ${PREV_EBM}"
        echo "  Resume: from ${PREV_EBM}"
    else
        echo "  Resume: none (no previous checkpoint or EBM_RESUME=${EBM_RESUME})"
    fi

    EMBEDDINGS_SAVE="${EBM_DIR}/embeddings.parquet"

    # Auto-resume from partial embeddings cache if it exists
    EMBEDDINGS_CACHE_FLAG=""
    if [ -f "$EMBEDDINGS_SAVE" ]; then
        EMBEDDINGS_CACHE_FLAG="--embeddings-cache ${EMBEDDINGS_SAVE}"
        echo "  Warm start: loading existing embeddings from ${EMBEDDINGS_SAVE}"
    fi

    echo ""
    echo "=== Step 2b: EBM Training ==="

    # shellcheck disable=SC2086
    $PROVER train-ebm \
        --trajectories "${TRAJ_FILES[@]}" \
        --server-url "$SGLANG_URL" \
        --hidden-size "${HIDDEN_SIZE:-4096}" \
        --output-dir "$EBM_DIR" \
        --steps "$EBM_STEPS" \
        --batch-size 128 \
        --save-embeddings "$EMBEDDINGS_SAVE" \
        $EMBEDDINGS_CACHE_FLAG \
        $RESUME_FLAG \
        --encode-batch-size "$ENCODE_BATCH_SIZE" \
        --encode-concurrency "$ENCODE_CONCURRENCY"
else
    echo ""
    echo "=== Step 2: Skipping EBM training (iteration 0) ==="
fi

# ── Step 3: Proof Search ──────────────────────────────────────────────────
echo ""
echo "=== Step 3: Proof Search ==="

EBM_FLAG=""
if [ "$ITER" -gt 0 ] && [ -d "$EBM_DIR" ] && [ -f "${EBM_DIR}/final.mpk" ]; then
    EBM_FLAG="--ebm-path ${EBM_DIR}"
fi

TRAJ_OUTPUT="${TRAJ_DIR}/iter_${ITER}.parquet"

# shellcheck disable=SC2086
$PROVER search \
    --config "$SEARCH_CONFIG" \
    --server-url "$SGLANG_URL" \
    $EBM_FLAG \
    --theorems "$THEOREM_INDEX" \
    --output "$TRAJ_OUTPUT" \
    --num-workers "$NUM_WORKERS" \
    --concurrency "$CONCURRENCY" \
    --max-theorems "$MAX_THEOREMS" \
    --imports Mathlib

# ── Step 3b: Noise injection search (iteration 0 only) ────────────────────
if [ "$ITER" -eq 0 ]; then
    echo ""
    echo "=== Step 3b: Noise Injection Search (temperature=1.2) ==="

    NOISY_OUTPUT="${TRAJ_DIR}/iter_0_noisy.parquet"

    $PROVER search \
        --config "$SEARCH_CONFIG" \
        --server-url "$SGLANG_URL" \
        --temperature 1.2 \
        --theorems "$THEOREM_INDEX" \
        --output "$NOISY_OUTPUT" \
        --num-workers "$NUM_WORKERS" \
        --concurrency "$CONCURRENCY" \
        --max-theorems "$MAX_THEOREMS" \
        --imports Mathlib
fi

# ── Step 4: Evaluation ────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Evaluation ==="

if [ -f "$MINIF2F" ]; then
    EVAL_THEOREMS="$MINIF2F"
else
    echo "Warning: miniF2F file not found at ${MINIF2F}, using theorem_index.json"
    EVAL_THEOREMS="$THEOREM_INDEX"
fi

# Eval WITH EBM (if available)
# shellcheck disable=SC2086
$PROVER eval \
    --config "$SEARCH_CONFIG" \
    --server-url "$SGLANG_URL" \
    $EBM_FLAG \
    --theorems "$EVAL_THEOREMS" \
    --budgets 600 \
    --output "${EVAL_DIR}/iter_${ITER}.json" \
    --num-workers "$NUM_WORKERS" \
    --concurrency "$CONCURRENCY" \
    --max-theorems "$EVAL_MAX_THEOREMS" \
    --num-candidates 16 \
    --imports Mathlib

# ── Step 4b: EBM Ablation (iter > 0 — eval WITHOUT EBM) ──────────────────
if [ "$ITER" -gt 0 ] && [ -n "$EBM_FLAG" ]; then
    echo ""
    echo "=== Step 4b: EBM Ablation (eval WITHOUT EBM) ==="

    $PROVER eval \
        --config "$SEARCH_CONFIG" \
        --server-url "$SGLANG_URL" \
        --theorems "$EVAL_THEOREMS" \
        --budgets 600 \
        --output "${EVAL_DIR}/iter_${ITER}_no_ebm.json" \
        --num-workers "$NUM_WORKERS" \
        --concurrency "$CONCURRENCY" \
        --max-theorems "$EVAL_MAX_THEOREMS" \
        --num-candidates 16 \
        --imports Mathlib
fi

# ── Step 5: Summary ──────────────────────────────────────────────────────
echo ""
echo "=== Step 5: Trajectory Summary ==="
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
