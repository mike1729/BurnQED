#!/bin/bash
# Expert iteration: pre-eval → LLM fine-tuning → export → restart server → post-eval.
#
# Usage:
#   ./scripts/run_iteration_train.sh <iteration_number>
#   ./scripts/run_iteration_train.sh 0   # First iteration (base model fine-tune)
#   ./scripts/run_iteration_train.sh 1   # Second iteration (adds trajectory data)
#   START_STEP=1 ./scripts/run_iteration_train.sh 1  # Skip pre-eval, start at training
#   START_STEP=2 ./scripts/run_iteration_train.sh 1  # Skip training, restart + post-eval
#
# Steps: 0=pre-eval, 1=LLM fine-tuning, 1b=export, 2=restart server, 3=post-eval
#
# Prerequisites:
#   - Python deps installed (pip install -r python/requirements.txt)
#   - Model weights available (HuggingFace or local)
#   - Training data prepared (python python/data/trace_mathlib.py + prepare_tactic_pairs.py)

set -euo pipefail
export PYTHONUNBUFFERED=1

ITER=${1:?"Usage: ./scripts/run_iteration_train.sh <iteration_number>"}
PREV=$((ITER - 1))

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib.sh"
LLM_BASE="${LLM_BASE:-deepseek-ai/DeepSeek-Prover-V2-7B}"
LLM_DIR="${REPO_ROOT}/models/llm/iter_${ITER}"
TRAJ_DIR="${REPO_ROOT}/trajectories"
CKPT_DIR="${REPO_ROOT}/checkpoints/llm"
TRAIN_DATA="${REPO_ROOT}/data/tactic_pairs/train_formatted.jsonl"
VAL_DATA="${REPO_ROOT}/data/tactic_pairs/val_formatted.jsonl"

# Pre/post eval model: iter 0 uses base HF model, iter N uses previous iter's export
if [ "$ITER" -eq 0 ]; then
    PRE_MODEL="$LLM_BASE"
else
    PRE_MODEL="${REPO_ROOT}/models/llm/iter_${PREV}"
fi

# Eval configuration
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
EVAL_DIR="${REPO_ROOT}/eval_results"
EVAL_BUDGET="${EVAL_BUDGET:-300}"
EVAL_MAX_THEOREMS_TRAIN="${EVAL_MAX_THEOREMS_TRAIN:-200}"
CONCURRENCY="${CONCURRENCY:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TRAIN_EVAL_THEOREMS="${REPO_ROOT}/data/train_eval_theorems.json"
SEARCH_CONFIG="${REPO_ROOT}/configs/search.toml"
PROVER="cargo run --release -p prover-core $CARGO_FEATURES --"

START_STEP="${START_STEP:-0}"

mkdir -p "$CKPT_DIR" "$LLM_DIR" "$EVAL_DIR"
mkdir -p "${REPO_ROOT}/logs"

echo "================================================================"
echo "  Expert Iteration ${ITER} — Training"
echo "================================================================"
echo "  LLM base:       ${LLM_BASE}"
echo "  Pre-train model: ${PRE_MODEL}"
echo "  LLM output:     ${LLM_DIR}"
echo "  Checkpoint dir:  ${CKPT_DIR}"
echo "  Train data:      ${TRAIN_DATA}"
echo "  Eval theorems:   ${TRAIN_EVAL_THEOREMS}"
echo "  Eval budget:     ${EVAL_BUDGET}"
echo "  Start step:      ${START_STEP} (0=pre-eval, 1=train, 2=restart, 3=post-eval)"
echo "================================================================"

# ── Step 0: Pre-training Eval ────────────────────────────────────────────
if [ "$START_STEP" -gt 0 ]; then
    echo ""
    echo "=== Step 0: Pre-training Eval [SKIPPED — START_STEP=${START_STEP}] ==="
else
    echo ""
    echo "=== Step 0: Pre-training Eval ==="

    ensure_server "$SGLANG_URL" "$PRE_MODEL"

    PRE_EVAL="${EVAL_DIR}/iter_${ITER}_pre_train.json"

    # shellcheck disable=SC2086
    $PROVER eval \
        --config "$SEARCH_CONFIG" \
        --server-url "$SGLANG_URL" \
        --theorems "$TRAIN_EVAL_THEOREMS" \
        --budgets "$EVAL_BUDGET" \
        --output "$PRE_EVAL" \
        --num-workers "$NUM_WORKERS" \
        --concurrency "$CONCURRENCY" \
        --max-theorems "$EVAL_MAX_THEOREMS_TRAIN" \
        --num-candidates 8 \
        --imports Mathlib

    echo "Pre-training eval saved to: $PRE_EVAL"
fi

# ── Step 1: LLM Fine-tuning ───────────────────────────────────────────────
if [ "$START_STEP" -gt 1 ]; then
    echo ""
    echo "=== Step 1: LLM Fine-tuning [SKIPPED — START_STEP=${START_STEP}] ==="
else
    echo ""
    echo "=== Step 1: LLM Fine-tuning ==="

    # Compute learning rate: halve each iteration (2e-4, 1e-4, 5e-5, ...)
    LR=$(python3 -c "print(2e-4 / (2 ** ${ITER}))")

    VAL_DATA_FLAG=""
    if [ -f "$VAL_DATA" ]; then
        VAL_DATA_FLAG="--val-data ${VAL_DATA}"
    fi

    if [ "$ITER" -eq 0 ]; then
        echo "Iteration 0: training on base Mathlib tactic pairs only"
        # shellcheck disable=SC2086
        accelerate launch "${REPO_ROOT}/python/training/train_llm.py" \
            --model-name "$LLM_BASE" \
            --data "$TRAIN_DATA" \
            $VAL_DATA_FLAG \
            --output "${CKPT_DIR}/iter_0" \
            --max-steps 1500 \
            --lr "$LR"
    else
        echo "Iteration ${ITER}: training with trajectory data from previous iterations"

        # Collect all trajectory sources: iter_N parquets + negatives pipeline output.
        # Each --extra-data flag takes one glob pattern (action="append" in argparse).
        EXTRA_DATA_ARGS=()
        for i in $(seq 0 "$PREV"); do
            pattern="${TRAJ_DIR}/iter_${i}*.parquet"
            # shellcheck disable=SC2086
            if compgen -G $pattern > /dev/null; then
                EXTRA_DATA_ARGS+=(--extra-data "$pattern")
            fi
        done

        # Include generate-negatives pipeline output (high-quality ground-truth data)
        NEG_PATTERN="${TRAJ_DIR}/negatives_2*.parquet"
        # shellcheck disable=SC2086
        if compgen -G $NEG_PATTERN > /dev/null; then
            EXTRA_DATA_ARGS+=(--extra-data "$NEG_PATTERN")
        fi

        BASE_CKPT="${CKPT_DIR}/iter_${PREV}"
        BASE_SUBSAMPLE="${BASE_SUBSAMPLE:-50000}"
        MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
        BATCH_SIZE="${BATCH_SIZE:-8}"
        GRAD_ACCUM="${GRAD_ACCUM:-2}"

        # shellcheck disable=SC2086
        accelerate launch "${REPO_ROOT}/python/training/train_llm.py" \
            --model-name "$LLM_BASE" \
            --data "$TRAIN_DATA" \
            $VAL_DATA_FLAG \
            "${EXTRA_DATA_ARGS[@]}" \
            --output "${CKPT_DIR}/iter_${ITER}" \
            --base "$BASE_CKPT" \
            --max-steps "$MAX_TRAIN_STEPS" \
            --base-subsample "$BASE_SUBSAMPLE" \
            --batch-size "$BATCH_SIZE" \
            --gradient-accumulation "$GRAD_ACCUM" \
            --lr "$LR"
    fi
fi

# ── Step 1b: Export to safetensors ─────────────────────────────────────────
if [ "$START_STEP" -le 1 ]; then
    echo ""
    echo "=== Step 1b: Export LLM to safetensors ==="
    python3 "${REPO_ROOT}/python/training/export_llm.py" \
        --checkpoint "${CKPT_DIR}/iter_${ITER}" \
        --base-model "$LLM_BASE" \
        --output "$LLM_DIR" \
        --verify
fi

# ── Step 2: Restart Server with New Model ──────────────────────────────────
if [ "$START_STEP" -gt 2 ]; then
    echo ""
    echo "=== Step 2: Restart Server [SKIPPED — START_STEP=${START_STEP}] ==="
else
    echo ""
    echo "=== Step 2: Restart Server with New Model ==="
    restart_inference_server "$SGLANG_URL" "$LLM_DIR"
fi

# ── Step 3: Post-training Eval ─────────────────────────────────────────────
if [ "$START_STEP" -gt 3 ]; then
    echo ""
    echo "=== Step 3: Post-training Eval [SKIPPED — START_STEP=${START_STEP}] ==="
else
    echo ""
    echo "=== Step 3: Post-training Eval ==="

    ensure_server "$SGLANG_URL" "$LLM_DIR"

    POST_EVAL="${EVAL_DIR}/iter_${ITER}_post_train.json"

    # shellcheck disable=SC2086
    $PROVER eval \
        --config "$SEARCH_CONFIG" \
        --server-url "$SGLANG_URL" \
        --theorems "$TRAIN_EVAL_THEOREMS" \
        --budgets "$EVAL_BUDGET" \
        --output "$POST_EVAL" \
        --num-workers "$NUM_WORKERS" \
        --concurrency "$CONCURRENCY" \
        --max-theorems "$EVAL_MAX_THEOREMS_TRAIN" \
        --num-candidates 8 \
        --imports Mathlib

    echo "Post-training eval saved to: $POST_EVAL"

    # Compare pre vs post if both exist
    PRE_EVAL="${EVAL_DIR}/iter_${ITER}_pre_train.json"
    if [ -f "$PRE_EVAL" ] && [ -f "$POST_EVAL" ]; then
        echo ""
        echo "=== Training Impact: Pre vs Post ==="
        $PROVER compare --results "$PRE_EVAL" "$POST_EVAL"
    fi
fi

echo ""
echo "================================================================"
echo "  Training complete!"
echo "  LoRA checkpoint: ${CKPT_DIR}/iter_${ITER}"
echo "  Merged model:    ${LLM_DIR}"
if [ -f "${EVAL_DIR}/iter_${ITER}_pre_train.json" ]; then
    echo "  Pre-train eval:  ${EVAL_DIR}/iter_${ITER}_pre_train.json"
fi
if [ -f "${EVAL_DIR}/iter_${ITER}_post_train.json" ]; then
    echo "  Post-train eval: ${EVAL_DIR}/iter_${ITER}_post_train.json"
fi
echo "================================================================"
echo ""
echo "  Next: ./scripts/run_iteration_search.sh ${ITER}"
echo ""
