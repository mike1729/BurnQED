#!/bin/bash
# Step 1 of an expert iteration: LLM fine-tuning + export.
#
# Run this first, then restart SGLang with the new model, then run
# run_iteration_search.sh to do search + eval.
#
# Usage:
#   ./scripts/run_iteration_train.sh <iteration_number>
#   ./scripts/run_iteration_train.sh 0   # First iteration (base model fine-tune)
#   ./scripts/run_iteration_train.sh 1   # Second iteration (adds trajectory data)
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
LLM_BASE="${LLM_BASE:-deepseek-ai/DeepSeek-Prover-V2-7B}"
LLM_DIR="${REPO_ROOT}/models/llm/iter_${ITER}"
TRAJ_DIR="${REPO_ROOT}/trajectories"
CKPT_DIR="${REPO_ROOT}/checkpoints/llm"
TRAIN_DATA="${REPO_ROOT}/data/tactic_pairs/train_formatted.jsonl"
VAL_DATA="${REPO_ROOT}/data/tactic_pairs/val_formatted.jsonl"

mkdir -p "$CKPT_DIR" "$LLM_DIR"
mkdir -p "${REPO_ROOT}/logs"

echo "================================================================"
echo "  Expert Iteration ${ITER} — Training"
echo "================================================================"
echo "  LLM base:       ${LLM_BASE}"
echo "  LLM output:     ${LLM_DIR}"
echo "  Checkpoint dir:  ${CKPT_DIR}"
echo "  Train data:      ${TRAIN_DATA}"
echo "================================================================"

# ── Step 1: LLM Fine-tuning ───────────────────────────────────────────────
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

# ── Step 1b: Export to safetensors ─────────────────────────────────────────
echo ""
echo "=== Step 1b: Export LLM to safetensors ==="
python3 "${REPO_ROOT}/python/training/export_llm.py" \
    --checkpoint "${CKPT_DIR}/iter_${ITER}" \
    --base-model "$LLM_BASE" \
    --output "$LLM_DIR" \
    --verify

echo ""
echo "================================================================"
echo "  Training complete!"
echo "  LoRA checkpoint: ${CKPT_DIR}/iter_${ITER}"
echo "  Merged model:    ${LLM_DIR}"
echo "================================================================"
echo ""
echo "  Next steps:"
echo "    1. Restart SGLang with the new model:"
echo "       python -m sglang.launch_server --model ${LLM_DIR} --tp 1 --port 30000 --enable-return-hidden-states"
echo "    2. Run search + eval:"
echo "       ./scripts/run_iteration_search.sh ${ITER}"
echo ""
