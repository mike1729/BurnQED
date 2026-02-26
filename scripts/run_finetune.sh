#!/bin/bash
# Standalone LLM fine-tuning script. Runs ONLY the training step —
# no eval, no export, no server restart.
#
# Usage:
#   ./scripts/run_finetune.sh <iteration_number>
#
# Examples:
#   # Single-shot r=32 LoRA with separation probe (30K steps)
#   LORA_MODE=single-shot LORA_R=32 LORA_ALPHA=64 LORA_MLP=1 \
#   LR=3e-5 MAX_TRAIN_STEPS=30000 SAVE_STEPS=2000 \
#   PROBE_DATA=data/separation_probe.json \
#     ./scripts/run_finetune.sh 3
#
#   # Quick iteration 1 training with defaults
#   ./scripts/run_finetune.sh 1
#
#   # Iter 0: base competition tactic pairs only
#   ./scripts/run_finetune.sh 0
#
# Environment variables:
#   LLM_BASE          Base model (default: deepseek-ai/DeepSeek-Prover-V2-7B)
#   LORA_MODE         continue|fresh|single-shot (default: continue)
#   LORA_R            LoRA rank (default: 16)
#   LORA_ALPHA         LoRA alpha (default: 32)
#   LORA_MLP          1 to include MLP layers (default: 0)
#   LR                Learning rate (default: auto-decay by iteration)
#   MAX_TRAIN_STEPS   Max training steps (default: 1500 iter0, 2000 iterN)
#   SAVE_STEPS        Checkpoint interval (default: auto)
#   PROBE_DATA        Path to separation probe JSON (default: none)
#   BASE_SUBSAMPLE    Subsample base data (default: 50000, iter>0 only)
#   BATCH_SIZE        Per-device batch size (default: 8)
#   GRAD_ACCUM        Gradient accumulation steps (default: 2)
#   KILL_SERVER       1 to kill inference server before training (default: 1)

set -euo pipefail
export PYTHONUNBUFFERED=1

ITER=${1:?"Usage: ./scripts/run_finetune.sh <iteration_number>"}
PREV=$((ITER - 1))

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib.sh"
LLM_BASE="${LLM_BASE:-deepseek-ai/DeepSeek-Prover-V2-7B}"
TRAJ_DIR="${REPO_ROOT}/trajectories"
CKPT_DIR="${REPO_ROOT}/checkpoints/llm"
TRAIN_DATA="${REPO_ROOT}/data/sft_train.jsonl"
VAL_DATA="${REPO_ROOT}/data/sft_val.jsonl"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ── Configuration ──────────────────────────────────────────────────────────
LR="${LR:-$(python3 -c "print(2e-4 / (2 ** ${ITER}))")}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_MODE="${LORA_MODE:-continue}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

echo "================================================================"
echo "  LLM Fine-tuning — Iteration ${ITER}"
echo "================================================================"
echo "  LLM base:       ${LLM_BASE}"
echo "  Checkpoint dir:  ${CKPT_DIR}/iter_${ITER}"
echo "  Train data:      ${TRAIN_DATA}"
echo "  LoRA mode:       ${LORA_MODE}"
echo "  LoRA rank:       ${LORA_R}, alpha: ${LORA_ALPHA}, MLP: ${LORA_MLP:-0}"
echo "  LR:              ${LR}"
echo "  Max seq len:     ${MAX_SEQ_LEN:-1024}"
echo "  Save steps:      ${SAVE_STEPS:-auto}"
echo "  Probe data:      ${PROBE_DATA:-none}"
echo "  Batch size:      ${BATCH_SIZE} × ${GRAD_ACCUM} accum"
echo "================================================================"

# ── Kill inference server to free VRAM ─────────────────────────────────────
if [ "${KILL_SERVER:-1}" = "1" ]; then
    stop_inference_server
fi

# ── Build flags ────────────────────────────────────────────────────────────
LORA_FLAGS="--lora-r $LORA_R --lora-alpha $LORA_ALPHA"
if [ "${LORA_MLP:-0}" = "1" ]; then
    LORA_FLAGS="$LORA_FLAGS --lora-mlp"
fi

VAL_DATA_FLAG=""
[ -f "$VAL_DATA" ] && VAL_DATA_FLAG="--val-data ${VAL_DATA}"

SAVE_STEPS_FLAG=""
[ -n "${SAVE_STEPS:-}" ] && SAVE_STEPS_FLAG="--save-steps $SAVE_STEPS"

PROBE_DATA_FLAG=""
[ -n "${PROBE_DATA:-}" ] && PROBE_DATA_FLAG="--probe-data $PROBE_DATA --probe-interval ${PROBE_INTERVAL:-500}"

MAX_SEQ_LEN_FLAG=""
[ -n "${MAX_SEQ_LEN:-}" ] && MAX_SEQ_LEN_FLAG="--max-seq-len $MAX_SEQ_LEN"

RESUME_FLAG=""
[ -n "${RESUME:-}" ] && RESUME_FLAG="--resume $RESUME"

STEP_LOG="${LOG_DIR}/iter_${ITER}_finetune.log"
echo "  Logging to: ${STEP_LOG}"

# ── Iteration 0: base data only ───────────────────────────────────────────
if [ "$ITER" -eq 0 ]; then
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1500}"
    echo "Iteration 0: training on base competition tactic pairs only (${MAX_TRAIN_STEPS} steps)"

    # shellcheck disable=SC2086
    run_logged "$STEP_LOG" accelerate launch \
        ${REPO_ROOT}/python/training/train_llm.py \
        --model-name $LLM_BASE \
        --data $TRAIN_DATA \
        $VAL_DATA_FLAG \
        $LORA_FLAGS \
        --output ${CKPT_DIR}/iter_0 \
        --max-steps $MAX_TRAIN_STEPS \
        --lr $LR \
        $SAVE_STEPS_FLAG $PROBE_DATA_FLAG $MAX_SEQ_LEN_FLAG $RESUME_FLAG
else
    # ── Iteration N>0: base + trajectory data ──────────────────────────────
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
    BASE_SUBSAMPLE="${BASE_SUBSAMPLE:-50000}"

    # Collect trajectory parquets
    EXTRA_DATA_ARGS=()
    if [ "$LORA_MODE" = "single-shot" ]; then
        TRAJ_MAX_ITER="$ITER"
    else
        TRAJ_MAX_ITER="$PREV"
    fi
    for i in $(seq 0 "$TRAJ_MAX_ITER"); do
        for f in "${TRAJ_DIR}"/iter_${i}*.parquet; do
            [ -f "$f" ] && EXTRA_DATA_ARGS+=(--extra-data "$f")
        done
    done
    for f in "${TRAJ_DIR}"/negatives_2*.parquet; do
        [ -f "$f" ] && EXTRA_DATA_ARGS+=(--extra-data "$f")
    done

    echo "Trajectory files: ${#EXTRA_DATA_ARGS[@]} flags ($(( ${#EXTRA_DATA_ARGS[@]} / 2 )) files)"
    echo "LoRA config: r=$LORA_R, alpha=$LORA_ALPHA, MLP=${LORA_MLP:-0}"
    echo "Training: ${MAX_TRAIN_STEPS} steps, base subsample=${BASE_SUBSAMPLE}"

    # Model and adapter selection
    BASE_FLAGS=""
    MODEL_NAME="$LLM_BASE"
    if [ "$LORA_MODE" = "single-shot" ]; then
        echo "Single-shot LoRA: fresh adapter on raw base model ${LLM_BASE}"
    elif [ "$LORA_MODE" = "fresh" ]; then
        PREV_MERGED="${REPO_ROOT}/models/llm/iter_${PREV}"
        if [ ! -d "$PREV_MERGED" ]; then
            echo "ERROR: LORA_MODE=fresh but merged model not found at ${PREV_MERGED}"
            exit 1
        fi
        MODEL_NAME="$PREV_MERGED"
        echo "Fresh LoRA: training adapters on merged model ${PREV_MERGED}"
    else
        BASE_FLAGS="--base ${CKPT_DIR}/iter_${PREV}"
        echo "Continue LoRA: resuming from adapter ${CKPT_DIR}/iter_${PREV}"
    fi

    # shellcheck disable=SC2086
    run_logged "$STEP_LOG" accelerate launch \
        ${REPO_ROOT}/python/training/train_llm.py \
        --model-name $MODEL_NAME \
        --data $TRAIN_DATA \
        $VAL_DATA_FLAG \
        ${EXTRA_DATA_ARGS[*]} \
        --output ${CKPT_DIR}/iter_${ITER} \
        $BASE_FLAGS \
        $LORA_FLAGS \
        --max-steps $MAX_TRAIN_STEPS \
        --base-subsample $BASE_SUBSAMPLE \
        --batch-size $BATCH_SIZE \
        --gradient-accumulation $GRAD_ACCUM \
        --lr $LR \
        $SAVE_STEPS_FLAG $PROBE_DATA_FLAG $MAX_SEQ_LEN_FLAG $RESUME_FLAG
fi

echo ""
echo "================================================================"
echo "  Fine-tuning complete!"
echo "  LoRA checkpoint: ${CKPT_DIR}/iter_${ITER}"
echo "  Log:             ${STEP_LOG}"
echo "================================================================"
