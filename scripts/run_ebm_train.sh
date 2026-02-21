#!/bin/bash
# Encode embeddings and train EBM for an expert iteration.
#
# Standalone script: can be run independently or called from run_iteration_train.sh.
# Stops the inference server to free VRAM for direct PyTorch encoding (SGLang batch
# hidden states are broken — see docs/encoding_bug.md), then restarts it for any
# subsequent steps.
#
# Usage:
#   ./scripts/run_ebm_train.sh <iteration_number>
#   ./scripts/run_ebm_train.sh 2
#   EBM_STEPS=100000 ./scripts/run_ebm_train.sh 3
#   START_STEP=2 ./scripts/run_ebm_train.sh 3  # skip encoding, start at EBM training
#
# Steps: 1=encode embeddings, 2=EBM training
#
# Environment variables:
#   EBM_STEPS          Training steps (default: 50000)
#   EBM_LR             Learning rate (default: 3e-5)
#   EBM_RESUME         Resume mode: auto|none|/path/to/dir (default: auto)
#   ENCODE_BATCH_SIZE  Batch size for PyTorch encoding (default: 32)
#   HIDDEN_SIZE        Model hidden size (default: 4096)
#   START_STEP         Skip steps: 1=encode+train, 2=train only (default: 1)

set -euo pipefail
export PYTHONUNBUFFERED=1

ITER=${1:?"Usage: ./scripts/run_ebm_train.sh <iteration_number>"}
PREV=$((ITER - 1))

if [ "$ITER" -eq 0 ]; then
    echo "Skipping EBM training for iteration 0 (no trajectory data)"
    exit 0
fi

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib.sh"
LLM_DIR="${REPO_ROOT}/models/llm/iter_${ITER}"
EBM_DIR="${REPO_ROOT}/checkpoints/ebm/iter_${ITER}"
TRAJ_DIR="${REPO_ROOT}/trajectories"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
EBM_STEPS="${EBM_STEPS:-50000}"
EBM_LR="${EBM_LR:-3e-5}"
EBM_RESUME="${EBM_RESUME:-auto}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
HIDDEN_SIZE="${HIDDEN_SIZE:-4096}"
LOSS_TYPE="${LOSS_TYPE:-info_nce}"
MARGIN="${MARGIN:-1}"
START_STEP="${START_STEP:-1}"

PROVER="cargo run --release -p prover-core $CARGO_FEATURES --"

mkdir -p "$EBM_DIR"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"

EMBEDDINGS_FILE="${EBM_DIR}/embeddings.parquet"

# ── Collect trajectory files ───────────────────────────────────────────────
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

echo "================================================================"
echo "  Expert Iteration ${ITER} — EBM Training"
echo "================================================================"
echo "  LLM model:       ${LLM_DIR}"
echo "  EBM output:      ${EBM_DIR}"
echo "  Embeddings:      ${EMBEDDINGS_FILE}"
echo "  Trajectory files: ${#TRAJ_FILES[@]}"
echo "  EBM steps:       ${EBM_STEPS}"
echo "  EBM LR:          ${EBM_LR}"
echo "  EBM resume:      ${EBM_RESUME}"
echo "  Loss type:       ${LOSS_TYPE} (margin: ${MARGIN})"
echo "  Encode batch:    ${ENCODE_BATCH_SIZE}"
echo "  Start step:      ${START_STEP} (1=encode, 2=train)"
echo "================================================================"

# ── Step 1: Encode Embeddings ──────────────────────────────────────────────
if [ "$START_STEP" -gt 1 ]; then
    echo ""
    echo "=== Step 1: Encode Embeddings [SKIPPED — START_STEP=${START_STEP}] ==="
else
    echo ""
    echo "=== Step 1: Encode Embeddings ==="
    echo "  (Direct PyTorch — SGLang batch hidden states broken, see docs/encoding_bug.md)"

    # Stop inference server to free VRAM for direct PyTorch encoding
    stop_inference_server

    STEP_LOG="${LOG_DIR}/iter_${ITER}_ebm_step_1_encode.log"
    echo "  Logging to: ${STEP_LOG}"

    # Resume from existing embeddings if present
    CACHE_FLAG=""
    if [ -f "$EMBEDDINGS_FILE" ]; then
        CACHE_FLAG="--cache ${EMBEDDINGS_FILE}"
        echo "  Warm start: resuming from existing ${EMBEDDINGS_FILE}"
    fi

    # shellcheck disable=SC2086
    run_logged "$STEP_LOG" python3 "${REPO_ROOT}/python/encode_embeddings.py" \
        --model-path "$LLM_DIR" \
        --trajectories ${TRAJ_FILES[*]} \
        --output "$EMBEDDINGS_FILE" \
        --batch-size "$ENCODE_BATCH_SIZE" \
        $CACHE_FLAG

    echo "  Embeddings saved to: ${EMBEDDINGS_FILE}"

    echo "  Server will be restarted after EBM training completes"
fi

# ── Step 2: EBM Training ──────────────────────────────────────────────────
echo ""
echo "=== Step 2: EBM Training ==="

# EBM resume logic
RESUME_FLAG=""
PREV_EBM="${REPO_ROOT}/checkpoints/ebm/iter_${PREV}"
if [ "$EBM_RESUME" = "none" ]; then
    echo "  Resume: disabled (EBM_RESUME=none)"
elif [ "$EBM_RESUME" = "auto" ]; then
    if [ -d "$PREV_EBM" ] && [ -f "${PREV_EBM}/final.mpk" ]; then
        RESUME_FLAG="--resume-from ${PREV_EBM}"
        echo "  Resume: from ${PREV_EBM} (auto)"
    else
        echo "  Resume: none (auto — no previous checkpoint at ${PREV_EBM})"
    fi
elif [ -d "$EBM_RESUME" ] && [ -f "${EBM_RESUME}/final.mpk" ]; then
    RESUME_FLAG="--resume-from ${EBM_RESUME}"
    echo "  Resume: from ${EBM_RESUME} (custom path)"
else
    echo "  ERROR: EBM_RESUME=${EBM_RESUME} but no final.mpk found"
    exit 1
fi

# Pre-computed embeddings cache (all states should be here from Step 1)
EMBEDDINGS_CACHE_FLAG=""
if [ -f "$EMBEDDINGS_FILE" ]; then
    EMBEDDINGS_CACHE_FLAG="--embeddings-cache ${EMBEDDINGS_FILE}"
    echo "  Embeddings cache: ${EMBEDDINGS_FILE}"
else
    echo "  WARNING: No pre-computed embeddings at ${EMBEDDINGS_FILE}"
    echo "           train-ebm will encode via server (slow, sequential)"
fi

# No server needed — all embeddings are pre-computed in Step 1.
# train-ebm still requires --server-url but won't use it if cache is complete.
STEP_LOG="${LOG_DIR}/iter_${ITER}_ebm_step_2_train.log"
echo "  Logging to: ${STEP_LOG}"

# shellcheck disable=SC2086
run_logged "$STEP_LOG" $PROVER train-ebm \
    --trajectories ${TRAJ_FILES[*]} \
    --server-url "${SGLANG_URL}" \
    --hidden-size $HIDDEN_SIZE \
    --output-dir $EBM_DIR \
    --steps $EBM_STEPS \
    --batch-size 256 \
    --lr $EBM_LR \
    --save-embeddings "$EMBEDDINGS_FILE" \
    $EMBEDDINGS_CACHE_FLAG \
    $RESUME_FLAG \
    --encode-batch-size $ENCODE_BATCH_SIZE \
    --encode-concurrency 1 \
    --loss-type $LOSS_TYPE \
    --margin $MARGIN

# Restart inference server for subsequent pipeline steps (search, eval)
restart_inference_server "$SGLANG_URL" "$LLM_DIR"

echo ""
echo "================================================================"
echo "  EBM training complete!"
echo "  Checkpoint:  ${EBM_DIR}"
echo "  Embeddings:  ${EMBEDDINGS_FILE}"
echo "================================================================"
