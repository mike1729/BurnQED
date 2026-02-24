#!/bin/bash
# Encode embeddings and train EBM for an expert iteration.
#
# Standalone script: can be run independently or called from run_iteration_train.sh.
# Uses the encode server (nf4 quantized, ~5GB VRAM) for embedding extraction,
# which can run alongside the inference server.
#
# Usage:
#   ./scripts/run_ebm_train.sh <iteration_number>
#   ./scripts/run_ebm_train.sh 2
#   EBM_STEPS=100000 ./scripts/run_ebm_train.sh 3
#   START_STEP=2 ./scripts/run_ebm_train.sh 3  # skip encoding, start at EBM training
#
# Steps: 1=encode embeddings (via encode server), 2=EBM training
#
# Environment variables:
#   EBM_STEPS          Training steps (default: 50000)
#   EBM_LR             Learning rate (default: 3e-5)
#   EBM_RESUME         Resume mode: auto|none|/path/to/dir (default: auto)
#   ENCODE_BATCH_SIZE  Batch size for encoding (default: 32)
#   ENCODE_URL         Encode server URL (default: http://localhost:30001)
#   HIDDEN_SIZE        Model hidden size (default: 4096)
#   HARD_RATIO         Fraction of hard (sibling) negatives (default: 0.3)
#   MEDIUM_RATIO       Fraction of medium (same-theorem) negatives (default: 0.4)
#   DROPOUT            Dropout probability for energy head (default: 0.1)
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
ENCODE_URL="${ENCODE_URL:-http://localhost:30001}"
EBM_STEPS="${EBM_STEPS:-50000}"
EBM_LR="${EBM_LR:-3e-5}"
EBM_RESUME="${EBM_RESUME:-auto}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
HIDDEN_SIZE="${HIDDEN_SIZE:-4096}"
LOSS_TYPE="${LOSS_TYPE:-info_nce}"
MARGIN="${MARGIN:-1}"
HARD_RATIO="${HARD_RATIO:-0.3}"
MEDIUM_RATIO="${MEDIUM_RATIO:-0.4}"
DROPOUT="${DROPOUT:-0.1}"
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
echo "  Hard ratio:      ${HARD_RATIO}"
echo "  Medium ratio:    ${MEDIUM_RATIO}"
echo "  Dropout:         ${DROPOUT}"
echo "  Encode batch:    ${ENCODE_BATCH_SIZE}"
echo "  Encode server:   ${ENCODE_URL}"
echo "  Start step:      ${START_STEP} (1=encode, 2=train)"
echo "================================================================"

# ── Step 1: Encode Embeddings ──────────────────────────────────────────────
if [ "$START_STEP" -gt 1 ]; then
    echo ""
    echo "=== Step 1: Encode Embeddings [SKIPPED — START_STEP=${START_STEP}] ==="
else
    echo ""
    echo "=== Step 1: Encode Embeddings ==="
    echo "  (via encode server at ${ENCODE_URL})"

    # Kill inference server to free VRAM for encoding throughput — encode server
    # needs full GPU bandwidth without contention from inference server
    stop_inference_server

    # Start encode server if not already running
    if ! curl -sf "${ENCODE_URL}/health" > /dev/null 2>&1; then
        echo "  Starting encode server (nf4, ~5GB VRAM)..."
        ENCODE_DTYPE=nf4 "${REPO_ROOT}/scripts/start_encode_server.sh" "$LLM_DIR" &
        ENCODE_PID=$!
        echo "  Waiting for encode server to be ready..."
        for i in $(seq 1 120); do
            if curl -sf "${ENCODE_URL}/health" > /dev/null 2>&1; then
                echo "  Encode server ready (${i}s)"
                break
            fi
            if ! kill -0 $ENCODE_PID 2>/dev/null; then
                echo "ERROR: Encode server process died"
                exit 1
            fi
            sleep 1
        done
        if ! curl -sf "${ENCODE_URL}/health" > /dev/null 2>&1; then
            echo "ERROR: Encode server failed to start within 120s"
            exit 1
        fi
    else
        ENCODE_PID=""
        echo "  Encode server already running"
    fi

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
        --encode-url "$ENCODE_URL" \
        --trajectories ${TRAJ_FILES[*]} \
        --output "$EMBEDDINGS_FILE" \
        --batch-size "$ENCODE_BATCH_SIZE" \
        --hidden-size "$HIDDEN_SIZE" \
        $CACHE_FLAG

    echo "  Embeddings saved to: ${EMBEDDINGS_FILE}"

    # Stop encode server to free VRAM for EBM training
    if [ -n "${ENCODE_PID:-}" ] && kill -0 "$ENCODE_PID" 2>/dev/null; then
        echo "  Stopping encode server (pid ${ENCODE_PID})..."
        kill "$ENCODE_PID" 2>/dev/null || true
        wait "$ENCODE_PID" 2>/dev/null || true
    fi
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
    --margin $MARGIN \
    --hard-ratio $HARD_RATIO \
    --medium-ratio $MEDIUM_RATIO \
    --dropout $DROPOUT

echo ""
echo "================================================================"
echo "  EBM training complete!"
echo "  Checkpoint:  ${EBM_DIR}"
echo "  Embeddings:  ${EMBEDDINGS_FILE}"
echo "================================================================"
