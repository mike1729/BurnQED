#!/bin/bash
# Run one complete expert iteration.
#
# Usage:
#   ./scripts/run_iteration.sh <iteration_number>
#   ./scripts/run_iteration.sh 0   # First iteration (base model fine-tune)
#   ./scripts/run_iteration.sh 1   # Second iteration (adds trajectory data + EBM)
#   NUM_WORKERS=64 ./scripts/run_iteration.sh 1
#
# Prerequisites:
#   - Rust prover-core built (cargo build --release -p prover-core)
#   - Python deps installed (pip install -r python/requirements.txt)
#   - Model weights downloaded to models/deepseek-prover-v2-7b/
#   - Pantograph built (./scripts/setup_pantograph.sh)
#   - Training data prepared (python python/data/trace_mathlib.py + prepare_tactic_pairs.py)

set -euo pipefail

ITER=${1:?"Usage: ./scripts/run_iteration.sh <iteration_number>"}
PREV=$((ITER - 1))

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLM_BASE="${LLM_BASE:-deepseek-ai/DeepSeek-Prover-V2-7B}"
LLM_DIR="${REPO_ROOT}/models/llm/iter_${ITER}"
EBM_DIR="${REPO_ROOT}/checkpoints/ebm/iter_${ITER}"
TRAJ_DIR="${REPO_ROOT}/trajectories"
EVAL_DIR="${REPO_ROOT}/eval_results"
CKPT_DIR="${REPO_ROOT}/checkpoints/llm"
THEOREM_INDEX="${REPO_ROOT}/data/theorem_index.json"
MINIF2F="${REPO_ROOT}/data/minif2f_test.json"
TRAIN_DATA="${REPO_ROOT}/data/tactic_pairs/train_formatted.jsonl"
VAL_DATA="${REPO_ROOT}/data/tactic_pairs/val_formatted.jsonl"
NUM_WORKERS="${NUM_WORKERS:-64}"
PROVER="cargo run --release -p prover-core --"

mkdir -p "$TRAJ_DIR" "$EVAL_DIR" "$CKPT_DIR" "$LLM_DIR" "$EBM_DIR"
mkdir -p "${REPO_ROOT}/logs"

echo "================================================================"
echo "  Expert Iteration ${ITER}"
echo "================================================================"
echo "  LLM base:       ${LLM_BASE}"
echo "  LLM output:     ${LLM_DIR}"
echo "  EBM output:     ${EBM_DIR}"
echo "  Trajectory dir:  ${TRAJ_DIR}"
echo "  Theorem index:   ${THEOREM_INDEX}"
echo "  Workers:         ${NUM_WORKERS}"
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
        --epochs 3 \
        --lr "$LR"
else
    echo "Iteration ${ITER}: training with trajectory data from previous iterations"

    EXTRA_DATA_ARGS=()
    for i in $(seq 0 "$PREV"); do
        pattern="${TRAJ_DIR}/iter_${i}*.parquet"
        # shellcheck disable=SC2086
        if compgen -G $pattern > /dev/null; then
            EXTRA_DATA_ARGS+=(--extra-data "$pattern")
        fi
    done

    BASE_CKPT="${CKPT_DIR}/iter_${PREV}"

    # shellcheck disable=SC2086
    accelerate launch "${REPO_ROOT}/python/training/train_llm.py" \
        --model-name "$LLM_BASE" \
        --data "$TRAIN_DATA" \
        $VAL_DATA_FLAG \
        "${EXTRA_DATA_ARGS[@]}" \
        --output "${CKPT_DIR}/iter_${ITER}" \
        --base "$BASE_CKPT" \
        --epochs 1 \
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

# ── Step 2: EBM Training (skip iteration 0) ───────────────────────────────
if [ "$ITER" -gt 0 ]; then
    echo ""
    echo "=== Step 2: EBM Training ==="

    TRAJ_FILES=()
    for i in $(seq 0 "$PREV"); do
        pattern="${TRAJ_DIR}/iter_${i}*.parquet"
        # shellcheck disable=SC2086
        for f in $pattern; do
            [ -f "$f" ] && TRAJ_FILES+=("$f")
        done
    done

    RESUME_FLAG=""
    PREV_EBM="${REPO_ROOT}/checkpoints/ebm/iter_${PREV}"
    if [ -d "$PREV_EBM" ] && [ -f "${PREV_EBM}/final.mpk" ]; then
        RESUME_FLAG="--resume-from ${PREV_EBM}"
    fi

    EMBEDDINGS_SAVE="${EBM_DIR}/embeddings.parquet"

    # shellcheck disable=SC2086
    $PROVER train-ebm \
        --trajectories "${TRAJ_FILES[@]}" \
        --llm-path "$LLM_DIR" \
        --output-dir "$EBM_DIR" \
        --steps 50000 \
        --save-embeddings "$EMBEDDINGS_SAVE" \
        $RESUME_FLAG
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
    --model-path "$LLM_DIR" \
    $EBM_FLAG \
    --theorems "$THEOREM_INDEX" \
    --output "$TRAJ_OUTPUT" \
    --num-workers "$NUM_WORKERS"

# ── Step 3b: Noise injection search (iteration 0 only) ────────────────────
if [ "$ITER" -eq 0 ]; then
    echo ""
    echo "=== Step 3b: Noise Injection Search (temperature=1.2) ==="

    NOISY_OUTPUT="${TRAJ_DIR}/iter_0_noisy.parquet"

    $PROVER search \
        --model-path "$LLM_DIR" \
        --temperature 1.2 \
        --theorems "$THEOREM_INDEX" \
        --output "$NOISY_OUTPUT" \
        --num-workers "$NUM_WORKERS"
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
    --model-path "$LLM_DIR" \
    $EBM_FLAG \
    --theorems "$EVAL_THEOREMS" \
    --budgets 100,300,600 \
    --output "${EVAL_DIR}/iter_${ITER}.json" \
    --num-workers "$NUM_WORKERS"

# ── Step 4b: EBM Ablation (iter > 0 — eval WITHOUT EBM) ──────────────────
if [ "$ITER" -gt 0 ] && [ -n "$EBM_FLAG" ]; then
    echo ""
    echo "=== Step 4b: EBM Ablation (eval WITHOUT EBM) ==="

    $PROVER eval \
        --model-path "$LLM_DIR" \
        --theorems "$EVAL_THEOREMS" \
        --budgets 100,300,600 \
        --output "${EVAL_DIR}/iter_${ITER}_no_ebm.json" \
        --num-workers "$NUM_WORKERS"
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
echo "  Iteration ${ITER} complete!"
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
