#!/bin/bash
# Great Reset: revert to base DeepSeek-Prover-V2-7B, train fresh EBM + attention-only LoRA.
#
# All fine-tuned weights across iter_0→3 progressively destroyed hidden-state
# geometry (MLP LoRA norms 70-110% larger than attention, negative embedding
# separation by iter_3). This script discards all LoRA weights, returns to the
# pristine base model, and retrains from scratch using all 794K trajectory
# records (verified by Lean).
#
# Phases:
#   0. Base-model baseline: encode states → train EBM → 100-theorem search
#   1. Kill SGLang → train attention-only LoRA → export merged model
#   2. Start merged model → re-encode states → train fresh EBM on merged embeddings
#   3. Proof search (merged model + merged-model EBM)
#   4. Eval (miniF2F with/without EBM ablation)
#   5. Summary + comparisons (vs iter_2, vs base-model baseline, EBM ablation)
#
# Usage:
#   ./scripts/great_reset.sh
#   START_STEP=1 ./scripts/great_reset.sh   # Skip baseline, start at LoRA training
#   START_STEP=3 ./scripts/great_reset.sh   # Skip to search
#
# Steps: 0=baseline, 1=LoRA, 2=EBM, 3=search, 4=eval, 5=summary

set -euo pipefail
export PYTHONUNBUFFERED=1

# Ensure cargo is in PATH (needed by run_logged which uses script(1))
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib.sh"

# ── Configuration ─────────────────────────────────────────────────────────
TAG="reset_1"
BASE_MODEL="${REPO_ROOT}/models/deepseek-prover-v2-7b"
LLM_BASE="deepseek-ai/DeepSeek-Prover-V2-7B"  # HF name for export script
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"

# Output paths (all new dirs, old data left intact)
EBM_BASE_DIR="${REPO_ROOT}/checkpoints/ebm/reset"         # base-model EBM
EBM_LORA_DIR="${REPO_ROOT}/checkpoints/ebm/reset_lora"     # merged-model EBM (production)
CKPT_DIR="${REPO_ROOT}/checkpoints/llm/${TAG}"
LLM_DIR="${REPO_ROOT}/models/llm/${TAG}"
TRAJ_OUTPUT="${REPO_ROOT}/trajectories/${TAG}.parquet"
TRAJ_BASELINE="${REPO_ROOT}/trajectories/reset_base_only.parquet"
EVAL_DIR="${REPO_ROOT}/eval_results"
LOG_DIR="${REPO_ROOT}/logs"
TRAJ_DIR="${REPO_ROOT}/trajectories"

# EBM parameters (matched to iter_2's successful run)
EBM_STEPS="${EBM_STEPS:-30000}"
EBM_LR="${EBM_LR:-3e-5}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-64}"
ENCODE_CONCURRENCY="${ENCODE_CONCURRENCY:-2}"

# LoRA parameters
LLM_LR="${LLM_LR:-2e-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
BASE_SUBSAMPLE="${BASE_SUBSAMPLE:-50000}"

# Search/eval parameters
CONCURRENCY="${CONCURRENCY:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_THEOREMS="${MAX_THEOREMS:-2000}"
BASELINE_MAX_THEOREMS="${BASELINE_MAX_THEOREMS:-100}"
EVAL_MAX_THEOREMS="${EVAL_MAX_THEOREMS:-500}"
SEARCH_CONFIG="${REPO_ROOT}/configs/search.toml"
SEARCH_THEOREMS="${REPO_ROOT}/data/theorem_index.json"
MINIF2F="${REPO_ROOT}/data/minif2f_test.json"
TRAIN_DATA="${REPO_ROOT}/data/tactic_pairs/train_formatted.jsonl"
VAL_DATA="${REPO_ROOT}/data/tactic_pairs/val_formatted.jsonl"
PROVER="cargo run --release -p prover-core $CARGO_FEATURES --"

START_STEP="${START_STEP:-0}"

mkdir -p "$EBM_BASE_DIR" "$EBM_LORA_DIR" "$CKPT_DIR" "$LLM_DIR" "$EVAL_DIR" "$LOG_DIR"

echo "================================================================"
echo "  Great Reset: Base Model + Fresh EBM + Attention-Only LoRA"
echo "================================================================"
echo "  Tag:               ${TAG}"
echo "  Base model:        ${BASE_MODEL}"
echo "  Base-model EBM:    ${EBM_BASE_DIR}"
echo "  Merged-model EBM:  ${EBM_LORA_DIR}"
echo "  LoRA checkpoint:   ${CKPT_DIR}"
echo "  Merged model:      ${LLM_DIR}"
echo "  Trajectory out:    ${TRAJ_OUTPUT}"
echo "  Baseline traj:     ${TRAJ_BASELINE}"
echo "  EBM:               ${EBM_STEPS} steps, lr=${EBM_LR}, batch=256"
echo "  LoRA:              ${MAX_TRAIN_STEPS} steps, lr=${LLM_LR}, attn-only"
echo "  Start step:        ${START_STEP} (0=baseline, 1=LoRA, 2=EBM, 3=search, 4=eval, 5=summary)"
echo "================================================================"

# ── Collect ALL trajectory files ──────────────────────────────────────────
# Include everything except tiny test/debug/smoke artifacts.
TRAJ_FILES=()
for f in "${TRAJ_DIR}"/*.parquet; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    case "$base" in
        *_test.parquet|*_debug.parquet|*_smoke.parquet) continue ;;
    esac
    TRAJ_FILES+=("$f")
done

echo ""
echo "Trajectory files (${#TRAJ_FILES[@]}):"
for f in "${TRAJ_FILES[@]}"; do
    fsize=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    echo "  $(basename "$f")  ($((fsize / 1048576))MB)"
done

if [ ${#TRAJ_FILES[@]} -eq 0 ]; then
    echo "ERROR: No trajectory files found in ${TRAJ_DIR}"
    exit 1
fi

# Quick data summary
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
" "${TRAJ_FILES[@]}"

# ══════════════════════════════════════════════════════════════════════════
# Step 0: Base-Model EBM Baseline
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -gt 0 ]; then
    echo ""
    echo "=== Step 0: Base-Model Baseline [SKIPPED — START_STEP=${START_STEP}] ==="
else
    echo ""
    echo "=== Step 0a: Start Base Model Server ==="
    ensure_server "$SGLANG_URL" "$BASE_MODEL"

    echo ""
    echo "=== Step 0b: Encode States with Base Model ==="

    EMBEDDINGS_BASE="${EBM_BASE_DIR}/embeddings.parquet"
    EMBEDDINGS_CACHE_FLAG=""
    if [ -f "$EMBEDDINGS_BASE" ]; then
        EMBEDDINGS_CACHE_FLAG="--embeddings-cache ${EMBEDDINGS_BASE}"
        echo "  Warm start: reusing existing embeddings from ${EMBEDDINGS_BASE}"
    fi

    echo ""
    echo "=== Step 0c: Train EBM on Base-Model Embeddings ==="

    STEP_LOG="${LOG_DIR}/${TAG}_step_0_ebm_base.log"
    echo "  Logging to: ${STEP_LOG}"

    # shellcheck disable=SC2086
    run_logged "$STEP_LOG" $PROVER train-ebm \
        --trajectories ${TRAJ_FILES[*]} \
        --server-url $SGLANG_URL \
        --hidden-size ${HIDDEN_SIZE:-4096} \
        --output-dir $EBM_BASE_DIR \
        --steps $EBM_STEPS \
        --batch-size 256 \
        --lr $EBM_LR \
        --save-embeddings $EMBEDDINGS_BASE \
        $EMBEDDINGS_CACHE_FLAG \
        --encode-batch-size $ENCODE_BATCH_SIZE \
        --encode-concurrency $ENCODE_CONCURRENCY

    echo "  EBM checkpoint: ${EBM_BASE_DIR}/final.mpk"
    echo "  Embeddings: ${EMBEDDINGS_BASE}"

    echo ""
    echo "=== Step 0d: Baseline Search (${BASELINE_MAX_THEOREMS} theorems, base model + base EBM) ==="

    BASELINE_EBM_FLAG=""
    if [ -f "${EBM_BASE_DIR}/final.mpk" ]; then
        BASELINE_EBM_FLAG="--ebm-path ${EBM_BASE_DIR}"
    fi

    STEP_LOG="${LOG_DIR}/${TAG}_step_0d_baseline_search.log"
    echo "  Logging to: ${STEP_LOG}"

    # shellcheck disable=SC2086
    run_logged "$STEP_LOG" $PROVER search \
        --config $SEARCH_CONFIG \
        --server-url $SGLANG_URL \
        $BASELINE_EBM_FLAG \
        --theorems $SEARCH_THEOREMS \
        --output $TRAJ_BASELINE \
        --num-workers $NUM_WORKERS \
        --concurrency $CONCURRENCY \
        --max-theorems $BASELINE_MAX_THEOREMS \
        --imports Mathlib

    echo ""
    echo "=== Step 0e: Baseline Summary ==="
    if [ -f "$TRAJ_BASELINE" ]; then
        $PROVER summary --input "$TRAJ_BASELINE"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Attention-Only LoRA Training
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -gt 1 ]; then
    echo ""
    echo "=== Step 1: LoRA Training [SKIPPED — START_STEP=${START_STEP}] ==="
else
    echo ""
    echo "=== Step 1a: Kill SGLang (free VRAM for training) ==="
    stop_inference_server

    echo ""
    echo "=== Step 1b: Attention-Only LoRA Training ==="

    # Collect trajectory files as --extra-data args
    EXTRA_DATA_ARGS=()
    for f in "${TRAJ_FILES[@]}"; do
        EXTRA_DATA_ARGS+=(--extra-data "$f")
    done

    VAL_DATA_FLAG=""
    if [ -f "$VAL_DATA" ]; then
        VAL_DATA_FLAG="--val-data ${VAL_DATA}"
    fi

    STEP_LOG="${LOG_DIR}/${TAG}_step_1_lora.log"
    echo "  Model:    ${BASE_MODEL} (pristine base)"
    echo "  Targets:  q_proj, k_proj, v_proj, o_proj (attention-only)"
    echo "  LR:       ${LLM_LR}"
    echo "  Steps:    ${MAX_TRAIN_STEPS}"
    echo "  Logging:  ${STEP_LOG}"

    # Fresh LoRA on base model — no --base flag, no --lora-mlp
    # shellcheck disable=SC2086
    run_logged "$STEP_LOG" accelerate launch \
        ${REPO_ROOT}/python/training/train_llm.py \
        --model-name $BASE_MODEL \
        --data $TRAIN_DATA \
        $VAL_DATA_FLAG \
        ${EXTRA_DATA_ARGS[*]} \
        --output $CKPT_DIR \
        --max-steps $MAX_TRAIN_STEPS \
        --base-subsample $BASE_SUBSAMPLE \
        --batch-size $BATCH_SIZE \
        --gradient-accumulation $GRAD_ACCUM \
        --lr $LLM_LR

    echo ""
    echo "=== Step 1c: Export Merged Model ==="
    STEP_LOG="${LOG_DIR}/${TAG}_step_1c_export.log"
    echo "  Logging to: ${STEP_LOG}"
    run_logged "$STEP_LOG" python3 \
        ${REPO_ROOT}/python/training/export_llm.py \
        --checkpoint $CKPT_DIR \
        --base-model $LLM_BASE \
        --output $LLM_DIR \
        --verify

    echo "  Merged model: ${LLM_DIR}"
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 2: EBM Training on Merged-Model Embeddings
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -gt 2 ]; then
    echo ""
    echo "=== Step 2: Merged-Model EBM [SKIPPED — START_STEP=${START_STEP}] ==="
else
    echo ""
    echo "=== Step 2a: Start Merged-Model Server ==="
    ensure_server "$SGLANG_URL" "$LLM_DIR"

    echo ""
    echo "=== Step 2b: Encode States with Merged Model ==="

    EMBEDDINGS_LORA="${EBM_LORA_DIR}/embeddings.parquet"
    EMBEDDINGS_CACHE_FLAG=""
    if [ -f "$EMBEDDINGS_LORA" ]; then
        EMBEDDINGS_CACHE_FLAG="--embeddings-cache ${EMBEDDINGS_LORA}"
        echo "  Warm start: reusing existing embeddings from ${EMBEDDINGS_LORA}"
    fi

    echo ""
    echo "=== Step 2c: Train EBM on Merged-Model Embeddings ==="

    STEP_LOG="${LOG_DIR}/${TAG}_step_2_ebm_lora.log"
    echo "  Logging to: ${STEP_LOG}"

    # shellcheck disable=SC2086
    run_logged "$STEP_LOG" $PROVER train-ebm \
        --trajectories ${TRAJ_FILES[*]} \
        --server-url $SGLANG_URL \
        --hidden-size ${HIDDEN_SIZE:-4096} \
        --output-dir $EBM_LORA_DIR \
        --steps $EBM_STEPS \
        --batch-size 256 \
        --lr $EBM_LR \
        --save-embeddings $EMBEDDINGS_LORA \
        $EMBEDDINGS_CACHE_FLAG \
        --encode-batch-size $ENCODE_BATCH_SIZE \
        --encode-concurrency $ENCODE_CONCURRENCY

    echo "  EBM checkpoint: ${EBM_LORA_DIR}/final.mpk"
    echo "  Embeddings: ${EMBEDDINGS_LORA}"
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Proof Search (merged model + merged-model EBM)
# ══════════════════════════════════════════════════════════════════════════

# EBM flag for search and eval — use merged-model EBM (production)
EBM_FLAG=""
if [ -d "$EBM_LORA_DIR" ] && [ -f "${EBM_LORA_DIR}/final.mpk" ]; then
    EBM_FLAG="--ebm-path ${EBM_LORA_DIR}"
fi

if [ "$START_STEP" -gt 3 ]; then
    echo ""
    echo "=== Step 3: Proof Search [SKIPPED — START_STEP=${START_STEP}] ==="
else
    echo ""
    echo "=== Step 3: Proof Search ==="

    # Ensure server is running with the merged model
    ensure_server "$SGLANG_URL" "$LLM_DIR"

    if [ ! -f "$SEARCH_THEOREMS" ]; then
        echo "ERROR: Search theorem file not found: ${SEARCH_THEOREMS}"
        exit 1
    fi

    STEP_LOG="${LOG_DIR}/${TAG}_step_3_search.log"
    echo "  Theorems: ${SEARCH_THEOREMS}"
    echo "  EBM:      ${EBM_FLAG:-none}"
    echo "  Logging:  ${STEP_LOG}"

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

    echo "  Trajectory: ${TRAJ_OUTPUT}"
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 4: Evaluation
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -gt 4 ]; then
    echo ""
    echo "=== Step 4: Evaluation [SKIPPED — START_STEP=${START_STEP}] ==="
else
    echo ""
    echo "=== Step 4a: Evaluation (with EBM) ==="

    ensure_server "$SGLANG_URL" "$LLM_DIR"

    if [ -f "$MINIF2F" ]; then
        EVAL_THEOREMS="$MINIF2F"
    else
        echo "Warning: miniF2F not found at ${MINIF2F}, using theorem_index.json"
        EVAL_THEOREMS="${REPO_ROOT}/data/theorem_index.json"
    fi

    STEP_LOG="${LOG_DIR}/${TAG}_step_4_eval.log"
    echo "  Theorems: ${EVAL_THEOREMS}"
    echo "  EBM:      ${EBM_FLAG:-none}"
    echo "  Logging:  ${STEP_LOG}"

    # shellcheck disable=SC2086
    run_logged "$STEP_LOG" $PROVER eval \
        --config $SEARCH_CONFIG \
        --server-url $SGLANG_URL \
        $EBM_FLAG \
        --theorems $EVAL_THEOREMS \
        --budgets 600 \
        --output ${EVAL_DIR}/${TAG}.json \
        --num-workers $NUM_WORKERS \
        --concurrency $CONCURRENCY \
        --max-theorems $EVAL_MAX_THEOREMS \
        --num-candidates 16 \
        --imports Mathlib

    # Ablation: eval WITHOUT EBM
    if [ -n "$EBM_FLAG" ]; then
        echo ""
        echo "=== Step 4b: EBM Ablation (eval WITHOUT EBM) ==="
        STEP_LOG="${LOG_DIR}/${TAG}_step_4b_ablation.log"
        echo "  Logging to: ${STEP_LOG}"

        # shellcheck disable=SC2086
        run_logged "$STEP_LOG" $PROVER eval \
            --config $SEARCH_CONFIG \
            --server-url $SGLANG_URL \
            --theorems $EVAL_THEOREMS \
            --budgets 600 \
            --output ${EVAL_DIR}/${TAG}_no_ebm.json \
            --num-workers $NUM_WORKERS \
            --concurrency $CONCURRENCY \
            --max-theorems $EVAL_MAX_THEOREMS \
            --num-candidates 16 \
            --imports Mathlib
    fi
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 5: Summary & Comparison
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Step 5: Summary ==="

if [ -f "$TRAJ_OUTPUT" ]; then
    $PROVER summary --input "$TRAJ_OUTPUT"
fi

# Compare with iter_2 (last good iteration)
ITER2_EVAL="${EVAL_DIR}/iter_2.json"
RESET_EVAL="${EVAL_DIR}/${TAG}.json"
if [ -f "$ITER2_EVAL" ] && [ -f "$RESET_EVAL" ]; then
    echo ""
    echo "=== Comparison: reset_1 vs iter_2 ==="
    $PROVER compare --results "$ITER2_EVAL" "$RESET_EVAL"
fi

# Compare with base-model baseline
if [ -f "$TRAJ_BASELINE" ] && [ -f "$TRAJ_OUTPUT" ]; then
    echo ""
    echo "=== Comparison: reset_1 vs base-model baseline ==="
    echo "  Baseline (base model, ${BASELINE_MAX_THEOREMS} theorems):"
    $PROVER summary --input "$TRAJ_BASELINE"
    echo "  Full run (merged model, ${MAX_THEOREMS} theorems):"
    $PROVER summary --input "$TRAJ_OUTPUT"
fi

# EBM ablation comparison
NO_EBM_EVAL="${EVAL_DIR}/${TAG}_no_ebm.json"
if [ -f "$NO_EBM_EVAL" ] && [ -f "$RESET_EVAL" ]; then
    echo ""
    echo "=== EBM Ablation: with vs without EBM ==="
    $PROVER compare --results "$NO_EBM_EVAL" "$RESET_EVAL"
fi

# Verify LoRA adapter config
ADAPTER_CONFIG="${CKPT_DIR}/adapter_config.json"
if [ -f "$ADAPTER_CONFIG" ]; then
    echo ""
    echo "=== LoRA Adapter Verification ==="
    echo "  Target modules:"
    python3 -c "
import json
cfg = json.load(open('$ADAPTER_CONFIG'))
modules = cfg.get('target_modules', [])
print('    ' + ', '.join(modules))
has_mlp = any(m in modules for m in ['gate_proj', 'up_proj', 'down_proj'])
print(f'    MLP layers: {\"PRESENT (unexpected!)\" if has_mlp else \"absent (correct)\"}')
print(f'    Rank: {cfg.get(\"r\", \"?\")}')
print(f'    Alpha: {cfg.get(\"lora_alpha\", \"?\")}')
"
fi

echo ""
echo "================================================================"
echo "  Great Reset Complete!"
echo "================================================================"
echo "  Base-model EBM:    ${EBM_BASE_DIR}/final.mpk"
echo "  Merged-model EBM:  ${EBM_LORA_DIR}/final.mpk"
echo "  LoRA checkpoint:   ${CKPT_DIR}"
echo "  Merged model:      ${LLM_DIR}"
if [ -f "$TRAJ_BASELINE" ]; then
    echo "  Baseline traj:    ${TRAJ_BASELINE}"
fi
if [ -f "$TRAJ_OUTPUT" ]; then
    echo "  Trajectory:        ${TRAJ_OUTPUT}"
fi
if [ -f "${EVAL_DIR}/${TAG}.json" ]; then
    echo "  Eval:              ${EVAL_DIR}/${TAG}.json"
fi
if [ -f "${EVAL_DIR}/${TAG}_no_ebm.json" ]; then
    echo "  Ablation:          ${EVAL_DIR}/${TAG}_no_ebm.json"
fi
echo "================================================================"
echo ""
echo "  Next steps:"
echo "    1. Check EBM rank reached 0.50+ (see log)"
echo "    2. Run: python scripts/compare_embeddings.py  (verify Δ cosine > 0)"
echo "    3. Compare eval_results/${TAG}.json vs eval_results/iter_2.json"
echo ""
