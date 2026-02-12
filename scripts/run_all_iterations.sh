#!/bin/bash
# Run the full BurnQED expert iteration experiment.
#
# Phases:
#   B: Baseline raw model evaluation
#   C-E: Expert iterations 0 through MAX_ITER
#   F: Final cross-iteration comparison and ablation
#
# Each iteration's output is logged to logs/iter_N.log.
# By default runs 5 iterations (0-4). Override with MAX_ITER env var.
#
# Usage:
#   ./scripts/run_all_iterations.sh
#   MAX_ITER=2 ./scripts/run_all_iterations.sh        # Only iterations 0-2
#   SKIP_BASELINE=1 ./scripts/run_all_iterations.sh   # Skip Phase B
#   NUM_WORKERS=64 ./scripts/run_all_iterations.sh    # Override worker count

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAX_ITER=${MAX_ITER:-4}
SKIP_BASELINE=${SKIP_BASELINE:-0}
export NUM_WORKERS="${NUM_WORKERS:-30}"
export CONCURRENCY="${CONCURRENCY:-8}"
export MAX_THEOREMS="${MAX_THEOREMS:-2000}"
export EBM_STEPS="${EBM_STEPS:-10000}"

mkdir -p "${REPO_ROOT}/logs"

echo "================================================================"
echo "  BurnQED Expert Iteration Experiment"
echo "  Running iterations 0 through ${MAX_ITER}"
echo "  Workers: ${NUM_WORKERS}"
echo "  Concurrency: ${CONCURRENCY}"
echo "  Max theorems: ${MAX_THEOREMS}"
echo "  EBM steps: ${EBM_STEPS}"
echo "================================================================"

# ── Phase B: Baseline ────────────────────────────────────────────────────
if [ "$SKIP_BASELINE" -eq 0 ]; then
    echo ""
    echo ">>> Phase B: Running baseline evaluation..."
    LOG_FILE="${REPO_ROOT}/logs/baseline.log"

    if "${REPO_ROOT}/scripts/run_baseline.sh" 2>&1 | tee "$LOG_FILE"; then
        echo ">>> Phase B completed successfully."
    else
        echo ">>> Phase B FAILED. Check ${LOG_FILE} for details."
        echo ">>> To skip baseline: SKIP_BASELINE=1 ./scripts/run_all_iterations.sh"
        exit 1
    fi
else
    echo ""
    echo ">>> Phase B: Skipped (SKIP_BASELINE=1)"
fi

# ── Phase C-E: Expert Iterations ─────────────────────────────────────────
for i in $(seq 0 "$MAX_ITER"); do
    LOG_FILE="${REPO_ROOT}/logs/iter_${i}.log"

    echo ""
    echo ">>> Starting iteration ${i} (log: ${LOG_FILE})"
    echo ""

    if "${REPO_ROOT}/scripts/run_iteration.sh" "$i" 2>&1 | tee "$LOG_FILE"; then
        echo ">>> Iteration ${i} completed successfully."
    else
        echo ">>> Iteration ${i} FAILED. Check ${LOG_FILE} for details."
        echo ">>> To resume: ./scripts/resume_search.sh ${i}"
        exit 1
    fi
done

# ── Phase F: Final Analysis ──────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Phase F: Final Cross-Iteration Comparison"
echo "================================================================"

# Auto-detect CUDA
CUDA_FEATURES=$(command -v nvidia-smi &>/dev/null && echo "--features cuda" || echo "")
PROVER="cargo run --release -p prover-core ${CUDA_FEATURES} --"
EVAL_DIR="${REPO_ROOT}/eval_results"
BASELINES_DIR="${REPO_ROOT}/baselines"

# F1. Full cross-iteration comparison (including baseline if available)
COMPARE_ARGS=()
if [ -f "${BASELINES_DIR}/raw_minif2f.json" ]; then
    COMPARE_ARGS+=("${BASELINES_DIR}/raw_minif2f.json")
fi
for i in $(seq 0 "$MAX_ITER"); do
    if [ -f "${EVAL_DIR}/iter_${i}.json" ]; then
        COMPARE_ARGS+=("${EVAL_DIR}/iter_${i}.json")
    fi
done

if [ "${#COMPARE_ARGS[@]}" -ge 2 ]; then
    echo ""
    echo "=== F1: Cross-Iteration Comparison ==="
    $PROVER compare --results "${COMPARE_ARGS[@]}"
fi

# F2. Final EBM ablation on last iteration
LAST_ITER_EVAL="${EVAL_DIR}/iter_${MAX_ITER}.json"
LAST_ITER_NO_EBM="${EVAL_DIR}/iter_${MAX_ITER}_no_ebm.json"

if [ -f "$LAST_ITER_EVAL" ] && [ -f "$LAST_ITER_NO_EBM" ]; then
    echo ""
    echo "=== F2: Final EBM Ablation (iter ${MAX_ITER}) ==="
    $PROVER compare --results "$LAST_ITER_NO_EBM" "$LAST_ITER_EVAL"
fi

# F2b. Baseline EBM artifact note
BASELINE_EBM="${REPO_ROOT}/checkpoints/ebm/baseline"
if [ -d "$BASELINE_EBM" ] && [ -f "${BASELINE_EBM}/final.mpk" ]; then
    echo ""
    echo "  Baseline EBM: checkpoints/ebm/baseline/"
fi

# F3. Print results table
echo ""
echo "================================================================"
echo "  Results Summary"
echo "================================================================"
echo ""
echo "  Iteration  │ EBM │ Results File"
echo "  ───────────┼─────┼─────────────────────"

if [ -f "${BASELINES_DIR}/raw_minif2f.json" ]; then
    echo "  Raw base   │ No  │ baselines/raw_minif2f.json"
fi
if [ -d "$BASELINE_EBM" ] && [ -f "${BASELINE_EBM}/final.mpk" ]; then
    echo "  Baseline   │ Yes │ checkpoints/ebm/baseline/"
fi
for i in $(seq 0 "$MAX_ITER"); do
    if [ -f "${EVAL_DIR}/iter_${i}.json" ]; then
        if [ "$i" -eq 0 ]; then
            echo "  Iter ${i}     │ No  │ eval_results/iter_${i}.json"
        else
            echo "  Iter ${i}     │ Yes │ eval_results/iter_${i}.json"
        fi
    fi
    if [ -f "${EVAL_DIR}/iter_${i}_no_ebm.json" ]; then
        echo "  Iter ${i} abl │ No  │ eval_results/iter_${i}_no_ebm.json"
    fi
done

# F4. List all artifacts
echo ""
echo "  Artifacts:"
echo "  ─────────"
for d in baselines eval_results trajectories checkpoints/llm checkpoints/ebm models/llm; do
    full_path="${REPO_ROOT}/${d}"
    if [ -d "$full_path" ]; then
        count=$(find "$full_path" -maxdepth 1 -type f 2>/dev/null | wc -l)
        echo "  ${d}/: ${count} files"
    fi
done

echo ""
echo "================================================================"
echo "  Experiment complete! ${MAX_ITER} iterations finished."
echo "================================================================"
