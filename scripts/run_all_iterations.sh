#!/bin/bash
# Run all expert iterations sequentially.
#
# Each iteration's output is logged to logs/iter_N.log.
# By default runs 5 iterations (0-4). Override with MAX_ITER env var.
#
# Usage:
#   ./scripts/run_all_iterations.sh
#   MAX_ITER=2 ./scripts/run_all_iterations.sh   # Only iterations 0-2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAX_ITER=${MAX_ITER:-4}

mkdir -p "${REPO_ROOT}/logs"

echo "================================================================"
echo "  BurnQED Expert Iteration Loop"
echo "  Running iterations 0 through ${MAX_ITER}"
echo "================================================================"

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

echo ""
echo "================================================================"
echo "  All ${MAX_ITER+1} iterations complete!"
echo ""
echo "  Compare results:"
echo "    cargo run --release -p prover-core -- compare \\"

COMPARE_ARGS=""
for i in $(seq 0 "$MAX_ITER"); do
    COMPARE_ARGS="${COMPARE_ARGS} ${REPO_ROOT}/eval_results/iter_${i}.json"
done
echo "      --results ${COMPARE_ARGS}"

echo "================================================================"
