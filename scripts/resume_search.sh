#!/bin/bash
# Resume a search after spot instance interruption.
#
# Checks if a partial trajectory Parquet exists for the given iteration.
# If yes: resumes search with --resume-from.
# If no: starts a fresh iteration via run_iteration.sh.
#
# Usage:
#   ./scripts/resume_search.sh <iteration_number>

set -euo pipefail

ITER=${1:?"Usage: ./scripts/resume_search.sh <iteration_number>"}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAJ_DIR="${REPO_ROOT}/trajectories"
TRAJ_OUTPUT="${TRAJ_DIR}/iter_${ITER}.parquet"
LLM_DIR="${REPO_ROOT}/models/llm/iter_${ITER}"
EBM_DIR="${REPO_ROOT}/checkpoints/ebm/iter_${ITER}"
THEOREM_INDEX="${REPO_ROOT}/data/theorem_index.json"
NUM_WORKERS="${NUM_WORKERS:-64}"
PROVER="cargo run --release -p prover-core --"

echo "================================================================"
echo "  Resume Search — Iteration ${ITER}"
echo "================================================================"

# Check if LLM export exists (Steps 1-1b must be complete)
if [ ! -d "$LLM_DIR" ] || [ -z "$(ls -A "$LLM_DIR"/*.safetensors 2>/dev/null)" ]; then
    echo "LLM weights not found at ${LLM_DIR}"
    echo "Running full iteration from the start..."
    exec "${REPO_ROOT}/scripts/run_iteration.sh" "$ITER"
fi

# Check for partial trajectory
if [ -f "$TRAJ_OUTPUT" ]; then
    DONE_COUNT=$(python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('${TRAJ_OUTPUT}', columns=['theorem_name'])
names = set(t.column('theorem_name').to_pylist())
print(len(names))
" 2>/dev/null || echo "0")

    TOTAL_COUNT=$(python3 -c "
import json
with open('${THEOREM_INDEX}') as f:
    print(len(json.load(f)['theorems']))
")

    echo "Found partial trajectory: ${DONE_COUNT}/${TOTAL_COUNT} theorems done"

    if [ "$DONE_COUNT" -ge "$TOTAL_COUNT" ]; then
        echo "All theorems already searched! Skipping to evaluation."
    else
        echo "Resuming search from partial results..."

        EBM_FLAG=""
        if [ "$ITER" -gt 0 ] && [ -d "$EBM_DIR" ] && [ -f "${EBM_DIR}/final.mpk" ]; then
            EBM_FLAG="--ebm-path ${EBM_DIR}"
        fi

        # shellcheck disable=SC2086
        $PROVER search \
            --model-path "$LLM_DIR" \
            $EBM_FLAG \
            --theorems "$THEOREM_INDEX" \
            --output "$TRAJ_OUTPUT" \
            --resume-from "$TRAJ_OUTPUT" \
            --num-workers "$NUM_WORKERS"
    fi
else
    echo "No partial trajectory found. Starting search from scratch..."

    EBM_FLAG=""
    if [ "$ITER" -gt 0 ] && [ -d "$EBM_DIR" ] && [ -f "${EBM_DIR}/final.mpk" ]; then
        EBM_FLAG="--ebm-path ${EBM_DIR}"
    fi

    # shellcheck disable=SC2086
    $PROVER search \
        --model-path "$LLM_DIR" \
        $EBM_FLAG \
        --theorems "$THEOREM_INDEX" \
        --output "$TRAJ_OUTPUT" \
        --num-workers "$NUM_WORKERS"
fi

# Summary
echo ""
echo "=== Trajectory Summary ==="
$PROVER summary --input "$TRAJ_OUTPUT"

echo ""
echo "================================================================"
echo "  Resume complete. Trajectory at: ${TRAJ_OUTPUT}"
echo "================================================================"
