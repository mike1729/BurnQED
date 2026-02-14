#!/bin/bash
# Run one complete expert iteration (train + search + eval).
#
# This is a convenience wrapper that calls:
#   1. run_iteration_train.sh  (fine-tune + export)
#   2. run_iteration_search.sh (EBM + search + eval + summary)
#
# For manual control (e.g., restarting SGLang between steps), run them separately:
#   ./scripts/run_iteration_train.sh 0
#   # restart SGLang with new model
#   ./scripts/run_iteration_search.sh 0
#
# Usage:
#   ./scripts/run_iteration.sh <iteration_number>
#   ./scripts/run_iteration.sh 0   # First iteration (base model fine-tune)
#   ./scripts/run_iteration.sh 1   # Second iteration (adds trajectory data + EBM)
#   NUM_WORKERS=30 ./scripts/run_iteration.sh 1
#   CONCURRENCY=16 ./scripts/run_iteration.sh 1

set -euo pipefail

ITER=${1:?"Usage: ./scripts/run_iteration.sh <iteration_number>"}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"${SCRIPT_DIR}/run_iteration_train.sh" "$ITER"
"${SCRIPT_DIR}/run_iteration_search.sh" "$ITER"
