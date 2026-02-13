#!/bin/bash
# Local test: run release binary search on 3 easy theorems with TinyLlama.
#
# Uses a low budget config (20 nodes, depth 10) for fast iteration.
# TinyLlama won't prove anything, but validates the full pipeline runs
# without panics: model load → Lean pool → search loop → Parquet write.
#
# Prerequisites:
#   - models/tinyllama-1.1b/ downloaded
#   - Pantograph built (./scripts/setup_pantograph.sh)
#
# Usage (from repo root):
#   bash scripts/test_local_pipeline.sh
#   bash scripts/test_local_pipeline.sh --dry-run
#   NUM_WORKERS=2 bash scripts/test_local_pipeline.sh

set -euo pipefail

export RUST_LOG=info
export RUST_BACKTRACE=1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${REPO_ROOT}/models/tinyllama-1.1b"
THEOREMS="${REPO_ROOT}/data/test_quick.json"
CONFIG="${REPO_ROOT}/configs/search_local.toml"
NUM_WORKERS="${NUM_WORKERS:-2}"
WORK_DIR="${REPO_ROOT}/local_test_output"
OUTPUT="${WORK_DIR}/tinyllama_search.parquet"
EXTRA_ARGS="${*}"

echo "================================================================"
echo "  BurnQED Local Pipeline Test (release build)"
echo "================================================================"
echo "  Model:      ${MODEL_PATH}"
echo "  Theorems:   ${THEOREMS} (3 theorems)"
echo "  Config:     ${CONFIG} (budget=20, depth=10)"
echo "  Workers:    ${NUM_WORKERS}"
echo "  Output:     ${OUTPUT}"
echo "================================================================"
echo ""

# Verify prerequisites
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: ${MODEL_PATH} not found"
    echo "Download with:"
    echo "  huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${THEOREMS}" ]; then
    echo "ERROR: ${THEOREMS} not found"
    exit 1
fi

if [ ! -d "${REPO_ROOT}/vendor/Pantograph/.lake" ]; then
    echo "ERROR: Pantograph not built. Run: ./scripts/setup_pantograph.sh"
    exit 1
fi

mkdir -p "$WORK_DIR"

# Build release
echo "=== Building prover-core (release) ==="
cargo build --release -p prover-core
echo ""

PROVER="cargo run --release -p prover-core --"

# Search
echo "=== Running search (3 theorems, budget=20) ==="
echo "NOTE: This script requires an SGLang server running."
echo "Start with: SGLANG_URL=http://localhost:30000 ./scripts/start_sglang.sh $MODEL_PATH"
echo ""
SGLANG_URL="${SGLANG_URL:?SGLANG_URL must be set (e.g. http://localhost:30000)}"

# shellcheck disable=SC2086
$PROVER search \
    --config "$CONFIG" \
    --server-url "$SGLANG_URL" \
    --theorems "$THEOREMS" \
    --output "$OUTPUT" \
    --num-workers "$NUM_WORKERS" \
    $EXTRA_ARGS

# Summary
echo ""
echo "=== Trajectory Summary ==="
$PROVER summary --input "$OUTPUT"

echo ""
echo "================================================================"
echo "  Done. Output at: ${OUTPUT}"
echo "================================================================"
