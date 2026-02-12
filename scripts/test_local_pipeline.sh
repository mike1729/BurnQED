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

# Auto-detect CUDA
CUDA_FEATURES=$(command -v nvidia-smi &>/dev/null && echo "--features cuda" || echo "")

# Build release
echo "=== Building prover-core (release) ==="
cargo build --release -p prover-core $CUDA_FEATURES
echo ""

PROVER="cargo run --release -p prover-core ${CUDA_FEATURES} --"

# Search
echo "=== Running search (3 theorems, budget=20) ==="
# shellcheck disable=SC2086
$PROVER search \
    --config "$CONFIG" \
    --model-path "$MODEL_PATH" \
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
