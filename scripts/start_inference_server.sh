#!/bin/bash
# Launch BurnQED custom inference server wrapping sgl.Engine.
#
# Replaces start_sglang.sh — same env vars, same model auto-detection,
# but uses python/inference_server.py with in-process mean-pooling
# for ~7-10x faster encoding.
#
# Usage:
#   ./scripts/start_inference_server.sh [model_path]
#   PORT=30000 TP=2 ./scripts/start_inference_server.sh
#
# Prerequisites:
#   pip install "sglang[all]" fastapi uvicorn nest-asyncio

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${1:-${REPO_ROOT}/models/deepseek-prover-v2-7b}"
PORT="${PORT:-30000}"
TP="${TP:-1}"

# Memory fraction: 0.65 leaves ~11GB free on 32GB GPU for the nf4 encode
# server (~7GB) which must coexist during EBM eval/encoding.
MEM_FRACTION="${MEM_FRACTION:-0.65}"

echo "================================================================"
echo "  Starting BurnQED Inference Server"
echo "================================================================"
echo "  Model:      ${MODEL_PATH}"
echo "  Port:       ${PORT}"
echo "  TP:         ${TP}"
echo "  Mem frac:   ${MEM_FRACTION}"
echo "================================================================"

export PORT TP MEM_FRACTION

python "${REPO_ROOT}/python/inference_server.py" \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --tp "$TP" \
    --mem-fraction "$MEM_FRACTION"
