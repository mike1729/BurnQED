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

# Auto-detect VRAM and adjust memory fraction.
# 7B fp16 model needs ~14GB. On 24GB GPUs (4090/3090) use 0.75 to leave
# headroom for hidden-state extraction. On 40GB+ GPUs use 0.85.
if [ -z "${MEM_FRACTION:-}" ]; then
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    if [ "$GPU_MEM_MB" -gt 0 ] && [ "$GPU_MEM_MB" -lt 30000 ]; then
        MEM_FRACTION="0.75"
    else
        MEM_FRACTION="0.85"
    fi
fi

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
