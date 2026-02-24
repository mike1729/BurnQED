#!/bin/bash
# Launch SGLang inference server for DeepSeek-Prover-V2-7B.
#
# Usage:
#   ./scripts/start_sglang.sh [model_path]
#   CUDA_DEVICE=0 MEM_FRACTION=0.90 ./scripts/start_sglang.sh
#
# Prerequisites:
#   pip install "sglang[all]"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${1:-${REPO_ROOT}/models/deepseek-prover-v2-7b}"
PORT="${PORT:-30000}"
TP="${TP:-1}"

# Memory fraction: 0.90 is safe when the GPU is dedicated to inference.
# Lower to 0.65 if coexisting with the encode server on the same GPU.
MEM_FRACTION="${MEM_FRACTION:-0.90}"

# Max concurrent running requests (default 256). Higher values increase
# throughput but use more KV-cache memory.
MAX_RUNNING="${MAX_RUNNING:-256}"

# Pin to a specific GPU (e.g., CUDA_DEVICE=0). Unset = use all GPUs.
CUDA_DEVICE="${CUDA_DEVICE:-}"

if [ -n "$CUDA_DEVICE" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

echo "================================================================"
echo "  Starting SGLang Server"
echo "================================================================"
echo "  Model:      ${MODEL_PATH}"
echo "  Port:       ${PORT}"
echo "  TP:         ${TP}"
echo "  Mem frac:   ${MEM_FRACTION}"
echo "  Max running:${MAX_RUNNING}"
echo "  GPU:        ${CUDA_DEVICE:-all}"
echo "================================================================"

EXTRA_ARGS="${EXTRA_ARGS:-}"

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --tp "$TP" \
    --trust-remote-code \
    --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "$MAX_RUNNING" \
    $EXTRA_ARGS
