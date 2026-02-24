#!/bin/bash
# Launch BurnQED standalone encode server (HuggingFace transformers, no SGLang).
#
# Provides true batch encoding over HTTP, bypassing SGLang's broken batch
# return_hidden_states (Issue #8066). Use alongside the inference server
# (which handles generation on port 30000).
#
# Usage:
#   ./scripts/start_encode_server.sh [model_path]
#   ENCODE_PORT=30001 ./scripts/start_encode_server.sh models/llm/iter_3
#
# Prerequisites:
#   pip install transformers torch fastapi uvicorn

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${1:-${REPO_ROOT}/models/deepseek-prover-v2-7b}"
ENCODE_PORT="${ENCODE_PORT:-30001}"
ENCODE_DTYPE="${ENCODE_DTYPE:-bfloat16}"

# Pin to a specific GPU (e.g., CUDA_DEVICE=1). Unset = use all GPUs.
CUDA_DEVICE="${CUDA_DEVICE:-}"

# Max batch size for encoding. 16 is safe with bfloat16 on a dedicated 24GB GPU
# (~13GB model + ~4-5GB activations at batch 16). Lower to 4 if sharing a GPU.
ENCODE_MAX_BATCH="${ENCODE_MAX_BATCH:-16}"

if [ -n "$CUDA_DEVICE" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

echo "================================================================"
echo "  Starting BurnQED Encode Server"
echo "================================================================"
echo "  Model:      ${MODEL_PATH}"
echo "  Port:       ${ENCODE_PORT}"
echo "  Dtype:      ${ENCODE_DTYPE}"
echo "  GPU:        ${CUDA_DEVICE:-all}"
echo "  Max batch:  ${ENCODE_MAX_BATCH}"
echo "================================================================"

export ENCODE_PORT ENCODE_DTYPE

python "${REPO_ROOT}/python/encode_server.py" \
    --model-path "$MODEL_PATH" \
    --port "$ENCODE_PORT" \
    --dtype "$ENCODE_DTYPE" \
    --max-batch-size "$ENCODE_MAX_BATCH"
