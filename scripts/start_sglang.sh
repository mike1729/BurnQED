#!/bin/bash
# Launch SGLang inference server for DeepSeek-Prover-V2-7B.
#
# Supports both tactic generation and hidden-state extraction for EBM.
#
# Usage:
#   ./scripts/start_sglang.sh [model_path]
#   PORT=30000 TP=2 ./scripts/start_sglang.sh
#
# Prerequisites:
#   pip install "sglang[all]"

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
echo "  Starting SGLang Server"
echo "================================================================"
echo "  Model:      ${MODEL_PATH}"
echo "  Port:       ${PORT}"
echo "  TP:         ${TP}"
echo "  Mem frac:   ${MEM_FRACTION}"
echo "================================================================"

EXTRA_ARGS="${EXTRA_ARGS:-}"

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --tp "$TP" \
    --trust-remote-code \
    --mem-fraction-static "$MEM_FRACTION" \
    $EXTRA_ARGS
