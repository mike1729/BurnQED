#!/bin/bash
# Launch SGLang inference server for DeepSeek-Prover-V2-7B.
#
# The server provides ~50-100x faster inference than in-process candle,
# supporting both tactic generation and hidden-state extraction for EBM.
#
# Usage:
#   ./scripts/start_sglang.sh [model_path]
#   PORT=30000 TP=2 ./scripts/start_sglang.sh
#
# Prerequisites:
#   pip install "sglang[all]"

set -euo pipefail

MODEL_PATH="${1:-deepseek-ai/DeepSeek-Prover-V2-7B}"
PORT="${PORT:-30000}"
TP="${TP:-1}"

echo "================================================================"
echo "  Starting SGLang Server"
echo "================================================================"
echo "  Model:  ${MODEL_PATH}"
echo "  Port:   ${PORT}"
echo "  TP:     ${TP}"
echo "================================================================"

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --tp "$TP" \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --enable-return-hidden-states
