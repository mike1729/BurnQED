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

# Memory fraction: 0.65 leaves ~11GB free on 32GB GPU for the nf4 encode
# server (~7GB) which must coexist during EBM eval/encoding.
MEM_FRACTION="${MEM_FRACTION:-0.65}"

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
    --enable-return-hidden-states \
    $EXTRA_ARGS
