#!/bin/bash
# Launch SGLang inference server for theorem proving models.
#
# Usage:
#   ./scripts/start_sglang.sh                  # default: deepseek
#   ./scripts/start_sglang.sh deepseek
#   ./scripts/start_sglang.sh goedel
#   ./scripts/start_sglang.sh /path/to/model
#   CUDA_DEVICE=0 MEM_FRACTION=0.90 ./scripts/start_sglang.sh goedel
#
# Model shortcuts:
#   deepseek  →  data/models/base/deepseek-prover-v2-7b  (fp8 quantization)
#   goedel    →  data/models/base/goedel-prover-v2-8b    (bfloat16, no quantization)
#
# Environment overrides:
#   MODEL_PATH, PORT, TP, MEM_FRACTION, MAX_RUNNING, QUANTIZATION,
#   DTYPE, CUDA_DEVICE, EXTRA_ARGS
#
# Prerequisites:
#   pip install "sglang[all]"

set -euo pipefail

# Ensure JIT compilation uses CUDA 12.8+ (required for Blackwell compute_120a)
if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME="/usr/local/cuda-12.8"
    export PATH="/usr/local/cuda-12.8/bin:$PATH"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Resolve model name / path ───────────────────────────────────────────────
MODEL_ARG="${1:-${MODEL_PATH:-deepseek}}"

# Map shorthand names → full paths + per-model defaults.
# QUANTIZATION and DTYPE use bash ${VAR-default} (unset → default, empty → keep empty).
case "$MODEL_ARG" in
    deepseek|deepseek-prover|deepseek-prover-v2-7b)
        MODEL_PATH="${REPO_ROOT}/data/models/base/deepseek-prover-v2-7b"
        # fp8 gives ~25% faster decode with no measurable quality loss
        QUANTIZATION="${QUANTIZATION-fp8}"
        DTYPE="${DTYPE-auto}"
        ;;
    goedel|goedel-v2|goedel-prover-v2-8b)
        MODEL_PATH="${REPO_ROOT}/data/models/base/goedel-prover-v2-8b"
        # Qwen3-8B: no fp8 calibration available; run in native bfloat16
        QUANTIZATION="${QUANTIZATION-}"
        DTYPE="${DTYPE-bfloat16}"
        ;;
    *)
        # Treat as a literal path
        MODEL_PATH="$MODEL_ARG"
        QUANTIZATION="${QUANTIZATION-fp8}"
        DTYPE="${DTYPE-auto}"
        ;;
esac

# ── Configuration ────────────────────────────────────────────────────────────
PORT="${PORT:-30000}"
TP="${TP:-1}"

# Memory fraction: 0.90 is safe when the GPU is dedicated to inference.
# Lower to 0.65 if coexisting with the encode server on the same GPU.
MEM_FRACTION="${MEM_FRACTION:-0.80}"

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
echo "  Quantize:   ${QUANTIZATION:-none}"
echo "  Dtype:      ${DTYPE}"
echo "  GPU:        ${CUDA_DEVICE:-all}"
echo "================================================================"

EXTRA_ARGS="${EXTRA_ARGS:-}"

QUANT_FLAG=""
if [ -n "$QUANTIZATION" ]; then
    QUANT_FLAG="--quantization $QUANTIZATION"
fi

DTYPE_FLAG=""
if [ -n "$DTYPE" ] && [ "$DTYPE" != "auto" ]; then
    DTYPE_FLAG="--dtype $DTYPE"
fi

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --tp "$TP" \
    --trust-remote-code \
    --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "$MAX_RUNNING" \
    $QUANT_FLAG \
    $DTYPE_FLAG \
    $EXTRA_ARGS
