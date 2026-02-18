#!/bin/bash
# Shared helpers for BurnQED scripts.
# Source this file: source "$(dirname "$0")/_lib.sh"

# Auto-detect CUDA for burn-rs training backend.
# Sets CARGO_FEATURES to "--features cuda" if nvidia-smi is available.
if command -v nvidia-smi &> /dev/null; then
    CARGO_FEATURES="--features cuda"
else
    CARGO_FEATURES=""
fi

# Ensure the inference server is reachable, starting it if needed.
#
# Usage:
#   ensure_sglang "$SGLANG_URL" [model_path]
#
# If the server is not reachable, starts it via start_inference_server.sh
# and waits up to 5 minutes for it to become ready.
ensure_sglang() {
    local url="${1:?Usage: ensure_sglang <url> [model_path]}"
    local model="${2:-deepseek-prover-v2-7b}"

    if curl -sf "${url}/health" > /dev/null 2>&1; then
        echo "Inference server already running at ${url}"
        return 0
    fi

    echo "Inference server not reachable at ${url}, starting..."
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    "${REPO_ROOT}/scripts/start_inference_server.sh" "${model}" &
    SGLANG_PID=$!

    # Clean up server on script exit
    trap "kill $SGLANG_PID 2>/dev/null" EXIT

    # Wait for server to become ready (up to 5 minutes)
    for i in $(seq 1 60); do
        if curl -sf "${url}/health" > /dev/null 2>&1; then
            echo "Inference server ready"
            return 0
        fi
        sleep 5
    done

    echo "ERROR: Inference server failed to start within 5 minutes"
    exit 1
}
