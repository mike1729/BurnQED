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
    _LIB_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    bash "${_LIB_REPO_ROOT}/scripts/start_inference_server.sh" "${model}" &
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

# Marker file tracking which model the inference server currently has loaded.
SERVER_MODEL_MARKER="/tmp/burnqed_server_model"

# Kill the running inference server, start a new one with the given model,
# and wait for it to become healthy.
#
# Usage:
#   restart_inference_server "$SGLANG_URL" "$MODEL_PATH"
restart_inference_server() {
    local url="${1:?Usage: restart_inference_server <url> <model_path>}"
    local model="${2:?Usage: restart_inference_server <url> <model_path>}"

    echo "Stopping inference server..."
    pkill -f "inference_server.py" 2>/dev/null || true
    sleep 3
    pkill -9 -f "inference_server.py" 2>/dev/null || true
    sleep 2

    rm -f "$SERVER_MODEL_MARKER"

    # Start server with new model and wait for health
    ensure_sglang "$url" "$model"

    echo "$model" > "$SERVER_MODEL_MARKER"
    echo "Server restarted with model: $model"
}

# Ensure the inference server is running with the correct model.
# If the server is healthy and already has the right model loaded, this is a no-op.
# Otherwise it restarts with the requested model.
#
# Usage:
#   ensure_server "$SGLANG_URL" "$MODEL_PATH"
ensure_server() {
    local url="${1:?Usage: ensure_server <url> <model_path>}"
    local model="${2:?Usage: ensure_server <url> <model_path>}"

    if curl -sf "${url}/health" > /dev/null 2>&1; then
        if [ -f "$SERVER_MODEL_MARKER" ] && [ "$(cat "$SERVER_MODEL_MARKER")" = "$model" ]; then
            echo "Inference server running with correct model: $model"
            return 0
        fi
        echo "Server running but wrong model (want: $model)"
    fi

    restart_inference_server "$url" "$model"
}
