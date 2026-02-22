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

# Stop the running inference server to free VRAM.
#
# Usage:
#   stop_inference_server
stop_inference_server() {
    echo "Stopping inference server..."
    pkill -f "inference_server.py" 2>/dev/null || true
    sleep 3
    pkill -9 -f "inference_server.py" 2>/dev/null || true
    sleep 2

    rm -f "$SERVER_MODEL_MARKER"
    echo "Inference server stopped"
}

# Kill the running inference server, start a new one with the given model,
# and wait for it to become healthy.
#
# Usage:
#   restart_inference_server "$SGLANG_URL" "$MODEL_PATH"
restart_inference_server() {
    local url="${1:?Usage: restart_inference_server <url> <model_path>}"
    local model="${2:?Usage: restart_inference_server <url> <model_path>}"

    stop_inference_server

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
        if [ -f "$SERVER_MODEL_MARKER" ]; then
            if [ "$(cat "$SERVER_MODEL_MARKER")" = "$model" ]; then
                echo "Inference server running with correct model: $model"
                return 0
            fi
            echo "Server running but wrong model (want: $model, have: $(cat "$SERVER_MODEL_MARKER"))"
        else
            # No marker — server was started manually, assume correct model
            echo "$model" > "$SERVER_MODEL_MARKER"
            echo "Inference server running (assuming correct model: $model)"
            return 0
        fi
    fi

    restart_inference_server "$url" "$model"
}

# Marker file tracking which model the encode server currently has loaded.
ENCODE_SERVER_MODEL_MARKER="/tmp/burnqed_encode_server_model"

# Ensure the encode server is reachable, starting it if needed.
#
# Usage:
#   ensure_encode_server "$ENCODE_URL" [model_path]
#
# If the server is not reachable, starts it via start_encode_server.sh
# and waits up to 5 minutes for it to become ready.
ensure_encode_server() {
    local url="${1:?Usage: ensure_encode_server <url> [model_path]}"
    local model="${2:-deepseek-prover-v2-7b}"

    if curl -sf "${url}/health" > /dev/null 2>&1; then
        echo "Encode server already running at ${url}"
        return 0
    fi

    echo "Encode server not reachable at ${url}, starting..."
    _LIB_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    bash "${_LIB_REPO_ROOT}/scripts/start_encode_server.sh" "${model}" &

    # Wait for server to become ready (up to 5 minutes)
    for i in $(seq 1 60); do
        if curl -sf "${url}/health" > /dev/null 2>&1; then
            echo "Encode server ready"
            echo "$model" > "$ENCODE_SERVER_MODEL_MARKER"
            return 0
        fi
        sleep 5
    done

    echo "ERROR: Encode server failed to start within 5 minutes"
    exit 1
}

# Stop the running encode server to free VRAM.
#
# Usage:
#   stop_encode_server
stop_encode_server() {
    echo "Stopping encode server..."
    pkill -f "encode_server.py" 2>/dev/null || true
    sleep 3
    pkill -9 -f "encode_server.py" 2>/dev/null || true
    sleep 2

    rm -f "$ENCODE_SERVER_MODEL_MARKER"
    echo "Encode server stopped"
}

# Run a command with output logged to a file, preserving TTY features
# (progress bars, colors). Uses `script` to allocate a PTY so programs
# that check isatty() still produce full output.
#
# Usage: run_logged LOGFILE COMMAND [ARGS...]
run_logged() {
    local logfile="$1"
    shift
    script -qefc "$*" "$logfile"
}
