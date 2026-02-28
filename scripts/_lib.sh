#!/bin/bash
# Shared helpers for BurnQED scripts.
# Source this file: source "$(dirname "$0")/_lib.sh"

# ── Canonical path variables ──────────────────────────────────────────────
# All generated/downloaded artifacts live under data/.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/data"
LEAN_DIR="${DATA_ROOT}/lean"
BENCH_DIR="${DATA_ROOT}/benchmarks"
SFT_DIR="${DATA_ROOT}/sft"
CKPT_DIR="${DATA_ROOT}/checkpoints"
LORA_DIR="${CKPT_DIR}/lora"
EBM_CKPT_DIR="${CKPT_DIR}/ebm"
MODEL_DIR="${DATA_ROOT}/models"
BASE_MODEL_DIR="${MODEL_DIR}/base"
MERGED_MODEL_DIR="${MODEL_DIR}/merged"
TRAJ_DIR="${DATA_ROOT}/trajectories"
EMBED_DIR="${DATA_ROOT}/embeddings"
EVAL_DIR="${DATA_ROOT}/evals"
LOG_DIR="${DATA_ROOT}/logs"

# Ensure cargo (Rust toolchain) is on PATH.
if [ -d "$HOME/.cargo/bin" ] && [[ ":$PATH:" != *":$HOME/.cargo/bin:"* ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Ensure elan (Lean toolchain manager) is on PATH.
if [ -d "$HOME/.elan/bin" ] && [[ ":$PATH:" != *":$HOME/.elan/bin:"* ]]; then
    export PATH="$HOME/.elan/bin:$PATH"
fi

# Auto-detect CUDA for burn-rs training backend.
# Sets CARGO_FEATURES to "--features cuda" if nvidia-smi is available.
if command -v nvidia-smi &> /dev/null; then
    CARGO_FEATURES="--features cuda"
else
    CARGO_FEATURES=""
fi

# Ensure the SGLang inference server is reachable, starting it if needed.
#
# Usage:
#   ensure_sglang "$SGLANG_URL" [model_path]
#
# If the server is not reachable, starts it via start_sglang.sh
# and waits up to 5 minutes for it to become ready.
ensure_sglang() {
    local url="${1:?Usage: ensure_sglang <url> [model_path]}"
    local model="${2:-deepseek-prover-v2-7b}"

    if curl -sf "${url}/health" > /dev/null 2>&1; then
        echo "SGLang server already running at ${url}"
        return 0
    fi

    echo "SGLang server not reachable at ${url}, starting..."
    _LIB_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    bash "${_LIB_REPO_ROOT}/scripts/start_sglang.sh" "${model}" &

    # Wait for server to become ready (up to 5 minutes)
    for i in $(seq 1 60); do
        if curl -sf "${url}/health" > /dev/null 2>&1; then
            echo "SGLang server ready"
            return 0
        fi
        sleep 5
    done

    echo "ERROR: SGLang server failed to start within 5 minutes"
    exit 1
}

# Marker file tracking which model the inference server currently has loaded.
SERVER_MODEL_MARKER="/tmp/burnqed_server_model"

# Stop the running SGLang server to free VRAM.
#
# Usage:
#   stop_sglang_server
stop_sglang_server() {
    echo "Stopping SGLang server..."
    pkill -f "sglang.launch_server" 2>/dev/null || true
    sleep 3
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 2

    rm -f "$SERVER_MODEL_MARKER"
    echo "SGLang server stopped"
}

# Kill the running SGLang server, start a new one with the given model,
# and wait for it to become healthy.
#
# Usage:
#   restart_sglang_server "$SGLANG_URL" "$MODEL_PATH"
restart_sglang_server() {
    local url="${1:?Usage: restart_sglang_server <url> <model_path>}"
    local model="${2:?Usage: restart_sglang_server <url> <model_path>}"

    stop_sglang_server

    # Start server with new model and wait for health
    ensure_sglang "$url" "$model"

    echo "$model" > "$SERVER_MODEL_MARKER"
    echo "Server restarted with model: $model"
}

# Ensure the SGLang server is running with the correct model.
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
                echo "SGLang server running with correct model: $model"
                return 0
            fi
            echo "Server running but wrong model (want: $model, have: $(cat "$SERVER_MODEL_MARKER"))"
        else
            # No marker — server was started manually, assume correct model
            echo "$model" > "$SERVER_MODEL_MARKER"
            echo "SGLang server running (assuming correct model: $model)"
            return 0
        fi
    fi

    restart_sglang_server "$url" "$model"
}

# Backward-compatible aliases for scripts that use the old names
stop_inference_server() { stop_sglang_server; }
restart_inference_server() { restart_sglang_server "$@"; }

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
