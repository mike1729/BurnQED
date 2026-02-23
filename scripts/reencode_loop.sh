#!/bin/bash
# Re-encode embeddings via inference server, restarting on zero-norm errors.
#
# Usage: ./scripts/reencode_loop.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-${REPO_ROOT}/models/llm/iter_4}"
INPUT="${INPUT:-${REPO_ROOT}/checkpoints/ebm/iter_4/embeddings.parquet}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/checkpoints/ebm/iter_4/embeddings_sglang.parquet}"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SERVER_URL="http://localhost:${PORT}"
MAX_RESTARTS="${MAX_RESTARTS:-50}"

restart_count=0

start_server() {
    echo "Starting inference server (model=${MODEL}, port=${PORT})..."
    pkill -f "inference_server.py.*--port ${PORT}" 2>/dev/null
    sleep 3
    nohup python3 "${REPO_ROOT}/python/inference_server.py" \
        --model-path "$MODEL" --port "$PORT" --mem-fraction "$MEM_FRACTION" \
        > /tmp/inference_server.log 2>&1 &
    # Wait for health
    for i in $(seq 1 60); do
        curl -sf "${SERVER_URL}/health" > /dev/null 2>&1 && break
        sleep 2
    done
    curl -sf "${SERVER_URL}/health" > /dev/null 2>&1 || {
        echo "ERROR: Server failed to start"; exit 1;
    }
    echo "Server ready."
}

# Initial start
start_server

while [ "$restart_count" -lt "$MAX_RESTARTS" ]; do
    echo ""
    echo "=== Encoding run #$((restart_count + 1)) ==="
    python3 "${REPO_ROOT}/scripts/reencode_via_server.py" \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --server-url "$SERVER_URL" \
        --batch-size "$BATCH_SIZE" \
        --save-every 2000
    rc=$?

    if [ "$rc" -eq 0 ]; then
        echo ""
        echo "All states encoded successfully!"
        break
    fi

    restart_count=$((restart_count + 1))
    echo ""
    echo "Restarting server (restart #${restart_count})..."
    start_server
done

if [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
    echo "ERROR: Hit max restarts ($MAX_RESTARTS). Check server stability."
    exit 1
fi

echo "Output: $OUTPUT"
