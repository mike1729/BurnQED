#!/bin/bash
# Transfer experiment data to a RunPod instance.
#
# Run this FROM the source machine (e.g., Lambda) to push data to RunPod.
# Uses rsync over SSH with support for custom ports.
#
# Usage:
#   # From Lambda, push to RunPod:
#   bash scripts/migrate_to_runpod.sh root@77.237.148.176 --port 32188
#
#   # With custom paths:
#   SRC_DIR=~/BurnQED DEST_DIR=/workspace/BurnQED \
#     bash scripts/migrate_to_runpod.sh root@runpod-ip --port 32188
#
#   # With custom SSH key:
#   bash scripts/migrate_to_runpod.sh root@runpod-ip --port 32188 --key ~/.ssh/id_ed25519
#
#   # Dry run (show what would be transferred):
#   DRY_RUN=1 bash scripts/migrate_to_runpod.sh root@runpod-ip --port 32188
#
# Prerequisites:
#   - SSH access to the RunPod instance (key in ssh-agent or specified via --key)
#   - rsync installed on both machines

set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────────
RUNPOD_HOST=""
SSH_PORT="22"
SSH_KEY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) SSH_PORT="$2"; shift 2 ;;
        --key)  SSH_KEY="$2"; shift 2 ;;
        *)      RUNPOD_HOST="$1"; shift ;;
    esac
done

if [ -z "$RUNPOD_HOST" ]; then
    echo "Usage: bash scripts/migrate_to_runpod.sh <runpod_host> [--port PORT] [--key SSH_KEY]"
    echo ""
    echo "Example:"
    echo "  bash scripts/migrate_to_runpod.sh root@77.237.148.176 --port 32188"
    exit 1
fi

SRC_DIR="${SRC_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DEST_DIR="${DEST_DIR:-/workspace/BurnQED}"
DRY_RUN="${DRY_RUN:-0}"

# Build SSH command with port and optional key
SSH_CMD="ssh -p ${SSH_PORT}"
if [ -n "$SSH_KEY" ]; then
    SSH_CMD="$SSH_CMD -i ${SSH_KEY}"
fi

RSYNC_OPTS="-rltz --progress --compress --no-owner --no-group -e '${SSH_CMD}'"
if [ "$DRY_RUN" -eq 1 ]; then
    RSYNC_OPTS="$RSYNC_OPTS --dry-run"
    echo "(DRY RUN — no files will be transferred)"
    echo ""
fi

echo "================================================================"
echo "  BurnQED Migration → RunPod"
echo "================================================================"
echo "  Source:      ${SRC_DIR}"
echo "  Destination: ${RUNPOD_HOST}:${DEST_DIR}"
echo "  SSH port:    ${SSH_PORT}"
echo "================================================================"

# Ensure destination directories exist
echo ""
echo "=== Creating destination directories ==="
$SSH_CMD "$RUNPOD_HOST" "mkdir -p ${DEST_DIR}/{models,checkpoints/llm,trajectories,eval_results,data,logs,baselines,configs}"

transfer() {
    local label="$1"
    local dir="$2"
    local src="${SRC_DIR}/${dir}/"
    local dst="${RUNPOD_HOST}:${DEST_DIR}/${dir}/"

    if [ ! -d "$src" ]; then
        echo "  (skipped ${dir}/ — not found)"
        return
    fi

    echo ""
    echo "=== Transferring ${label} ==="
    # shellcheck disable=SC2086
    eval rsync $RSYNC_OPTS "$src" "$dst"
}

transfer "model weights"            models
transfer "LoRA checkpoints"        checkpoints/llm
transfer "trajectories"             trajectories
transfer "eval results"             eval_results
transfer "training data"            data
transfer "baselines"                baselines
transfer "logs"                     logs
transfer "configs"                  configs

# NOTE: checkpoints/ebm/ is deliberately excluded.
# The EBM architecture changed (3-layer → 4-layer MLP) and old .mpk files
# are incompatible. Retrain EBM on RunPod from trajectory data:
#   cargo run -p prover-core -- train-ebm \
#     --trajectories trajectories/*.parquet --server-url http://localhost:30000

echo ""
echo "================================================================"
echo "  Migration complete!"
echo ""
echo "  Transferred:"
echo "    - models/            (base + merged safetensors)"
echo "    - checkpoints/llm/   (LoRA adapters)"
echo "    - trajectories/      (Parquet files)"
echo "    - eval_results/      (JSON eval results)"
echo "    - data/              (theorem index, tactic pairs)"
echo "    - baselines/         (raw model baselines)"
echo "    - logs/              (experiment logs)"
echo "    - configs/           (search.toml, models.toml)"
echo ""
echo "  NOT transferred (retrain on RunPod):"
echo "    - checkpoints/ebm/   (incompatible — 4-layer architecture change)"
echo ""
echo "  Next steps on RunPod:"
echo "    1. git clone + bash scripts/setup_runpod.sh"
echo "    2. Start server: ./scripts/start_inference_server.sh models/llm/iter_N"
echo "    3. Retrain EBM: cargo run -p prover-core -- train-ebm ..."
echo "    4. Continue experiment: ./scripts/run_iteration_search.sh N"
echo "================================================================"
