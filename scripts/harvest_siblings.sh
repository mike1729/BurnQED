#!/bin/bash
# Harvest sibling states from proved theorems across existing trajectory files.
#
# Two-phase pipeline:
#   1. export-proof-paths: extract proof tactic sequences from Parquet → JSONL
#   2. generate-negatives: replay proofs in Pantograph, generate LLM candidates
#      + probe tactics at each step, classify as positive/negative
#
# The output Parquet is ready for EBM training (merge with existing trajectories).
#
# Usage:
#   SGLANG_URL=http://localhost:30000 ./scripts/harvest_siblings.sh
#   TRAJECTORIES="trajectories/iter_0.parquet trajectories/iter_0_noisy.parquet" ./scripts/harvest_siblings.sh
#   MIN_STEPS=3 CANDIDATES=8 ./scripts/harvest_siblings.sh
#
# Prerequisites:
#   - SGLang inference server running (./scripts/start_sglang.sh)
#   - Pantograph built (./scripts/setup_pantograph.sh)
#   - cargo build --release -p prover-core
#   - At least one trajectory Parquet file with proved theorems

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$(dirname "$0")/_lib.sh"

SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
ensure_sglang "$SGLANG_URL"

# ── Configuration ─────────────────────────────────────────────────────
# Space-separated list of trajectory Parquet files to harvest from.
# Default: all existing trajectory files.
if [ -z "${TRAJECTORIES:-}" ]; then
    TRAJECTORIES=""
    for f in "${REPO_ROOT}"/trajectories/*.parquet; do
        [ -f "$f" ] && TRAJECTORIES="${TRAJECTORIES} ${f}"
    done
    TRAJECTORIES="${TRAJECTORIES# }"  # trim leading space
fi

if [ -z "$TRAJECTORIES" ]; then
    echo "ERROR: No trajectory Parquet files found in trajectories/"
    echo "Set TRAJECTORIES env var to specify files manually."
    exit 1
fi

MIN_STEPS="${MIN_STEPS:-2}"
CANDIDATES="${CANDIDATES:-8}"
TARGET_NEGATIVES="${TARGET_NEGATIVES:-15}"
TEMPERATURE="${TEMPERATURE:-1.0}"
CONCURRENCY="${CONCURRENCY:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMPORTS="${IMPORTS:-Init,Mathlib}"

SEARCH_CONFIG="${REPO_ROOT}/configs/search.toml"
OUTPUT_DIR="${REPO_ROOT}/trajectories"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PROOF_PATHS="${OUTPUT_DIR}/proof_paths_${TIMESTAMP}.jsonl"
OUTPUT="${OUTPUT:-${OUTPUT_DIR}/siblings_${TIMESTAMP}.parquet}"

PROVER="${REPO_ROOT}/target/release/prover-core"

# ── Validation ────────────────────────────────────────────────────────
if [ ! -f "$PROVER" ]; then
    echo "Building prover-core (release)..."
    cargo build --release -p prover-core $CARGO_FEATURES
fi

mkdir -p "$OUTPUT_DIR"

# Count input files
FILE_COUNT=0
for f in $TRAJECTORIES; do
    [ -f "$f" ] && FILE_COUNT=$((FILE_COUNT + 1))
done

echo "================================================================"
echo "  Harvest Siblings — Post-Hoc Sibling Expansion"
echo "================================================================"
echo "  SGLang:           ${SGLANG_URL}"
echo "  Input files:      ${FILE_COUNT}"
for f in $TRAJECTORIES; do
    echo "    - $(basename "$f")"
done
echo "  Min steps:        ${MIN_STEPS}"
echo "  Candidates/step:  ${CANDIDATES}"
echo "  Target neg/thm:   ${TARGET_NEGATIVES}"
echo "  Temperature:      ${TEMPERATURE}"
echo "  Concurrency:      ${CONCURRENCY}"
echo "  Lean workers:     ${NUM_WORKERS}"
echo "  Imports:          ${IMPORTS}"
echo "  Proof paths:      ${PROOF_PATHS}"
echo "  Output:           ${OUTPUT}"
echo "================================================================"

# ── Phase 1: Export proof paths ──────────────────────────────────────
echo ""
echo "=== Phase 1: Exporting proof paths from trajectory files ==="

# Build --trajectories flags
TRAJ_FLAGS=""
for f in $TRAJECTORIES; do
    TRAJ_FLAGS="${TRAJ_FLAGS} ${f}"
done

"$PROVER" export-proof-paths \
    --trajectories $TRAJ_FLAGS \
    --output "$PROOF_PATHS" \
    --min-steps "$MIN_STEPS"

STEP_COUNT=$(wc -l < "$PROOF_PATHS")
if [ "$STEP_COUNT" -eq 0 ]; then
    echo "No proof paths found with min_steps=${MIN_STEPS}. Nothing to harvest."
    rm -f "$PROOF_PATHS"
    exit 0
fi

echo "Exported ${STEP_COUNT} tactic steps to ${PROOF_PATHS}"

# ── Phase 2: Generate negatives (sibling harvest) ────────────────────
echo ""
echo "=== Phase 2: Generating sibling states via Pantograph ==="

RUST_LOG="${RUST_LOG:-info,prover_core::pipeline=debug}" \
"$PROVER" generate-negatives \
    --config "$SEARCH_CONFIG" \
    --tactic-pairs "$PROOF_PATHS" \
    --server-url "$SGLANG_URL" \
    --output "$OUTPUT" \
    --candidates-per-step "$CANDIDATES" \
    --target-negatives "$TARGET_NEGATIVES" \
    --temperature "$TEMPERATURE" \
    --concurrency "$CONCURRENCY" \
    --num-workers "$NUM_WORKERS" \
    --imports "$IMPORTS"

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Harvest complete!"
echo "  Proof paths:  ${PROOF_PATHS}"
echo "  Output:       ${OUTPUT}"
echo ""
echo "  Next steps:"
echo "    1. Merge with existing training data:"
echo "       cargo run --release -p prover-core -- train-ebm \\"
echo "           --trajectories trajectories/iter_0_noisy.parquet ${OUTPUT} \\"
echo "           --server-url ${SGLANG_URL} \\"
echo "           --output-dir checkpoints/ebm/with_siblings"
echo ""
echo "    2. Or run search with new EBM:"
echo "       cargo run --release -p prover-core -- search \\"
echo "           --ebm-path checkpoints/ebm/with_siblings \\"
echo "           --server-url ${SGLANG_URL} \\"
echo "           --theorems data/theorem_index.json \\"
echo "           --output trajectories/iter_with_siblings.parquet"
echo "================================================================"
