#!/bin/bash
# Generate high-quality contrastive training data for the EBM.
#
# Walks LeanDojo proof traces through Pantograph, generates LLM candidates
# at each step, and classifies results as Positive (ground-truth or
# alternative proof) or Negative (divergent path).
#
# Produces a Parquet file compatible with train-ebm's ContrastiveSampler.
#
# Usage:
#   SGLANG_URL=http://localhost:30000 ./scripts/generate_ebm_training_data.sh
#   NUM_THEOREMS=5000 MIN_STEPS=3 ./scripts/generate_ebm_training_data.sh
#
# Prerequisites:
#   - SGLang inference server running (./scripts/start_inference_server.sh)
#   - Pantograph built (./scripts/setup_pantograph.sh)
#   - cargo build --release -p prover-core
#   - Tactic pairs JSONL: data/tactic_pairs/train.jsonl (from prepare_data.sh)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$(dirname "$0")/_lib.sh"

SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
ensure_sglang "$SGLANG_URL"

# ── Configuration ─────────────────────────────────────────────────────
NUM_THEOREMS="${NUM_THEOREMS:-5000}"
MIN_STEPS="${MIN_STEPS:-3}"
CANDIDATES="${CANDIDATES:-8}"
TARGET_NEGATIVES="${TARGET_NEGATIVES:-15}"
TEMPERATURE="${TEMPERATURE:-1.0}"
CONCURRENCY="${CONCURRENCY:-6}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMPORTS="${IMPORTS:-Init,Mathlib}"
TACTIC_PAIRS="${TACTIC_PAIRS:-${REPO_ROOT}/data/tactic_pairs/train.jsonl}"

SEARCH_CONFIG="${REPO_ROOT}/configs/search.toml"
OUTPUT_DIR="${REPO_ROOT}/trajectories"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${OUTPUT:-${OUTPUT_DIR}/negatives_${TIMESTAMP}.parquet}"

PROVER="${REPO_ROOT}/target/release/prover-core"

# ── Validation ────────────────────────────────────────────────────────
if [ ! -f "$PROVER" ]; then
    echo "ERROR: prover-core binary not found at $PROVER"
    echo "Run: cargo build --release -p prover-core"
    exit 1
fi

if [ ! -f "$TACTIC_PAIRS" ]; then
    echo "ERROR: Tactic pairs file not found at $TACTIC_PAIRS"
    echo "Run: ./scripts/prepare_data.sh"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Print config ──────────────────────────────────────────────────────
echo "================================================================"
echo "  Generate EBM Training Data (Contrastive Negatives)"
echo "================================================================"
echo "  SGLang:           ${SGLANG_URL}"
echo "  Tactic pairs:     ${TACTIC_PAIRS}"
echo "  Output:           ${OUTPUT}"
echo "  Num theorems:     ${NUM_THEOREMS}"
echo "  Min steps:        ${MIN_STEPS}"
echo "  Candidates/step:  ${CANDIDATES}"
echo "  Target neg/thm:   ${TARGET_NEGATIVES}"
echo "  Temperature:      ${TEMPERATURE}"
echo "  Concurrency:      ${CONCURRENCY}"
echo "  Lean workers:     ${NUM_WORKERS}"
echo "  Imports:          ${IMPORTS}"
echo "================================================================"

# ── Run ───────────────────────────────────────────────────────────────
RUST_LOG="${RUST_LOG:-info,prover_core::pipeline=debug}" \
"$PROVER" generate-negatives \
    --config "$SEARCH_CONFIG" \
    --tactic-pairs "$TACTIC_PAIRS" \
    --server-url "$SGLANG_URL" \
    --output "$OUTPUT" \
    --num-theorems "$NUM_THEOREMS" \
    --min-steps "$MIN_STEPS" \
    --candidates-per-step "$CANDIDATES" \
    --target-negatives "$TARGET_NEGATIVES" \
    --temperature "$TEMPERATURE" \
    --concurrency "$CONCURRENCY" \
    --num-workers "$NUM_WORKERS" \
    --imports "$IMPORTS"

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Generation complete!"
echo "  Output: ${OUTPUT}"
echo ""
echo "  Next steps:"
echo "    1. Check survival rate in the summary above"
echo "    2. Train EBM:"
echo "       cargo run --release -p prover-core -- train-ebm \\"
echo "           --trajectories ${OUTPUT} \\"
echo "           --server-url ${SGLANG_URL} \\"
echo "           --output-dir checkpoints/ebm/negatives"
echo "================================================================"
