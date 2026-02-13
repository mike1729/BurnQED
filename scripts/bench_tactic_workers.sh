#!/usr/bin/env bash
# Benchmark tactic_workers impact on search performance.
# Requires: model weights, Lean/Pantograph built.
#
# Usage: ./scripts/bench_tactic_workers.sh [--theorems data/test_theorems.json]
#
# Environment variables:
#   MODEL_PATH  - path to HuggingFace model directory (required)
#   THEOREMS    - path to theorem index JSON (default: data/test_theorems.json)
#   NUM_WORKERS - number of Lean workers (default: 10)
#   CONFIG      - search config TOML (default: configs/search.toml)

set -euo pipefail

SGLANG_URL="${SGLANG_URL:?SGLANG_URL must be set (e.g. http://localhost:30000)}"
THEOREMS="${THEOREMS:-data/test_theorems.json}"
NUM_WORKERS="${NUM_WORKERS:-10}"
CONFIG="${CONFIG:-configs/search.toml}"

echo "=== Tactic Workers Benchmark ==="
echo "SGLang:     $SGLANG_URL"
echo "Theorems:   $THEOREMS"
echo "Workers:    $NUM_WORKERS"
echo "Config:     $CONFIG"
echo ""

for tw in 1 2 4 8; do
    echo "=== tactic_workers=$tw ==="
    TMPOUT=$(mktemp "/tmp/bench_tw${tw}_XXXXXX.parquet")

    # Create temp config with tactic_workers override
    TMPCFG=$(mktemp "/tmp/search_tw${tw}_XXXXXX.toml")
    cp "$CONFIG" "$TMPCFG"
    # Append tactic_workers to [search] section
    echo "tactic_workers = $tw" >> "$TMPCFG"

    START=$(date +%s%N)

    RUST_LOG=search=debug cargo run --release -p prover-core -- search \
        --config "$TMPCFG" \
        --server-url "$SGLANG_URL" \
        --theorems "$THEOREMS" \
        --output "$TMPOUT" \
        --num-workers "$NUM_WORKERS" \
        2>&1 | grep -E "(Parallel tactic|Proof found|Search exhausted|wall_time|burn-qed Search)"

    END=$(date +%s%N)
    ELAPSED=$(( (END - START) / 1000000 ))
    echo "Wall time: ${ELAPSED}ms"
    echo ""

    rm -f "$TMPOUT" "$TMPCFG"
done

echo "=== Benchmark complete ==="
