#!/bin/bash
# Benchmark LeanDojo tracing at different scales to determine parallelism.
#
# Runs trace_goedel_local.sh with increasing N and measures:
#   - Wall clock time (total, and per-proof)
#   - Peak RSS memory (via GNU time)
#   - Tactic pairs extracted per proof
#   - Cache/tmp disk usage
#
# The LeanDojo cache is NOT cleared between runs — Mathlib download (~2 min)
# is constant overhead we don't need to re-measure.
#
# Usage:
#   scripts/trace_benchmark.sh                    # default sizes: 5 10 20 50
#   scripts/trace_benchmark.sh 8 16 32            # custom sizes
#   WORKERS=16 scripts/trace_benchmark.sh         # override worker count

set -euo pipefail

source "$(dirname "$0")/_lib.sh"

WORKERS="${WORKERS:-32}"
LOCAL_BASE="${LOCAL_BASE:-/root/goedel_trace}"
BENCH_LOG="${LOG_DIR}/trace_benchmark.txt"
TRACED_DIR="${DATA_ROOT}/traced"

mkdir -p "$LOG_DIR"

if [[ $# -gt 0 ]]; then
    SIZES=("$@")
else
    SIZES=(5 10 20 50)
fi

echo "=== LeanDojo Trace Benchmark ==="
echo "  Workers: ${WORKERS}"
echo "  Sizes:   ${SIZES[*]}"
echo "  RAM:     $(free -g | awk '/Mem:/{print $2}') GB total, $(free -g | awk '/Mem:/{print $7}') GB available"
echo ""

RESULTS=()
HEADER=$(printf "%-6s  %10s  %10s  %8s  %8s  %8s  %10s" \
    "N" "Wall (s)" "Per-proof" "Pairs" "P/proof" "Peak MB" "Disk MB")
RESULTS+=("$HEADER")
RESULTS+=("$(printf '%0.s-' {1..76})")

for N in "${SIZES[@]}"; do
    echo "────────────────────────────────────────────────"
    echo "  Benchmarking N=${N} proofs"
    echo "────────────────────────────────────────────────"

    rm -f "${TRACED_DIR}/goedel_427_test_pairs.jsonl"

    # Clean only tmp between runs (keep cache for Mathlib reuse)
    rm -rf "${LOCAL_BASE}/tmp"
    mkdir -p "${LOCAL_BASE}/tmp"

    TRACE_LOG="${LOG_DIR}/trace_bench_n${N}.log"
    TIME_LOG="${LOG_DIR}/trace_bench_n${N}_time.log"

    WALL_START=$(date +%s%N)

    # GNU time writes its stats to stderr. Redirect trace script's
    # stdout+stderr to TRACE_LOG, and time's own stderr to TIME_LOG.
    set +e
    /usr/bin/time -v bash "${REPO_ROOT}/scripts/trace_goedel_local.sh" \
        --test "$N" --workers "$WORKERS" \
        >"$TRACE_LOG" 2>"$TIME_LOG"
    EXIT_CODE=$?
    set -e

    WALL_END=$(date +%s%N)

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "  FAILED for N=${N} (exit=$EXIT_CODE)"
        echo "  --- Last 20 lines ---"
        tail -20 "$TIME_LOG" 2>/dev/null | sed 's/^/    /'
        echo "  ---"
        RESULTS+=("$(printf "%-6s  %10s  %10s  %8s  %8s  %8s  %10s" \
            "$N" "FAIL" "-" "-" "-" "-" "-")")
        continue
    fi

    WALL_SEC=$(( (WALL_END - WALL_START) / 1000000000 ))
    PER_PROOF=$(( WALL_SEC / N ))

    # Peak RSS from GNU time (kilobytes)
    PEAK_KB=$(grep "Maximum resident" "$TIME_LOG" | awk '{print $NF}')
    PEAK_MB=$(( ${PEAK_KB:-0} / 1024 ))

    # Count pairs
    OUTPUT_FILE="${TRACED_DIR}/goedel_427_test_pairs.jsonl"
    if [[ -f "$OUTPUT_FILE" ]]; then
        PAIRS=$(wc -l < "$OUTPUT_FILE")
    else
        PAIRS=0
    fi
    PPP=$(( PAIRS / (N > 0 ? N : 1) ))

    # Cache+tmp disk usage
    DISK_KB=$(du -sk "${LOCAL_BASE}/cache" "${LOCAL_BASE}/tmp" 2>/dev/null \
        | awk '{s+=$1} END {print s+0}')
    DISK_MB=$(( DISK_KB / 1024 ))

    ROW=$(printf "%-6s  %10s  %10s  %8s  %8s  %8s  %10s" \
        "$N" "${WALL_SEC}s" "${PER_PROOF}s" "$PAIRS" "$PPP" "$PEAK_MB" "$DISK_MB")
    RESULTS+=("$ROW")

    echo "  N=${N}: ${WALL_SEC}s wall, ${PER_PROOF}s/proof, ${PAIRS} pairs, peak=${PEAK_MB}MB"
done

# Summary table
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                        TRACE BENCHMARK SUMMARY                             ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
for line in "${RESULTS[@]}"; do
    echo "║ ${line} ║"
done
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Workers: ${WORKERS}, Machine: $(nproc) cores, $(free -g | awk '/Mem:/{print $2}') GB RAM"

{
    echo "LeanDojo Trace Benchmark — $(date -Iseconds)"
    echo "Workers: ${WORKERS}, Machine: $(nproc) cores, $(free -g | awk '/Mem:/{print $2}') GB RAM"
    echo ""
    for line in "${RESULTS[@]}"; do
        echo "$line"
    done
    echo ""
    echo "Interpretation:"
    echo "  - 'Per-proof' = wall time / N. Compare across N to see if time scales linearly."
    echo "  - 'Peak MB' = peak RSS of the trace process tree. Watch for OOM risk at larger N."
    echo "  - 'Disk MB' = LeanDojo cache + tmp. Extrapolate to estimate full-run disk needs."
    echo "  - 'P/proof' = tactic pairs per proof. Should be ~3-5 for Goedel proofs."
} > "$BENCH_LOG"

echo "Results saved to ${BENCH_LOG}"
