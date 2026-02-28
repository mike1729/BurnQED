#!/bin/bash
# Evaluate on miniF2F benchmarks across multiple versions.
#
# Each version has a precompiled olean library that must be passed via --imports
# so Pantograph can resolve the theorem declarations. Build them first with
# ./scripts/prepare_data.sh or ./scripts/setup_runpod.sh.
#
# Usage:
#   ./scripts/run_minif2f_eval.sh                          # all available versions
#   VERSIONS="v2s_test v2c_test" ./scripts/run_minif2f_eval.sh   # specific versions
#   ITER=1 ./scripts/run_minif2f_eval.sh                   # use iter_1 model + EBM
#   BUDGET=300 ./scripts/run_minif2f_eval.sh               # override node budget
#   TAG=ablation ./scripts/run_minif2f_eval.sh             # custom output subdirectory
#   DRY_RUN=1 ./scripts/run_minif2f_eval.sh                # print commands without running
#
# Environment:
#   VERSIONS        Space-separated list from: test valid v2s_test v2s_valid v2c_test v2c_valid
#                   Default: all versions whose benchmark JSON exists
#   ITER            Iteration number (default: 0). Determines model + optional EBM.
#   BUDGET          Node budget per theorem (default: 600)
#   TAG             Output subdirectory tag (default: "minif2f_eval")
#   SGLANG_URL      SGLang inference server (default: http://localhost:30000)
#   ENCODE_URL      Encode server for EBM (default: http://localhost:30001)
#   CONCURRENCY     Parallel theorem searches (default: 5)
#   NUM_WORKERS     Lean REPL pool size (default: 8)
#   MAX_THEOREMS    Cap theorems per version (default: unlimited)
#   NUM_CANDIDATES  Tactic candidates per node (default: 16)
#   CONFIG          Search config TOML (default: configs/search_minif2f.toml)
#   NO_EBM          Set to 1 to skip EBM even if available (default: 0)
#   DRY_RUN         Set to 1 to print commands without executing (default: 0)
#   PASS_N          Best-of-N attempts per theorem (default: 1)

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib.sh"

# ── Configuration ─────────────────────────────────────────────────────────

ITER="${ITER:-0}"
BUDGET="${BUDGET:-600}"
TAG="${TAG:-minif2f_eval}"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
ENCODE_URL="${ENCODE_URL:-http://localhost:30001}"
CONCURRENCY="${CONCURRENCY:-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_THEOREMS="${MAX_THEOREMS:-}"
NUM_CANDIDATES="${NUM_CANDIDATES:-16}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/search_minif2f.toml}"
NO_EBM="${NO_EBM:-0}"
DRY_RUN="${DRY_RUN:-0}"
PASS_N="${PASS_N:-1}"

PROVER="cargo run --release -p prover-core $CARGO_FEATURES --"

# ── Benchmark → import mapping ────────────────────────────────────────────
# Each entry: version_key  json_filename  import_module_name
#
# The import module must be precompiled (lake build <module>) inside
# vendor/Pantograph so that `lake exe repl Mathlib <module>` can load it.

declare -A BENCH_JSON BENCH_IMPORT
BENCH_JSON[test]="minif2f_test.json";           BENCH_IMPORT[test]="BenchMinIF2FTest"
BENCH_JSON[valid]="minif2f_valid.json";          BENCH_IMPORT[valid]="BenchMinIF2FValid"
BENCH_JSON[v2s_test]="minif2f_v2s_test.json";   BENCH_IMPORT[v2s_test]="BenchMinIF2FV2STest"
BENCH_JSON[v2s_valid]="minif2f_v2s_valid.json";  BENCH_IMPORT[v2s_valid]="BenchMinIF2FV2SValid"
BENCH_JSON[v2c_test]="minif2f_v2c_test.json";   BENCH_IMPORT[v2c_test]="BenchMinIF2FV2CTest"
BENCH_JSON[v2c_valid]="minif2f_v2c_valid.json";  BENCH_IMPORT[v2c_valid]="BenchMinIF2FV2CValid"

ALL_VERSIONS="test valid v2s_test v2s_valid v2c_test v2c_valid"

# ── Resolve which versions to run ─────────────────────────────────────────

if [ -n "${VERSIONS:-}" ]; then
    # User-specified versions — validate them
    for v in $VERSIONS; do
        if [ -z "${BENCH_JSON[$v]:-}" ]; then
            echo "ERROR: Unknown version '$v'. Choose from: $ALL_VERSIONS"
            exit 1
        fi
        if [ ! -f "${BENCH_DIR}/${BENCH_JSON[$v]}" ]; then
            echo "ERROR: Benchmark file not found: ${BENCH_DIR}/${BENCH_JSON[$v]}"
            echo "  Run ./scripts/prepare_data.sh to generate it."
            exit 1
        fi
    done
else
    # Auto-detect: run all versions whose JSON file exists
    VERSIONS=""
    for v in $ALL_VERSIONS; do
        if [ -f "${BENCH_DIR}/${BENCH_JSON[$v]}" ]; then
            VERSIONS="${VERSIONS:+$VERSIONS }$v"
        fi
    done
    if [ -z "$VERSIONS" ]; then
        echo "ERROR: No miniF2F benchmark files found in ${BENCH_DIR}/"
        echo "  Run ./scripts/prepare_data.sh to download and generate them."
        exit 1
    fi
fi

# ── Resolve model + EBM ──────────────────────────────────────────────────

if [ "$ITER" -eq 0 ]; then
    LLM_DIR="${LLM_BASE:-deepseek-ai/DeepSeek-Prover-V2-7B}"
else
    LLM_DIR="${MERGED_MODEL_DIR}/iter_${ITER}"
    if [ ! -d "$LLM_DIR" ]; then
        echo "ERROR: Merged model not found at ${LLM_DIR}"
        echo "  Run ./scripts/run_iteration_train.sh ${ITER} first."
        exit 1
    fi
fi

EBM_FLAG=""
ENCODE_FLAG=""
EBM_DIR="${EBM_CKPT_DIR}/iter_${ITER}"
if [ "$NO_EBM" -eq 0 ] && [ "$ITER" -gt 0 ] && [ -d "$EBM_DIR" ] && [ -f "${EBM_DIR}/final/model.mpk" ]; then
    EBM_FLAG="--ebm-path ${EBM_DIR}"
    ENCODE_FLAG="--encode-url ${ENCODE_URL}"
fi

# ── Output directory ──────────────────────────────────────────────────────

OUT_DIR="${EVAL_DIR}/${TAG}/iter_${ITER}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# ── Print plan ────────────────────────────────────────────────────────────

echo "================================================================"
echo "  miniF2F Evaluation — iter_${ITER}"
echo "================================================================"
echo "  Versions:     ${VERSIONS}"
echo "  Budget:       ${BUDGET} nodes"
echo "  Pass@N:       ${PASS_N}"
echo "  Config:       ${CONFIG}"
echo "  Model:        ${LLM_DIR}"
if [ -n "$EBM_FLAG" ]; then
    echo "  EBM:          ${EBM_DIR}"
else
    echo "  EBM:          (none)"
fi
echo "  SGLang:       ${SGLANG_URL}"
echo "  Candidates:   ${NUM_CANDIDATES}"
echo "  Concurrency:  ${CONCURRENCY}"
echo "  Workers:      ${NUM_WORKERS}"
echo "  Max theorems: ${MAX_THEOREMS:-all}"
echo "  Output:       ${OUT_DIR}/"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "  Mode:         DRY RUN (commands printed, not executed)"
fi
echo "================================================================"
echo ""

# ── Ensure server ─────────────────────────────────────────────────────────

if [ "$DRY_RUN" -eq 0 ]; then
    ensure_server "$SGLANG_URL" "$LLM_DIR"
fi

# ── Run eval for each version ─────────────────────────────────────────────

declare -A RESULT_FILES
FAILED_VERSIONS=""

for VERSION in $VERSIONS; do
    JSON_FILE="${BENCH_DIR}/${BENCH_JSON[$VERSION]}"
    IMPORT_MODULE="${BENCH_IMPORT[$VERSION]}"
    OUTPUT_FILE="${OUT_DIR}/${VERSION}.json"
    LOG_FILE="${LOG_DIR}/minif2f_eval_iter${ITER}_${VERSION}.log"
    RESULT_FILES[$VERSION]="$OUTPUT_FILE"

    echo "── ${VERSION} ──────────────────────────────────────────────────"
    echo "  Theorems: ${JSON_FILE}"
    echo "  Import:   Mathlib,${IMPORT_MODULE}"
    echo "  Output:   ${OUTPUT_FILE}"

    MAX_FLAG=""
    if [ -n "$MAX_THEOREMS" ]; then
        MAX_FLAG="--max-theorems $MAX_THEOREMS"
    fi

    PASS_FLAG=""
    if [ "$PASS_N" -gt 1 ]; then
        PASS_FLAG="--pass-n $PASS_N"
    fi

    CMD="$PROVER eval \
        --config $CONFIG \
        --server-url $SGLANG_URL \
        $EBM_FLAG \
        $ENCODE_FLAG \
        --theorems $JSON_FILE \
        --budgets $BUDGET \
        --output $OUTPUT_FILE \
        --num-workers $NUM_WORKERS \
        --concurrency $CONCURRENCY \
        --num-candidates $NUM_CANDIDATES \
        $PASS_FLAG \
        $MAX_FLAG \
        --imports Mathlib,$IMPORT_MODULE"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [DRY RUN] $CMD"
        echo ""
        continue
    fi

    echo "  Logging to: ${LOG_FILE}"

    # shellcheck disable=SC2086
    if run_logged "$LOG_FILE" $CMD; then
        echo "  ✓ ${VERSION} complete"
    else
        echo "  ✗ ${VERSION} FAILED (see ${LOG_FILE})"
        FAILED_VERSIONS="${FAILED_VERSIONS:+$FAILED_VERSIONS }$VERSION"
    fi
    echo ""
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run complete. No commands were executed."
    exit 0
fi

# ── Summary ───────────────────────────────────────────────────────────────

echo "================================================================"
echo "  Results — iter_${ITER}, budget=${BUDGET}"
echo "================================================================"
echo ""

COMPARE_FILES=()

for VERSION in $VERSIONS; do
    OUTPUT_FILE="${RESULT_FILES[$VERSION]}"
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "  ${VERSION}: (no results file)"
        continue
    fi

    COMPARE_FILES+=("$OUTPUT_FILE")

    python3 -c "
import json, sys
with open('$OUTPUT_FILE') as f:
    r = json.load(f)
version = '$VERSION'
for br in r.get('budget_results', []):
    print(f'  {version:12s}  budget {br[\"budget\"]:>4d}: {br[\"solved\"]:>3d}/{br[\"total\"]:>3d}  ({br[\"rate\"]*100:5.1f}%)  avg_nodes={br.get(\"avg_nodes\", 0):.0f}  avg_time={br.get(\"avg_time_secs\", 0):.1f}s')
if not r.get('budget_results'):
    s = r.get('solved', r.get('cumulative_solved', '?'))
    t = r.get('total', r.get('total_theorems', '?'))
    rate = r.get('rate', r.get('cumulative_rate', 0))
    print(f'  {version:12s}  {s}/{t}  ({rate*100:.1f}%)')
" 2>/dev/null || echo "  ${VERSION}: (could not parse results)"
done

echo ""

# Cross-version comparison if multiple results exist and compare subcommand works
if [ ${#COMPARE_FILES[@]} -gt 1 ]; then
    echo "── Cross-Version Comparison ──"
    # shellcheck disable=SC2086
    $PROVER compare --results "${COMPARE_FILES[@]}" 2>/dev/null || true
    echo ""
fi

if [ -n "$FAILED_VERSIONS" ]; then
    echo "WARNING: The following versions failed: ${FAILED_VERSIONS}"
    echo "  Check logs in ${LOG_DIR}/"
    exit 1
fi

echo "All results saved to: ${OUT_DIR}/"
echo "================================================================"
