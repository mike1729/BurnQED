#!/bin/bash
# Evaluate on miniF2F benchmarks across multiple versions using `search`.
#
# Uses the `search` subcommand to produce full trajectory parquets, then
# post-processes them into IterationResult-compatible JSON for the summary
# section and `compare` subcommand.
#
# Each version has a precompiled olean library that must be passed via --imports
# so Pantograph can resolve the theorem declarations. Build them first with
# ./scripts/prepare_data.sh or ./scripts/setup_runpod.sh.
#
# Usage:
#   ./scripts/run_minif2f_eval.sh                          # all available versions
#   VERSIONS="v2s_test v2c_test" ./scripts/run_minif2f_eval.sh   # specific versions
#   ITER=1 ./scripts/run_minif2f_eval.sh                   # use iter_1 model + EBM
#   TAG=ablation ./scripts/run_minif2f_eval.sh             # custom output subdirectory
#   DRY_RUN=1 ./scripts/run_minif2f_eval.sh                # print commands without running
#
# Environment:
#   VERSIONS        Space-separated list from: test valid v2s_test v2s_valid v2c_test v2c_valid
#                   Default: all versions whose benchmark JSON exists
#   ITER            Iteration number (default: 0). Determines model + optional EBM.
#   TAG             Output subdirectory tag (default: "minif2f_eval")
#   SGLANG_URL      SGLang inference server (default: http://localhost:30000)
#   ENCODE_URL      Encode server for EBM (default: http://localhost:30001)
#   CONCURRENCY     Parallel theorem searches (default: 4)
#   NUM_WORKERS     Lean REPL pool size (default: 8)
#   MAX_THEOREMS    Cap theorems per version (default: unlimited)
#   CONFIG          Search config TOML (default: configs/search_minif2f.toml)
#   NO_EBM          Set to 1 to skip EBM even if available (default: 0)
#   RESUME          Set to 1 to auto-resume from partial trajectories (default: 1)
#   DRY_RUN         Set to 1 to print commands without executing (default: 0)

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib.sh"

# ── Configuration ─────────────────────────────────────────────────────────

ITER="${ITER:-0}"
TAG="${TAG:-minif2f_eval}"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
ENCODE_URL="${ENCODE_URL:-http://localhost:30001}"
CONCURRENCY="${CONCURRENCY:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_THEOREMS="${MAX_THEOREMS:-}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/search_minif2f.toml}"
NO_EBM="${NO_EBM:-0}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"

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
TEMP_FLAG=""
EBM_DIR="${EBM_CKPT_DIR}/iter_${ITER}"
if [ "$NO_EBM" -eq 0 ] && [ "$ITER" -gt 0 ] && [ -d "$EBM_DIR" ] && [ -f "${EBM_DIR}/final/model.mpk" ]; then
    EBM_FLAG="--ebm-path ${EBM_DIR}"
    ENCODE_FLAG="--encode-url ${ENCODE_URL}"
    TEMP_FLAG="--temperature 1.4"
fi

# ── Output directory ──────────────────────────────────────────────────────

OUT_DIR="${EVAL_DIR}/${TAG}/iter_${ITER}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# ── Print plan ────────────────────────────────────────────────────────────

echo "================================================================"
echo "  miniF2F Evaluation — iter_${ITER}"
echo "================================================================"
echo "  Versions:     ${VERSIONS}"
echo "  Config:       ${CONFIG}"
echo "  Model:        ${LLM_DIR}"
if [ -n "$EBM_FLAG" ]; then
    echo "  EBM:          ${EBM_DIR}"
else
    echo "  EBM:          (none)"
fi
echo "  SGLang:       ${SGLANG_URL}"
echo "  Concurrency:  ${CONCURRENCY}"
echo "  Workers:      ${NUM_WORKERS}"
echo "  Max theorems: ${MAX_THEOREMS:-all}"
echo "  Output:       ${OUT_DIR}/"
if [ "$RESUME" -eq 1 ]; then
    echo "  Resume:       ON (will skip already-searched theorems)"
fi
if [ "$DRY_RUN" -eq 1 ]; then
    echo "  Mode:         DRY RUN (commands printed, not executed)"
fi
echo "================================================================"
echo ""

# ── Ensure server ─────────────────────────────────────────────────────────

if [ "$DRY_RUN" -eq 0 ]; then
    ensure_server "$SGLANG_URL" "$LLM_DIR"
fi

# ── Run search for each version ──────────────────────────────────────────

declare -A RESULT_FILES
FAILED_VERSIONS=""

for VERSION in $VERSIONS; do
    JSON_FILE="${BENCH_DIR}/${BENCH_JSON[$VERSION]}"
    IMPORT_MODULE="${BENCH_IMPORT[$VERSION]}"
    TRAJECTORY_FILE="${OUT_DIR}/4090/${VERSION}.parquet"
    OUTPUT_FILE="${OUT_DIR}/4090/${VERSION}.json"
    LOG_FILE="${LOG_DIR}/4090/minif2f_eval_iter${ITER}_${VERSION}.log"
    RESULT_FILES[$VERSION]="$OUTPUT_FILE"

    echo "── ${VERSION} ──────────────────────────────────────────────────"
    echo "  Theorems:    ${JSON_FILE}"
    echo "  Import:      Mathlib,${IMPORT_MODULE}"
    echo "  Trajectory:  ${TRAJECTORY_FILE}"
    echo "  Summary:     ${OUTPUT_FILE}"

    MAX_FLAG=""
    if [ -n "$MAX_THEOREMS" ]; then
        MAX_FLAG="--max-theorems $MAX_THEOREMS"
    fi

    # Auto-resume from partial trajectory if it exists.
    # The prover only merges when --resume-from differs from --output,
    # so we copy the old file to a .resume temp path.
    RESUME_FLAG=""
    RESUME_TMP="${TRAJECTORY_FILE%.parquet}.resume.parquet"
    if [ "$RESUME" -eq 1 ] && [ -f "$TRAJECTORY_FILE" ] && [ -s "$TRAJECTORY_FILE" ]; then
        DONE_COUNT=$(python3 -c "import pyarrow.parquet as pq; t = pq.read_table('$TRAJECTORY_FILE', columns=['theorem_name']); print(len(set(t.column('theorem_name').to_pylist())))" 2>/dev/null || echo "0")
        if [ "$DONE_COUNT" -gt 0 ]; then
            cp "$TRAJECTORY_FILE" "$RESUME_TMP"
            echo "  Resuming from partial trajectory: ${DONE_COUNT} theorems already done"
            RESUME_FLAG="--resume-from $RESUME_TMP"
        fi
    fi

    CMD="$PROVER search \
        --config $CONFIG \
        --server-url $SGLANG_URL \
        $EBM_FLAG \
        $ENCODE_FLAG \
        $TEMP_FLAG \
        --theorems $JSON_FILE \
        --output $TRAJECTORY_FILE \
        --num-workers $NUM_WORKERS \
        --concurrency $CONCURRENCY \
        $RESUME_FLAG \
        $MAX_FLAG \
        --imports Mathlib,$IMPORT_MODULE"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [DRY RUN] $CMD"
        rm -f "$RESUME_TMP"
        echo ""
        continue
    fi

    echo "  Logging to: ${LOG_FILE}"

    # shellcheck disable=SC2086
    if run_logged "$LOG_FILE" $CMD; then
        echo "  ✓ ${VERSION} search complete"
    else
        echo "  ✗ ${VERSION} FAILED (see ${LOG_FILE})"
        FAILED_VERSIONS="${FAILED_VERSIONS:+$FAILED_VERSIONS }$VERSION"
        rm -f "$RESUME_TMP"
        echo ""
        continue
    fi

    # Clean up resume temp file
    rm -f "$RESUME_TMP"

    # Post-process: parquet → IterationResult JSON for backward compat
    echo "  Converting trajectory to summary JSON..."
    python3 -c "
import json, sys
import pyarrow.parquet as pq
from datetime import datetime, timezone
from collections import defaultdict
from statistics import median

table = pq.read_table('$TRAJECTORY_FILE')
df_names = table.column('theorem_name').to_pylist()
df_complete = table.column('is_proof_complete').to_pylist()
df_timestamp = table.column('timestamp_ms').to_pylist()
df_depth = table.column('depth_from_root').to_pylist()

# Group by theorem
theorems = defaultdict(lambda: {'proved': False, 'nodes': 0, 'ts_min': float('inf'), 'ts_max': 0, 'max_depth': 0, 'proof_depth': None})
for name, complete, ts, depth in zip(df_names, df_complete, df_timestamp, df_depth):
    t = theorems[name]
    t['nodes'] += 1
    if depth > t['max_depth']:
        t['max_depth'] = depth
    if complete:
        t['proved'] = True
        if t['proof_depth'] is None or depth < t['proof_depth']:
            t['proof_depth'] = depth
    if ts < t['ts_min']:
        t['ts_min'] = ts
    if ts > t['ts_max']:
        t['ts_max'] = ts

per_theorem = []
times = []
proof_depths = []
for name in sorted(theorems):
    t = theorems[name]
    time_secs = (t['ts_max'] - t['ts_min']) / 1000.0
    times.append(time_secs)
    reason = '' if t['proved'] else 'budget_exhausted'
    per_theorem.append({
        'name': name,
        'proved': t['proved'],
        'nodes_used': t['nodes'],
        'time_secs': round(time_secs, 2),
        'failure_reason': reason,
        'max_depth': t['max_depth'],
        'proof_depth': t['proof_depth'],
    })
    if t['proof_depth'] is not None:
        proof_depths.append(t['proof_depth'])

solved = sum(1 for t in per_theorem if t['proved'])
total = len(per_theorem)
rate = solved / total if total > 0 else 0.0
avg_nodes = sum(t['nodes_used'] for t in per_theorem) / total if total > 0 else 0.0
avg_time = sum(times) / total if total > 0 else 0.0
med_time = median(times) if times else 0.0

# Depth distribution of proved theorems
depth_dist = defaultdict(int)
for d in proof_depths:
    depth_dist[d] += 1
depth_distribution = {int(k): v for k, v in sorted(depth_dist.items())}

result = {
    'iteration': $ITER if $ITER > 0 else None,
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'llm_path': '$LLM_DIR',
    'ebm_path': '${EBM_DIR}' if '$EBM_FLAG' else None,
    'benchmark': '$JSON_FILE',
    'total_theorems': total,
    'solved': solved,
    'total': total,
    'rate': round(rate, 4),
    'avg_nodes': round(avg_nodes, 1),
    'avg_time_secs': round(avg_time, 1),
    'median_time_secs': round(med_time, 1),
    'depth_distribution': depth_distribution,
    'per_theorem': per_theorem,
}

with open('$OUTPUT_FILE', 'w') as f:
    json.dump(result, f, indent=2)

# Print partial results immediately
print()
print(f'  $VERSION results:')
print(f'    Solved:     {solved}/{total} ({rate*100:.1f}%)')
print(f'    Avg nodes:  {avg_nodes:.1f}')
print(f'    Avg time:   {avg_time:.1f}s   Median: {med_time:.1f}s')

if depth_distribution:
    print(f'    Depth distribution (proved):')
    print(f'      {\"Depth\":>5s} │ {\"Count\":>5s} │ Bar')
    print(f'      {\"─\"*5:s}─┼─{\"─\"*5:s}─┼─{\"─\"*20:s}')
    max_count = max(depth_distribution.values())
    for d, c in sorted(depth_distribution.items()):
        bar = '█' * int(20 * c / max_count) if max_count > 0 else ''
        print(f'      {d:5d} │ {c:5d} │ {bar}')

print(f'    Written to: $OUTPUT_FILE')
"
    echo ""
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run complete. No commands were executed."
    exit 0
fi

# ── Summary ───────────────────────────────────────────────────────────────

echo "================================================================"
echo "  Results — iter_${ITER}"
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
