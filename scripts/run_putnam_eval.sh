#!/bin/bash
# Evaluate on PutnamBench (672 Putnam exam problems) using `search`.
#
# Uses the `search` subcommand to produce full trajectory parquets, then
# post-processes them into IterationResult-compatible JSON for the summary.
#
# PutnamBench requires BenchPutnam oleans — build first with:
#   ./scripts/prepare_putnam.sh
#
# Usage:
#   ./scripts/run_putnam_eval.sh                          # default (DeepSeek)
#   ITER=1 ./scripts/run_putnam_eval.sh                   # use iter_1 model + EBM
#   TAG=ablation ./scripts/run_putnam_eval.sh             # custom output subdirectory
#   MAX_THEOREMS=50 ./scripts/run_putnam_eval.sh          # cap theorems
#   DRY_RUN=1 ./scripts/run_putnam_eval.sh                # print command without running
#
# Environment:
#   MODEL           Model family: "deepseek" (default).
#                   Selects HF model ID, config TOML, and prompt format.
#   ITER            Iteration number (unset = base model, 0+ = merged model).
#   TAG             Output subdirectory tag (default: "putnam_eval")
#   SGLANG_URL      SGLang inference server (default: http://localhost:30000)
#   ENCODE_URL      Encode server for EBM (default: http://localhost:30001)
#   CONCURRENCY     Parallel theorem searches (default: 3)
#   NUM_WORKERS     Lean REPL pool size (default: 4)
#   MAX_THEOREMS    Cap theorems (default: unlimited)
#   CONFIG          Search config TOML (overrides MODEL-based default)
#   NO_EBM          Set to 1 to skip EBM even if available (default: 0)
#   RESUME          Set to 1 to auto-resume from partial trajectories (default: 1)
#   DRY_RUN         Set to 1 to print commands without executing (default: 0)

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib.sh"

# ── Configuration ─────────────────────────────────────────────────────────

MODEL="${MODEL:-deepseek}"
ITER="${ITER:-}"
TAG="${TAG:-putnam_eval}"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
ENCODE_URL="${ENCODE_URL:-http://localhost:30001}"
CONCURRENCY="${CONCURRENCY:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_THEOREMS="${MAX_THEOREMS:-}"
NO_EBM="${NO_EBM:-0}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"

# ── Model-family defaults ────────────────────────────────────────────────
# MODEL selects HF model ID and config TOML. CONFIG env var overrides.

case "${MODEL,,}" in
    deepseek|ds)
        DEFAULT_LLM_BASE="deepseek-ai/DeepSeek-Prover-V2-7B"
        DEFAULT_CONFIG="${REPO_ROOT}/configs/search_putnam.toml"
        MODEL_LABEL="DeepSeek-Prover-V2-7B"
        ;;
    *)
        echo "ERROR: Unknown MODEL=${MODEL}. Expected 'deepseek'."
        exit 1
        ;;
esac

CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"

PROVER="cargo run --release -p prover-core $CARGO_FEATURES --"

# ── Benchmark files ──────────────────────────────────────────────────────

JSON_FILE="${BENCH_DIR}/putnam.json"
IMPORT_MODULE="BenchPutnam"

if [ ! -f "$JSON_FILE" ]; then
    echo "ERROR: Benchmark file not found: ${JSON_FILE}"
    echo "  Run ./scripts/prepare_putnam.sh to download and generate it."
    exit 1
fi

# ── Resolve model + EBM ──────────────────────────────────────────────────

if [ -z "$ITER" ]; then
    LLM_DIR="${LLM_BASE:-${DEFAULT_LLM_BASE}}"
    ITER_LABEL="base"
else
    LLM_DIR="${LLM_BASE:-${MERGED_MODEL_DIR}/iter_${ITER}}"
    ITER_LABEL="iter_${ITER}"
    if [ ! -d "$LLM_DIR" ]; then
        echo "ERROR: Merged model not found at ${LLM_DIR}"
        echo "  Run: python python/training/export_llm.py --checkpoint data/checkpoints/lora/iter_${ITER} --output ${LLM_DIR}"
        exit 1
    fi
fi

EBM_FLAG=""
ENCODE_FLAG=""
TEMP_FLAG=""
EBM_DIR="${EBM_CKPT_DIR}/iter_${ITER:-base}"
if [ -n "$ITER" ] && [ "$NO_EBM" -eq 0 ] && [ -d "$EBM_DIR" ] && [ -f "${EBM_DIR}/final/model.mpk" ]; then
    EBM_FLAG="--ebm-path ${EBM_DIR}"
    ENCODE_FLAG="--encode-url ${ENCODE_URL}"
    TEMP_FLAG="--temperature 1.4"
fi

# ── Output directory ──────────────────────────────────────────────────────

OUT_DIR="${EVAL_DIR}/${TAG}/${MODEL,,}/${ITER_LABEL}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

TRAJECTORY_FILE="${OUT_DIR}/putnam.parquet"
OUTPUT_FILE="${OUT_DIR}/putnam.json"
LOG_FILE="${LOG_DIR}/putnam_eval_${ITER_LABEL}.log"

# ── Print plan ────────────────────────────────────────────────────────────

echo "================================================================"
echo "  PutnamBench Evaluation — ${MODEL_LABEL} ${ITER_LABEL}"
echo "================================================================"
echo "  Model family: ${MODEL_LABEL}"
echo "  Benchmark:    ${JSON_FILE}"
echo "  Config:       ${CONFIG}"
echo "  Model path:   ${LLM_DIR}"
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

# ── Run search ────────────────────────────────────────────────────────────

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
    echo "[DRY RUN] $CMD"
    rm -f "$RESUME_TMP"
    echo ""
    echo "Dry run complete. No commands were executed."
    exit 0
fi

echo "  Logging to: ${LOG_FILE}"

# shellcheck disable=SC2086
if run_logged "$LOG_FILE" $CMD; then
    echo "  Search complete"
else
    echo "  FAILED (see ${LOG_FILE})"
    rm -f "$RESUME_TMP"
    exit 1
fi

# Clean up resume temp file
rm -f "$RESUME_TMP"

# ── Post-process: parquet → JSON ─────────────────────────────────────────

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
    'iteration': ${ITER:-None},
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

# Print results
print()
print(f'  PutnamBench results:')
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

# ── Summary ───────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "  Results — ${ITER_LABEL}"
echo "================================================================"
echo ""

if [ -f "$OUTPUT_FILE" ]; then
    python3 -c "
import json
with open('$OUTPUT_FILE') as f:
    r = json.load(f)
s = r.get('solved', '?')
t = r.get('total', '?')
rate = r.get('rate', 0)
print(f'  putnam       {s}/{t}  ({rate*100:.1f}%)')
" 2>/dev/null || echo "  putnam: (could not parse results)"
else
    echo "  putnam: (no results file)"
fi

echo ""
echo "All results saved to: ${OUT_DIR}/"
echo "================================================================"
