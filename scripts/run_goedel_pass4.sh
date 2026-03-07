#!/bin/bash
# Search on Goedel theorems with pass@4 (base model).
#
# Step 1: Use pre-built benchmark JSON (or extract from traced pairs)
# Step 2: Run pass@4 search
#
# Usage:
#   DEPTH=4plus ./scripts/run_goedel_pass4.sh          # depth >= 4 (default)
#   DEPTH=3 ./scripts/run_goedel_pass4.sh              # depth == 3
#   CONCURRENCY=4 NUM_WORKERS=8 ./scripts/run_goedel_pass4.sh
#   MAX_THEOREMS=100 ./scripts/run_goedel_pass4.sh     # quick test
#   DRY_RUN=1 ./scripts/run_goedel_pass4.sh            # print command only

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib.sh"

SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
CONCURRENCY="${CONCURRENCY:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_THEOREMS="${MAX_THEOREMS:-}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
DEPTH="${DEPTH:-4plus}"

# ── Resolve depth filter ────────────────────────────────────────────────

case "$DEPTH" in
    4plus|4+|geq4)
        DEPTH_LABEL="depth4plus"
        DEPTH_DESC="depth >= 4"
        DEPTH_FILTER="max(p['depth'] for p in ps) >= 4"
        ;;
    3)
        DEPTH_LABEL="depth3"
        DEPTH_DESC="depth == 3"
        DEPTH_FILTER="max(p['depth'] for p in ps) == 3"
        ;;
    3plus|3+|geq3)
        DEPTH_LABEL="depth3plus"
        DEPTH_DESC="depth >= 3"
        DEPTH_FILTER="max(p['depth'] for p in ps) >= 3"
        ;;
    *)
        echo "ERROR: Unknown DEPTH=${DEPTH}. Use: 3, 3plus, 4plus"
        exit 1
        ;;
esac

TAG="${TAG:-goedel_pass4}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/search_goedel_pass4.toml}"
PROVER="cargo run --release -p prover-core $CARGO_FEATURES --"

BENCH_FILE="${BENCH_DIR}/goedel_${DEPTH_LABEL}.json"
OUT_DIR="${EVAL_DIR}/${TAG}/base"
TRAJ_FILE="${OUT_DIR}/goedel_${DEPTH_LABEL}.parquet"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# ── Step 1: Prepare theorem benchmark ────────────────────────────────────

if [ ! -f "$BENCH_FILE" ] || [ "${FORCE_PREPARE:-0}" -eq 1 ]; then
    echo "=== Step 1: Extracting ${DEPTH_DESC} Goedel theorems ==="

    python3 -c "
import json
from collections import defaultdict, Counter

# Load traced pairs
pairs_file = '${REPO_ROOT}/data/traced/pantograph_pairs/goedel_427_pairs.jsonl'
theorems = defaultdict(list)
with open(pairs_file) as f:
    for line in f:
        p = json.loads(line)
        theorems[p['theorem']].append(p)

# Filter by depth
selected = {thm: ps for thm, ps in theorems.items()
            if ${DEPTH_FILTER}}

print(f'Total theorems: {len(theorems)}')
print(f'${DEPTH_DESC}: {len(selected)}')

# Extract statement from root goal state
output_theorems = []
missing = 0
for thm in sorted(selected.keys()):
    root = [p for p in selected[thm] if p['depth'] == 0]
    if not root:
        missing += 1
        continue
    state = root[0]['state']
    lines = state.strip().split('\n')
    goal_idx = next((i for i, l in enumerate(lines) if l.startswith('⊢')), None)
    if goal_idx is None:
        missing += 1
        continue
    hyps = lines[:goal_idx]
    # Full goal = everything after ⊢, including continuation lines
    goal_lines = [lines[goal_idx][2:].strip()]
    for l in lines[goal_idx + 1:]:
        goal_lines.append(l)
    goal = ' '.join(gl.strip() for gl in goal_lines)
    # Build binders from hypothesis lines
    binder_parts = []
    for h in hyps:
        h = h.strip()
        if ' : ' in h:
            binder_parts.append(f'({h})')
    if binder_parts:
        stmt = '∀ ' + ' '.join(binder_parts) + ', ' + goal
    else:
        stmt = goal
    output_theorems.append({'name': thm, 'statement': stmt})

# Depth distribution
depth_dist = Counter(max(p['depth'] for p in selected[t['name']]) for t in output_theorems)
print(f'Output: {len(output_theorems)} theorems ({missing} skipped)')
print('Depth distribution:')
for d in sorted(depth_dist.keys()):
    print(f'  depth {d}: {depth_dist[d]}')

with open('${BENCH_FILE}', 'w') as f:
    json.dump({'theorems': output_theorems}, f, ensure_ascii=False)
print(f'Written to ${BENCH_FILE}')
"
else
    THEOREM_COUNT=$(python3 -c "import json; print(len(json.load(open('$BENCH_FILE'))['theorems']))")
    echo "=== Step 1: Using existing benchmark (${THEOREM_COUNT} theorems) ==="
fi

# ── Step 2: Run pass@4 search ───────────────────────────────────────────

THEOREM_COUNT=$(python3 -c "import json; print(len(json.load(open('$BENCH_FILE'))['theorems']))")

echo ""
echo "================================================================"
echo "  Goedel pass@4 Search — Base Model (${DEPTH_DESC})"
echo "================================================================"
echo "  Theorems:     ${THEOREM_COUNT} (${DEPTH_DESC})"
echo "  Config:       ${CONFIG}"
echo "  SGLang:       ${SGLANG_URL}"
echo "  Concurrency:  ${CONCURRENCY}"
echo "  Workers:      ${NUM_WORKERS}"
echo "  Max theorems: ${MAX_THEOREMS:-all}"
echo "  Output:       ${TRAJ_FILE}"
if [ "$RESUME" -eq 1 ]; then
    echo "  Resume:       ON"
fi
echo "================================================================"

if [ "$DRY_RUN" -eq 0 ]; then
    ensure_sglang "$SGLANG_URL"
fi

MAX_FLAG=""
if [ -n "$MAX_THEOREMS" ]; then
    MAX_FLAG="--max-theorems $MAX_THEOREMS"
fi

# Auto-resume
RESUME_FLAG=""
RESUME_TMP="${TRAJ_FILE%.parquet}.resume.parquet"
if [ "$RESUME" -eq 1 ] && [ -f "$TRAJ_FILE" ] && [ -s "$TRAJ_FILE" ]; then
    DONE_COUNT=$(python3 -c "import pyarrow.parquet as pq; t = pq.read_table('$TRAJ_FILE', columns=['theorem_name']); print(len(set(t.column('theorem_name').to_pylist())))" 2>/dev/null || echo "0")
    if [ "$DONE_COUNT" -gt 0 ]; then
        cp "$TRAJ_FILE" "$RESUME_TMP"
        echo "  Resuming: ${DONE_COUNT} theorems already done"
        RESUME_FLAG="--resume-from $RESUME_TMP"
    fi
fi

LOG_FILE="${LOG_DIR}/goedel_pass4_${DEPTH_LABEL}_base.log"

CMD="$PROVER search \
    --config $CONFIG \
    --server-url $SGLANG_URL \
    --theorems $BENCH_FILE \
    --output $TRAJ_FILE \
    --num-workers $NUM_WORKERS \
    --concurrency $CONCURRENCY \
    $RESUME_FLAG \
    $MAX_FLAG \
    --imports Mathlib"

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "[DRY RUN] $CMD"
    rm -f "$RESUME_TMP"
    exit 0
fi

echo "  Logging to: ${LOG_FILE}"

# shellcheck disable=SC2086
if run_logged "$LOG_FILE" $CMD; then
    echo "Search complete!"
else
    echo "Search FAILED (see ${LOG_FILE})"
    rm -f "$RESUME_TMP"
    exit 1
fi

rm -f "$RESUME_TMP"

# ── Post-process ────────────────────────────────────────────────────────

echo ""
echo "=== Results ==="
$PROVER summary --input "$TRAJ_FILE"

echo ""
echo "Trajectory: ${TRAJ_FILE}"
echo "================================================================"
