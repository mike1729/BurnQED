#!/bin/bash
# Trace merged Goedel chunks with LeanDojo.
#
# Phase 1: Build the merged repo (lake build, uses all CPUs)
# Phase 2: Run ExtractData on each chunk file in parallel
# Phase 3: Collect results via LeanDojo's TracedRepo
#
# Usage:
#   scripts/trace_merged.sh [--n-chunks 32] [--workers 12]

set -euo pipefail

source "$(dirname "$0")/_lib.sh"

N_CHUNKS="${N_CHUNKS:-32}"
WORKERS="${WORKERS:-12}"
LOCAL_BASE="${LOCAL_BASE:-/root/goedel_trace}"
MERGED_DIR="${LOCAL_BASE}/merged"
CACHE_DIR="${LOCAL_BASE}/cache"
TMP_DIR="${LOCAL_BASE}/tmp"
TRACED_DIR="${DATA_ROOT}/traced"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-chunks) N_CHUNKS="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Merged Goedel Trace ==="
echo "  Chunks:  ${N_CHUNKS}"
echo "  Workers: ${WORKERS} (concurrent ExtractData)"
echo "  RAM:     $(free -g | awk '/Mem:/{print $2}') GB total, $(free -g | awk '/Mem:/{print $7}') GB available"
echo ""

# ── Step 1: Merge proofs ─────────────────────────────────────────────
if [[ -d "${MERGED_DIR}/GoedelMerged" ]]; then
    echo "=== Merged repo already exists at ${MERGED_DIR} ==="
    CHUNK_COUNT=$(ls "${MERGED_DIR}/GoedelMerged/Chunk_"*.lean 2>/dev/null | wc -l)
    if [[ "$CHUNK_COUNT" -ne "$N_CHUNKS" ]]; then
        echo "  Chunk count mismatch: found ${CHUNK_COUNT}, expected ${N_CHUNKS}"
        echo "  Re-merging..."
        rm -rf "${MERGED_DIR}"
    else
        echo "  Found ${CHUNK_COUNT} chunks, reusing"
    fi
fi

if [[ ! -d "${MERGED_DIR}/GoedelMerged" ]]; then
    echo "=== Merging proofs into ${N_CHUNKS} chunks ==="
    python3 "${REPO_ROOT}/python/data/goedel_migration/merge_proofs.py" \
        --n-chunks "$N_CHUNKS" \
        --output-dir "$MERGED_DIR"
fi

COMMIT=$(git -C "$MERGED_DIR" rev-parse HEAD)
echo "  Commit: ${COMMIT:0:8}"
echo ""

# ── Step 2: Trace with LeanDojo ──────────────────────────────────────
echo "=== Running LeanDojo trace ==="
echo "  Workers (NUM_PROCS): ${WORKERS}"
echo "  Cache: ${CACHE_DIR}"

TRACE_START=$(date +%s)

export CACHE_DIR TMP_DIR
export NUM_PROCS="$WORKERS"
# LAKE_JOBS → LEAN_NUM_THREADS: controls lake build parallelism (each chunk ~16 GB RSS)
# NUM_PROCS: concurrent ExtractData processes (each ~12 GB RSS)
# Lake has no -j flag; our LeanDojo patch maps LAKE_JOBS → LEAN_NUM_THREADS
export LAKE_JOBS="${LAKE_JOBS:-2}"
echo "  LAKE_JOBS (build threads): ${LAKE_JOBS}"

python3 -c "
import os, time, json
os.environ['CACHE_DIR'] = '$CACHE_DIR'
os.environ['TMP_DIR'] = '$TMP_DIR'
os.environ['NUM_PROCS'] = '$WORKERS'

from lean_dojo import LeanGitRepo, trace

repo = LeanGitRepo('$MERGED_DIR', '$COMMIT')
print(f'Lean version: {repo.lean_version}')
print(f'Starting trace at {time.strftime(\"%H:%M:%S\")}')
t0 = time.monotonic()
traced = trace(repo, build_deps=False)
elapsed = time.monotonic() - t0
print(f'Trace complete in {elapsed:.0f}s ({elapsed/60:.1f} min)')

# Extract pairs
import re
BANNED_RE = re.compile(r'\b(?:sorry|admit|cheat|sorryAx)\b')
pairs = []
contaminated = 0
errors = 0
theorems_seen = set()

for tf in traced.traced_files:
    try:
        thms = tf.get_traced_theorems()
    except AttributeError:
        thms = tf.traced_theorems
    for thm in thms:
        if not thm.has_tactic_proof:
            continue
        theorem_name = thm.theorem.full_name
        theorems_seen.add(theorem_name)
        try:
            traced_tactics = thm.get_traced_tactics()
        except Exception as e:
            errors += 1
            continue
        theorem_contaminated = False
        for tt in traced_tactics:
            if BANNED_RE.search(tt.tactic):
                theorem_contaminated = True
                contaminated += 1
                break
        if theorem_contaminated:
            continue
        for i, tt in enumerate(traced_tactics):
            state_before = tt.state_before
            tactic_text = tt.tactic
            if not state_before.strip() or not tactic_text.strip():
                continue
            pairs.append({
                'theorem': theorem_name,
                'state': state_before,
                'tactic': tactic_text,
                'depth': i,
                'source': 'goedel_workbook',
                'num_goals': state_before.count('⊢'),
            })

print(f'Theorems: {len(theorems_seen)}')
print(f'Pairs: {len(pairs)} ({contaminated} contaminated, {errors} errors)')

# Write output
output_path = '$TRACED_DIR/goedel_427_pairs.jsonl'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')
print(f'Output: {output_path}')
"

TRACE_END=$(date +%s)
TRACE_ELAPSED=$((TRACE_END - TRACE_START))

echo ""
echo "=== Trace complete ==="
echo "  Total time: ${TRACE_ELAPSED}s ($(( TRACE_ELAPSED / 60 )) min)"
echo "  Output: ${TRACED_DIR}/goedel_427_pairs.jsonl"
if [[ -f "${TRACED_DIR}/goedel_427_pairs.jsonl" ]]; then
    PAIRS=$(wc -l < "${TRACED_DIR}/goedel_427_pairs.jsonl")
    echo "  Pairs: ${PAIRS}"
fi
