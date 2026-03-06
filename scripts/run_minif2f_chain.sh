#!/bin/bash
# Chain: monitor pass@32 → resume base eval → run pass@128
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

check_progress() {
    python3 -c "
import pyarrow.parquet as pq, sys
try:
    t = pq.read_table('$1', columns=['theorem_name', 'is_proof_complete'])
    names = set(t.column('theorem_name').to_pylist())
    proved = sum(1 for p in t.column('is_proof_complete').to_pylist() if p)
    print(f'{proved}/{len(names)} ({100*proved/len(names):.1f}%)')
except:
    print('no data yet')
" 2>/dev/null
}

echo "=== Step 1: Waiting for pass@32 to finish ==="
PASS32_FILE="data/evals/minif2f_pass32/deepseek/base/v2s_test.parquet"
while true; do
    # Check if the eval process is still running
    if ! pgrep -f "minif2f_pass32" > /dev/null 2>&1 && \
       ! pgrep -f "search_minif2f_pass32" > /dev/null 2>&1; then
        # Process gone — check if we have results
        if [ -f "$PASS32_FILE" ]; then
            DONE=$(python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('$PASS32_FILE', columns=['theorem_name'])
print(len(set(t.column('theorem_name').to_pylist())))
" 2>/dev/null || echo "0")
            if [ "$DONE" -ge 200 ]; then
                echo "pass@32 complete: $(check_progress $PASS32_FILE)"
                break
            fi
        fi
        echo "pass@32 process not found and results incomplete. Waiting..."
    else
        echo "  pass@32 progress: $(check_progress $PASS32_FILE)"
    fi
    sleep 60
done

echo ""
echo "=== Step 2: Resume base hybrid eval ==="
ITER= VERSIONS=v2s_test ./scripts/run_minif2f_eval.sh
echo ""

echo "=== Step 3: Run pass@128 8K eval ==="
ITER= VERSIONS=v2s_test CONFIG=configs/search_minif2f_pass128.toml TAG=minif2f_pass128 ./scripts/run_minif2f_eval.sh
echo ""

echo "=== All done ==="
echo "Results:"
echo "  pass@32:  $(check_progress data/evals/minif2f_pass32/deepseek/base/v2s_test.parquet)"
echo "  hybrid:   $(check_progress data/evals/minif2f_eval/deepseek/base/v2s_test.parquet)"
echo "  pass@128: $(check_progress data/evals/minif2f_pass128/deepseek/base/v2s_test.parquet)"
