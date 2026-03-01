#!/bin/bash
# Prepare PutnamBench benchmark: download, generate .lean, build oleans.
#
# PutnamBench: 672 Lean 4 competition math problems (Putnam exam, 1962–2025).
# Targets Lean 4.27.0 / Mathlib v4.27.0 — identical to our Pantograph.
#
# Steps:
#   1. Download PutnamBench → putnam.json (TheoremIndex format)
#   2. Generate BenchPutnam.lean from putnam.json (reuses miniF2F generate_lean.py)
#   3. Register BenchPutnam in lakefile.lean if not present
#   4. Build oleans with `lake build BenchPutnam`
#   5. Validate outputs
#
# Usage:
#   ./scripts/prepare_putnam.sh              # Full pipeline
#   ./scripts/prepare_putnam.sh --force      # Re-download and rebuild
#
# Exit codes:
#   0  Success
#   1  Validation failed
#   3  Download error
#   5  Olean build error

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib.sh"

# ── Configuration ───────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
PANTOGRAPH_DIR="${REPO_ROOT}/vendor/Pantograph"

FORCE=0

# ── Parse arguments ─────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --force)
            FORCE=1
            ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: ./scripts/prepare_putnam.sh [--force] [--help]"
            exit 1
            ;;
    esac
done

echo "================================================================"
echo "  PutnamBench Preparation"
echo "================================================================"
echo "  Force re-download: $([ $FORCE -eq 1 ] && echo 'yes' || echo 'no')"
echo "  Output dir:        ${BENCH_DIR}"
echo "================================================================"

# ── Step 1: Download PutnamBench ────────────────────────────────────────────
echo ""
echo "=== Step 1: Download PutnamBench ==="

DOWNLOAD_ARGS=("--output-dir" "$BENCH_DIR")
if [ $FORCE -eq 1 ]; then
    DOWNLOAD_ARGS+=("--force")
fi

if ! $PYTHON "${REPO_ROOT}/python/data/putnam/download.py" "${DOWNLOAD_ARGS[@]}"; then
    echo "ERROR: PutnamBench download failed."
    exit 3
fi

# ── Step 2: Generate BenchPutnam.lean ────────────────────────────────────────
echo ""
echo "=== Step 2: Generate BenchPutnam.lean ==="

PUTNAM_JSON="${BENCH_DIR}/putnam.json"
MODULE_NAME="BenchPutnam"
LEAN_FILE="${PANTOGRAPH_DIR}/${MODULE_NAME}.lean"

if [ ! -f "$PUTNAM_JSON" ]; then
    echo "ERROR: ${PUTNAM_JSON} not found. Download step may have failed."
    exit 3
fi

# PutnamBench theorems use identifiers from many Mathlib namespaces.
# Open the full union so all 672 problems compile in a single module.
# Open namespaces used by PutnamBench theorems. Excluded:
#   Complex (clashes with Real: cos, sin, exp)
#   MvPolynomial (clashes with Polynomial: X, C, eval)
PUTNAM_OPENS=(
    Filter Topology Set Metric Nat Polynomial
    MeasureTheory ProbabilityTheory EuclideanGeometry
    Function Matrix Finset BigOperators
    Bornology RingHom Classical Interval
    InnerProductSpace intervalIntegral
)

SKIP_FILE="${BENCH_DIR}/putnam_skip.txt"
GENERATE_ARGS=(
    --input "$PUTNAM_JSON"
    --output "$LEAN_FILE"
    --module-name "$MODULE_NAME"
    --extra-opens "${PUTNAM_OPENS[@]}"
)
if [ -f "$SKIP_FILE" ]; then
    GENERATE_ARGS+=(--skip-file "$SKIP_FILE")
    echo "  Using skip list: $(wc -l < "$SKIP_FILE") theorems"
fi

echo "  Generating ${MODULE_NAME}.lean from putnam.json..."
$PYTHON "${REPO_ROOT}/python/data/generate_lean.py" "${GENERATE_ARGS[@]}"

# ── Step 3: Register in lakefile.lean ────────────────────────────────────────
echo ""
echo "=== Step 3: Register BenchPutnam in lakefile.lean ==="

if [ -f "${PANTOGRAPH_DIR}/lakefile.lean" ]; then
    if ! grep -qw "lean_lib ${MODULE_NAME}" "${PANTOGRAPH_DIR}/lakefile.lean"; then
        printf "\nlean_lib %s {\n}\n" "${MODULE_NAME}" >> "${PANTOGRAPH_DIR}/lakefile.lean"
        echo "  Registered ${MODULE_NAME} in lakefile.lean"
    else
        echo "  ${MODULE_NAME} already registered in lakefile.lean"
    fi
else
    echo "  WARNING: lakefile.lean not found at ${PANTOGRAPH_DIR}"
fi

# ── Step 4: Build oleans (iterative: skip theorems that fail) ────────────────
echo ""
echo "=== Step 4: Build oleans ==="

if command -v lake &>/dev/null; then
    # Iterative build: some PutnamBench theorems don't compile under our global
    # open statements (namespace clashes, syntax issues). We do up to 3 passes:
    # each pass captures failures, adds them to the skip list, and regenerates.
    MAX_PASSES=3
    for pass in $(seq 1 "$MAX_PASSES"); do
        echo "  Build pass ${pass}/${MAX_PASSES}..."
        cd "${PANTOGRAPH_DIR}"
        BUILD_OUTPUT=$(lake build "${MODULE_NAME}" 2>&1) && {
            echo "  Built oleans for ${MODULE_NAME}"
            cd "$REPO_ROOT"
            break
        }
        cd "$REPO_ROOT"

        # Extract theorem names with errors
        NEW_SKIPS=$(echo "$BUILD_OUTPUT" | grep '^error: BenchPutnam.lean:' | \
            grep -v 'Lean exited\|required\|build failed\|maximum number' | \
            sed 's/error: BenchPutnam.lean:\([0-9]*\):.*/\1/' | sort -un | while read -r line; do
                awk -v errline="$line" 'NR <= errline && /^theorem / { name=$2 } NR == errline { print name }' \
                    "$LEAN_FILE"
            done | sort -u)

        if [ -z "$NEW_SKIPS" ]; then
            echo "  No new failures to skip, but build still failed."
            echo "ERROR: lake build ${MODULE_NAME} failed."
            exit 5
        fi

        NEW_COUNT=$(echo "$NEW_SKIPS" | wc -l)
        echo "  Found ${NEW_COUNT} failing theorems, adding to skip list..."

        # Append to skip list and regenerate
        touch "$SKIP_FILE"
        echo "$NEW_SKIPS" >> "$SKIP_FILE"
        sort -u "$SKIP_FILE" -o "$SKIP_FILE"

        GENERATE_ARGS=(
            --input "$PUTNAM_JSON"
            --output "$LEAN_FILE"
            --module-name "$MODULE_NAME"
            --extra-opens "${PUTNAM_OPENS[@]}"
            --skip-file "$SKIP_FILE"
        )
        $PYTHON "${REPO_ROOT}/python/data/generate_lean.py" "${GENERATE_ARGS[@]}"
    done

    COMPILED=$(grep -c '^theorem ' "$LEAN_FILE" || echo 0)
    SKIPPED=$(wc -l < "$SKIP_FILE" 2>/dev/null || echo 0)
    echo "  Compiled: ${COMPILED}/672 theorems (${SKIPPED} skipped)"
else
    echo "  lake not found — skipping olean build."
    echo "  Build later with: cd vendor/Pantograph && lake build ${MODULE_NAME}"
fi

# ── Step 5: Validate ────────────────────────────────────────────────────────
echo ""
echo "=== Step 5: Validate ==="

PASS=0
FAIL=0

# Check JSON theorem count
THEOREM_COUNT=$($PYTHON -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print(len(data.get('theorems', [])))
" "$PUTNAM_JSON" 2>/dev/null || echo "0")

if [ "$THEOREM_COUNT" -ge 600 ]; then
    echo "  PASS  putnam.json: ${THEOREM_COUNT} theorems (>= 600)"
    PASS=$((PASS + 1))
else
    echo "  FAIL  putnam.json: ${THEOREM_COUNT} theorems (need >= 600)"
    FAIL=$((FAIL + 1))
fi

# Check .lean file exists
if [ -f "$LEAN_FILE" ]; then
    echo "  PASS  ${MODULE_NAME}.lean exists"
    PASS=$((PASS + 1))
else
    echo "  FAIL  ${MODULE_NAME}.lean not found"
    FAIL=$((FAIL + 1))
fi

# Check olean build
if [ -f "${PANTOGRAPH_DIR}/.lake/build/lib/lean/${MODULE_NAME}.olean" ]; then
    echo "  PASS  ${MODULE_NAME} oleans built"
    PASS=$((PASS + 1))
else
    echo "  WARN  ${MODULE_NAME} oleans not found (may need lake build)"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
if [ $FAIL -eq 0 ]; then
    echo "  PutnamBench Preparation PASSED"
else
    echo "  PutnamBench Preparation FAILED"
fi
echo "================================================================"
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo ""
echo "  Output files:"
if [ -f "$PUTNAM_JSON" ]; then
    size=$(du -h "$PUTNAM_JSON" 2>/dev/null | cut -f1)
    echo "    ${size}  data/benchmarks/putnam.json"
fi
if [ -f "$LEAN_FILE" ]; then
    size=$(du -h "$LEAN_FILE" 2>/dev/null | cut -f1)
    echo "    ${size}  vendor/Pantograph/${MODULE_NAME}.lean"
fi
echo ""
echo "  Next: ./scripts/run_putnam_eval.sh"
echo "================================================================"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
