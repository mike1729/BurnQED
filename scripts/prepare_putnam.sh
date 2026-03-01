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

echo "  Generating ${MODULE_NAME}.lean from putnam.json..."
$PYTHON "${REPO_ROOT}/python/data/minif2f/generate_lean.py" \
    --input "$PUTNAM_JSON" \
    --output "$LEAN_FILE" \
    --module-name "$MODULE_NAME"

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

# ── Step 4: Build oleans ────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Build oleans ==="

if command -v lake &>/dev/null; then
    echo "  Building ${MODULE_NAME}..."
    cd "${PANTOGRAPH_DIR}"
    if lake build "${MODULE_NAME}"; then
        echo "  Built oleans for ${MODULE_NAME}"
    else
        echo "ERROR: lake build ${MODULE_NAME} failed."
        echo "  Check build output above for errors."
        cd "$REPO_ROOT"
        exit 5
    fi
    cd "$REPO_ROOT"
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

# Check olean build (look for .olean or build directory)
if [ -d "${PANTOGRAPH_DIR}/.lake/build/lib/${MODULE_NAME}" ] || \
   [ -f "${PANTOGRAPH_DIR}/.lake/build/lib/${MODULE_NAME}.olean" ]; then
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
