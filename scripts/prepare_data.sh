#!/bin/bash
# Prepare training data for BurnQED expert iteration experiments.
#
# Orchestrates the full data pipeline:
#   1. Set up Python virtual environment
#   2. Trace Mathlib4 with LeanDojo (or download pre-traced data with --fallback)
#   3. Format tactic pairs for LLM training
#   4. Validate all output files
#
# Usage:
#   ./scripts/prepare_data.sh                     # Full trace (requires LeanDojo, hours)
#   ./scripts/prepare_data.sh --fallback          # Download pre-traced data (~5 min)
#   ./scripts/prepare_data.sh --fallback --force   # Re-run even if outputs exist
#
# Environment variables:
#   MODEL_PATH      Local model dir for tokenizer (optional, speeds up formatting)
#   PYTHON          Python executable (default: python3)
#   MATHLIB_COMMIT  Mathlib4 tag to trace (default: v4.27.0)
#
# Exit codes:
#   0  Success — all outputs validated
#   1  Validation failed
#   2  Python environment setup error
#   3  Mathlib trace error
#   4  Tactic pair formatting error

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Configuration ───────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
MATHLIB_COMMIT="${MATHLIB_COMMIT:-v4.27.0}"
MODEL_PATH="${MODEL_PATH:-}"
DATA_DIR="${REPO_ROOT}/data"
VENV_DIR="${REPO_ROOT}/.venv"
REQUIREMENTS="${REPO_ROOT}/python/requirements.txt"

FALLBACK=0
FORCE=0

# ── Parse arguments ─────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --fallback)
            FALLBACK=1
            ;;
        --force)
            FORCE=1
            ;;
        --help|-h)
            # Print the header comment block as usage
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: ./scripts/prepare_data.sh [--fallback] [--force] [--help]"
            exit 1
            ;;
    esac
done

echo "================================================================"
echo "  BurnQED Data Preparation"
echo "================================================================"
echo "  Fallback mode:   $([ $FALLBACK -eq 1 ] && echo 'yes (download pre-traced)' || echo 'no (LeanDojo trace)')"
echo "  Force re-run:    $([ $FORCE -eq 1 ] && echo 'yes' || echo 'no')"
echo "  Mathlib commit:  ${MATHLIB_COMMIT}"
echo "  Output dir:      ${DATA_DIR}"
if [ -n "$MODEL_PATH" ]; then
    echo "  Tokenizer:       ${MODEL_PATH}"
else
    echo "  Tokenizer:       HuggingFace default (deepseek-ai/DeepSeek-Prover-V2-7B)"
fi
echo "================================================================"

# ── Step 1: Python environment ──────────────────────────────────────────────
echo ""
echo "=== Step 1: Python environment ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    if ! "$PYTHON" -m venv "$VENV_DIR"; then
        echo "ERROR: Failed to create virtual environment."
        echo "Ensure python3-venv is installed: sudo apt-get install python3-venv"
        exit 2
    fi
fi

# Activate venv (cross-platform: check both bin/ and Scripts/)
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/Scripts/activate"
else
    echo "ERROR: Cannot find venv activate script in ${VENV_DIR}"
    exit 2
fi

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Install/upgrade requirements if needed
if ! python -c "import lean_dojo" 2>/dev/null && [ $FALLBACK -eq 0 ]; then
    echo "Installing Python dependencies..."
    python -m pip install --upgrade pip -q
    python -m pip install -r "$REQUIREMENTS" -q
elif [ $FALLBACK -eq 1 ]; then
    # Fallback only needs curl/tar (system) and json (stdlib), but
    # prepare_tactic_pairs.py may need transformers for tokenizer
    echo "Checking minimal dependencies for fallback mode..."
    python -m pip install --upgrade pip -q
    # Install only what's needed: transformers for tokenizer
    python -m pip install "transformers>=4.38.0" -q 2>/dev/null || true
fi
echo "Python environment ready."

# ── Step 2: Trace Mathlib ───────────────────────────────────────────────────
echo ""
echo "=== Step 2: Trace Mathlib ==="

TRACE_OUTPUTS=(
    "${DATA_DIR}/theorem_index.json"
    "${DATA_DIR}/tactic_pairs/train.jsonl"
    "${DATA_DIR}/tactic_pairs/val.jsonl"
)

# Check if trace outputs already exist
TRACE_EXISTS=1
for f in "${TRACE_OUTPUTS[@]}"; do
    if [ ! -f "$f" ]; then
        TRACE_EXISTS=0
        break
    fi
done

if [ $TRACE_EXISTS -eq 1 ] && [ $FORCE -eq 0 ]; then
    echo "Trace outputs already exist (use --force to re-run):"
    for f in "${TRACE_OUTPUTS[@]}"; do
        echo "  $(wc -l < "$f") lines  $f"
    done
else
    TRACE_ARGS=("--output-dir" "$DATA_DIR" "--mathlib-commit" "$MATHLIB_COMMIT")

    if [ $FALLBACK -eq 1 ]; then
        TRACE_ARGS+=("--fallback")
        echo "Downloading pre-traced LeanDojo data..."
    else
        echo "Tracing Mathlib4 at ${MATHLIB_COMMIT} (this may take hours)..."
    fi

    if ! python "${REPO_ROOT}/python/data/trace_mathlib.py" "${TRACE_ARGS[@]}"; then
        echo "ERROR: Mathlib trace failed."
        if [ $FALLBACK -eq 0 ]; then
            echo "Try: ./scripts/prepare_data.sh --fallback"
        fi
        exit 3
    fi
    echo "Trace complete."
fi

# ── Step 3: Format tactic pairs ────────────────────────────────────────────
echo ""
echo "=== Step 3: Format tactic pairs ==="

TOKENIZER_ARG="deepseek-ai/DeepSeek-Prover-V2-7B"
if [ -n "$MODEL_PATH" ]; then
    TOKENIZER_ARG="$MODEL_PATH"
    echo "Using local tokenizer: ${MODEL_PATH}"
else
    echo "Using HuggingFace tokenizer: ${TOKENIZER_ARG}"
fi

# Format train split
TRAIN_RAW="${DATA_DIR}/tactic_pairs/train.jsonl"
TRAIN_FMT="${DATA_DIR}/tactic_pairs/train_formatted.jsonl"

if [ -f "$TRAIN_FMT" ] && [ $FORCE -eq 0 ]; then
    echo "Train formatted already exists: $(wc -l < "$TRAIN_FMT") lines (use --force to re-run)"
else
    if [ ! -f "$TRAIN_RAW" ]; then
        echo "ERROR: ${TRAIN_RAW} not found. Step 2 may have failed."
        exit 4
    fi
    echo "Formatting train split..."
    if ! python "${REPO_ROOT}/python/data/prepare_tactic_pairs.py" \
        --input "$TRAIN_RAW" \
        --output "$TRAIN_FMT" \
        --max-seq-len 2048 \
        --tokenizer "$TOKENIZER_ARG"; then
        echo "ERROR: Failed to format train tactic pairs."
        exit 4
    fi
fi

# Format val split
VAL_RAW="${DATA_DIR}/tactic_pairs/val.jsonl"
VAL_FMT="${DATA_DIR}/tactic_pairs/val_formatted.jsonl"

if [ -f "$VAL_FMT" ] && [ $FORCE -eq 0 ]; then
    echo "Val formatted already exists: $(wc -l < "$VAL_FMT") lines (use --force to re-run)"
else
    if [ ! -f "$VAL_RAW" ]; then
        echo "ERROR: ${VAL_RAW} not found. Step 2 may have failed."
        exit 4
    fi
    echo "Formatting val split..."
    if ! python "${REPO_ROOT}/python/data/prepare_tactic_pairs.py" \
        --input "$VAL_RAW" \
        --output "$VAL_FMT" \
        --max-seq-len 2048 \
        --tokenizer "$TOKENIZER_ARG"; then
        echo "ERROR: Failed to format val tactic pairs."
        exit 4
    fi
fi

# ── Step 4: Validate outputs ───────────────────────────────────────────────
echo ""
echo "=== Step 4: Validate outputs ==="

PASS=0
FAIL=0
WARN=0

check_file() {
    local path="$1"
    local min_lines="$2"
    local required="$3"  # "required" or "optional"
    local desc="$4"

    if [ ! -f "$path" ]; then
        if [ "$required" = "required" ]; then
            echo "  FAIL  ${desc}: file not found"
            FAIL=$((FAIL + 1))
        else
            echo "  WARN  ${desc}: file not found (optional)"
            WARN=$((WARN + 1))
        fi
        return
    fi

    local lines
    lines=$(wc -l < "$path")

    if [ "$lines" -lt "$min_lines" ]; then
        echo "  FAIL  ${desc}: ${lines} lines (need >= ${min_lines})"
        FAIL=$((FAIL + 1))
    else
        echo "  PASS  ${desc}: ${lines} lines"
        PASS=$((PASS + 1))
    fi
}

# For JSON files with {"theorems": [...]}, count theorems not lines
check_json_theorems() {
    local path="$1"
    local min_count="$2"
    local required="$3"
    local desc="$4"

    if [ ! -f "$path" ]; then
        if [ "$required" = "required" ]; then
            echo "  FAIL  ${desc}: file not found"
            FAIL=$((FAIL + 1))
        else
            echo "  WARN  ${desc}: file not found (optional)"
            WARN=$((WARN + 1))
        fi
        return
    fi

    local count
    count=$(python -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print(len(data.get('theorems', [])))
" "$(cygpath -w "$path" 2>/dev/null || echo "$path")" 2>/dev/null || echo "0")

    if [ "$count" -lt "$min_count" ]; then
        echo "  FAIL  ${desc}: ${count} theorems (need >= ${min_count})"
        FAIL=$((FAIL + 1))
    else
        echo "  PASS  ${desc}: ${count} theorems"
        PASS=$((PASS + 1))
    fi
}

check_json_theorems "${DATA_DIR}/theorem_index.json"      1000 "required" "theorem_index.json"
check_json_theorems "${DATA_DIR}/minif2f_test.json"        1    "optional" "minif2f_test.json"
check_json_theorems "${DATA_DIR}/minif2f_valid.json"       1    "optional" "minif2f_valid.json"
check_file          "${DATA_DIR}/tactic_pairs/train.jsonl" 1    "required" "tactic_pairs/train.jsonl"
check_file          "${DATA_DIR}/tactic_pairs/val.jsonl"   1    "required" "tactic_pairs/val.jsonl"
check_file          "$TRAIN_FMT"                           1000 "required" "tactic_pairs/train_formatted.jsonl"
check_file          "$VAL_FMT"                             1    "required" "tactic_pairs/val_formatted.jsonl"

# ── Step 5: Summary ────────────────────────────────────────────────────────
echo ""
echo "================================================================"
if [ $FAIL -eq 0 ]; then
    echo "  Data Preparation PASSED"
else
    echo "  Data Preparation FAILED"
fi
echo "================================================================"
echo "  Results: ${PASS} passed, ${FAIL} failed, ${WARN} warnings"
echo ""
echo "  Output files:"
for f in \
    "${DATA_DIR}/theorem_index.json" \
    "${DATA_DIR}/minif2f_test.json" \
    "${DATA_DIR}/minif2f_valid.json" \
    "${DATA_DIR}/tactic_pairs/train.jsonl" \
    "${DATA_DIR}/tactic_pairs/val.jsonl" \
    "$TRAIN_FMT" \
    "$VAL_FMT"; do
    if [ -f "$f" ]; then
        local_path="${f#"$REPO_ROOT/"}"
        size=$(du -h "$f" 2>/dev/null | cut -f1)
        echo "    ${size}  ${local_path}"
    fi
done

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "  Next steps:"
    echo "    1. Cloud bootstrap:  ./scripts/setup_cloud.sh"
    echo "    2. Run baseline:     ./scripts/run_baseline.sh"
    echo "    3. Run experiment:   NUM_WORKERS=64 ./scripts/run_all_iterations.sh"
fi
echo "================================================================"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
