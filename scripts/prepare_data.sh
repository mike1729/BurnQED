#!/bin/bash
# Prepare training data for BurnQED expert iteration experiments.
#
# Orchestrates the full data pipeline:
#   1. Set up Python virtual environment
#   2. Download pre-traced Mathlib data (or trace locally with --trace)
#   3. Format tactic pairs for LLM training
#   4. Validate all output files
#
# Usage:
#   ./scripts/prepare_data.sh                # Download pre-traced data (~5 min, default)
#   ./scripts/prepare_data.sh --trace        # Local LeanDojo trace (requires LeanDojo, hours)
#   ./scripts/prepare_data.sh --force        # Re-run even if outputs exist
#
# Environment variables:
#   MODEL_PATH      Local model dir for tokenizer (optional, speeds up formatting)
#   PYTHON          Python executable (default: python3)
#   MATHLIB_COMMIT  Mathlib4 tag to trace (default: v4.26.0)
#
# Exit codes:
#   0  Success — all outputs validated
#   1  Validation failed
#   2  Python environment setup error
#   3  Mathlib trace error
#   4  Tactic pair formatting error
#
# Note: Embedding precomputation is handled during EBM training (see run_baseline.sh
# --save-embeddings flag). miniF2F extraction is handled automatically in both
# download and trace modes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Configuration ───────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
MATHLIB_COMMIT="${MATHLIB_COMMIT:-v4.26.0}"
MODEL_PATH="${MODEL_PATH:-}"
DATA_DIR="${REPO_ROOT}/data"
VENV_DIR="${REPO_ROOT}/.venv"
REQUIREMENTS="${REPO_ROOT}/python/requirements.txt"

TRACE=0
FORCE=0

# ── Parse arguments ─────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --trace)
            TRACE=1
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
            echo "Usage: ./scripts/prepare_data.sh [--trace] [--force] [--help]"
            exit 1
            ;;
    esac
done

echo "================================================================"
echo "  BurnQED Data Preparation"
echo "================================================================"
echo "  Mode:            $([ $TRACE -eq 1 ] && echo 'LeanDojo trace (local)' || echo 'download pre-traced (default)')"
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
if [ $TRACE -eq 1 ]; then
    if ! python -c "import lean_dojo" 2>/dev/null; then
        echo "Installing Python dependencies (including LeanDojo for --trace mode)..."
        python -m pip install --upgrade pip -q
        python -m pip install -r "$REQUIREMENTS" -q
    fi
else
    # Download mode only needs curl/tar (system) and json (stdlib), but
    # prepare_tactic_pairs.py may need transformers for tokenizer
    echo "Checking minimal dependencies for download mode..."
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

    if [ $TRACE -eq 1 ]; then
        TRACE_ARGS+=("--trace")
        echo "Tracing Mathlib4 at ${MATHLIB_COMMIT} (this may take hours)..."
    else
        echo "Downloading pre-traced LeanDojo data..."
    fi

    if ! python "${REPO_ROOT}/python/data/trace_mathlib.py" "${TRACE_ARGS[@]}"; then
        echo "ERROR: Mathlib trace failed."
        if [ $TRACE -eq 1 ]; then
            echo "Try running without --trace to download pre-traced data instead."
        fi
        exit 3
    fi
    echo "Trace complete."
fi

# ── Step 2b: Download miniF2F ─────────────────────────────────────────────
echo ""
echo "=== Step 2b: Download miniF2F ==="

MINIF2F_TEST="${DATA_DIR}/minif2f_test.json"
MINIF2F_VALID="${DATA_DIR}/minif2f_valid.json"

if [ -f "$MINIF2F_TEST" ] && [ -f "$MINIF2F_VALID" ] && [ $FORCE -eq 0 ]; then
    echo "miniF2F files already exist (use --force to re-download)"
else
    MINIF2F_ARGS=("--output-dir" "$DATA_DIR")
    if [ $FORCE -eq 1 ]; then
        MINIF2F_ARGS+=("--force")
    fi
    if python "${REPO_ROOT}/python/data/download_minif2f.py" "${MINIF2F_ARGS[@]}"; then
        echo "miniF2F download complete."
    else
        echo "WARNING: miniF2F download failed (evaluation will use theorem_index.json instead)"
    fi
fi

# ── Step 2c: Compile benchmark theorems ────────────────────────────────────
echo ""
echo "=== Step 2c: Compile benchmark theorems for Pantograph ==="

PANTOGRAPH_DIR="${REPO_ROOT}/vendor/Pantograph"
BENCH_GENERATOR="${REPO_ROOT}/python/data/generate_benchmark_lean.py"

# Compile each benchmark JSON into a sorry-file + .olean so that
# Pantograph can use copyFrom(name) instead of re-elaborating expressions.
compile_benchmark() {
    local json_path="$1"
    local module_name="$2"
    local lean_file="${PANTOGRAPH_DIR}/${module_name}.lean"
    local olean_file="${PANTOGRAPH_DIR}/.lake/build/lib/lean/${module_name}.olean"

    if [ ! -f "$json_path" ]; then
        echo "  SKIP  ${module_name}: input not found (${json_path})"
        return
    fi

    if [ -f "$olean_file" ] && [ "$olean_file" -nt "$json_path" ] && [ $FORCE -eq 0 ]; then
        echo "  SKIP  ${module_name}: .olean up to date"
        return
    fi

    echo "  Generating ${module_name}.lean..."
    python "$BENCH_GENERATOR" \
        --input "$json_path" \
        --output "$lean_file" \
        --module-name "$module_name"

    echo "  Compiling ${module_name} (one-time cost, may take a few minutes)..."
    if (cd "$PANTOGRAPH_DIR" && lake build "$module_name" 2>&1 | grep -v "^warning:" | head -5); then
        echo "  DONE  ${module_name}: compiled"
    else
        echo "  WARN  ${module_name}: compilation failed (will fall back to expr elaboration)"
    fi
}

compile_benchmark "${DATA_DIR}/minif2f_v2s_test.json"  "BenchMinIF2FV2STest"
compile_benchmark "${DATA_DIR}/minif2f_v2s_valid.json" "BenchMinIF2FV2SValid"

# IMO-Steps benchmarks (if converted)
compile_benchmark "${DATA_DIR}/imo_steps_lemmas.json"   "BenchIMOStepsLemmas"
compile_benchmark "${DATA_DIR}/imo_steps_theorems.json" "BenchIMOStepsTheorems"

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
    echo "    1. Cloud bootstrap:  ./scripts/setup_runpod.sh  (or setup_lambda.sh)"
    echo "    2. Run baseline:     ./scripts/run_baseline.sh"
    echo "    3. Run experiment:   NUM_WORKERS=64 ./scripts/run_all_iterations.sh"
fi
echo "================================================================"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
