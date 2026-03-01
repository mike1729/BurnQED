#!/bin/bash
# Prepare data for BurnQED v2 expert iteration experiments.
#
# Orchestrates the full data pipeline:
#   1. Set up Python virtual environment
#   2. Download HuggingFace training datasets (Lean Workbook, Goedel proofs, LEAN-GitHub, NuminaMath)
#   3. Download miniF2F benchmarks (v1 + v2)
#   4. Generate .lean benchmark files and build oleans (if lake available)
#   5. Validate all outputs
#
# Usage:
#   ./scripts/prepare_data.sh                     # Full pipeline
#   ./scripts/prepare_data.sh --skip-datasets     # Skip large HF dataset downloads
#   ./scripts/prepare_data.sh --force             # Re-download everything
#
# Environment variables:
#   HF_TOKEN        HuggingFace token for authenticated (fast) downloads
#   PYTHON          Python executable (default: python3)
#
# Exit codes:
#   0  Success — all outputs validated
#   1  Validation failed
#   2  Python environment setup error
#   3  miniF2F download error
#   4  HF dataset download error
#   5  Olean build error

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib.sh"

# ── Configuration ───────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
VENV_DIR="${REPO_ROOT}/.venv"
PANTOGRAPH_DIR="${REPO_ROOT}/vendor/Pantograph"

SKIP_DATASETS=0
FORCE=0

# ── Parse arguments ─────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --skip-datasets)
            SKIP_DATASETS=1
            ;;
        --force)
            FORCE=1
            ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: ./scripts/prepare_data.sh [--skip-datasets] [--force] [--help]"
            exit 1
            ;;
    esac
done

echo "================================================================"
echo "  BurnQED v2 Data Preparation"
echo "================================================================"
echo "  Skip HF datasets: $([ $SKIP_DATASETS -eq 1 ] && echo 'yes' || echo 'no')"
echo "  Force re-download: $([ $FORCE -eq 1 ] && echo 'yes' || echo 'no')"
echo "  Output dir:        ${DATA_ROOT}"
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

# Install huggingface_hub CLI for dataset downloads
echo "Installing dependencies..."
python -m pip install --upgrade pip -q
python -m pip install "huggingface_hub[cli]" -q
echo "Python environment ready."

# ── Step 1b: HuggingFace authentication ─────────────────────────────────────
if [ $SKIP_DATASETS -eq 0 ]; then
    echo ""
    echo "=== Step 1b: HuggingFace authentication ==="

    if [ -z "${HF_TOKEN:-}" ]; then
        # Check if already logged in (token cached by previous login)
        if ! huggingface-cli whoami &>/dev/null; then
            echo "HuggingFace authentication required for fast downloads."
            echo "Set HF_TOKEN env var or run: huggingface-cli login"
            echo ""
            echo "Options:"
            echo "  1. Set HF_TOKEN=hf_... and re-run this script"
            echo "  2. Proceed with interactive login now"
            echo "  3. Continue without auth (downloads will be rate-limited)"
            read -rp "Login interactively? [Y/n] " answer
            if [[ "${answer:-Y}" =~ ^[Yy] ]]; then
                huggingface-cli login
            else
                echo "Continuing without HF auth. Downloads may be slow."
            fi
        else
            echo "Already logged in to HuggingFace."
        fi
    else
        export HF_TOKEN
        echo "Using HF_TOKEN from environment."
    fi
fi

# ── Step 2: Download HuggingFace training datasets ──────────────────────────
echo ""
echo "=== Step 2: Download HF training datasets ==="

if [ $SKIP_DATASETS -eq 1 ]; then
    echo "Skipping HF dataset downloads (--skip-datasets)."
else
    HF_DATASETS=(
        "internlm/Lean-Workbook|data/lean/workbook"
        "Goedel-LM/Lean-workbook-proofs|data/lean/goedel_proofs"
        "internlm/Lean-Github|data/lean/lean_github"
        "AI-MO/NuminaMath-LEAN|data/lean/numinamath"
    )

    for entry in "${HF_DATASETS[@]}"; do
        IFS='|' read -r repo local_dir <<< "$entry"
        full_dir="${REPO_ROOT}/${local_dir}"

        if [ $FORCE -eq 0 ] && [ -d "$full_dir" ] && [ "$(ls -A "$full_dir" 2>/dev/null)" ]; then
            echo "  ${repo} already downloaded to ${local_dir} (use --force to re-download)"
            continue
        fi

        echo "  Downloading ${repo}..."
        if ! huggingface-cli download "$repo" --repo-type dataset --local-dir "$full_dir"; then
            echo "ERROR: Failed to download ${repo}"
            exit 4
        fi
        echo "  Downloaded ${repo} → ${local_dir}"
    done

    echo "HF dataset downloads complete."
fi

# ── Step 3: Download miniF2F (v1 + v2) ─────────────────────────────────────
echo ""
echo "=== Step 3: Download miniF2F benchmarks ==="

MINIF2F_ARGS=("--output-dir" "$BENCH_DIR" "--version" "all")
if [ $FORCE -eq 1 ]; then
    MINIF2F_ARGS+=("--force")
fi

if ! python "${REPO_ROOT}/python/data/minif2f/download.py" "${MINIF2F_ARGS[@]}"; then
    echo "ERROR: miniF2F download failed."
    exit 3
fi
echo "miniF2F download complete."

# ── Step 4: Generate miniF2F benchmark oleans ───────────────────────────────
echo ""
echo "=== Step 4: Generate miniF2F benchmark .lean files + oleans ==="

BENCHMARKS=(
    "minif2f_test.json BenchMinIF2FTest"
    "minif2f_valid.json BenchMinIF2FValid"
    "minif2f_v2s_test.json BenchMinIF2FV2STest"
    "minif2f_v2s_valid.json BenchMinIF2FV2SValid"
    "minif2f_v2c_test.json BenchMinIF2FV2CTest"
    "minif2f_v2c_valid.json BenchMinIF2FV2CValid"
)

BENCH_LIBS_TO_BUILD=()

for pair in "${BENCHMARKS[@]}"; do
    read -r json_file module_name <<< "$pair"

    if [ ! -f "${BENCH_DIR}/${json_file}" ]; then
        echo "  ${json_file} not found, skipping ${module_name}"
        continue
    fi

    # Generate .lean file from JSON
    echo "  Generating ${module_name}.lean from ${json_file}..."
    python "${REPO_ROOT}/python/data/generate_lean.py" \
        --input "${BENCH_DIR}/${json_file}" \
        --output "${PANTOGRAPH_DIR}/${module_name}.lean" \
        --module-name "${module_name}"

    # Register lean_lib in lakefile.lean if not already present
    if [ -f "${PANTOGRAPH_DIR}/lakefile.lean" ]; then
        if ! grep -qw "lean_lib ${module_name}" "${PANTOGRAPH_DIR}/lakefile.lean"; then
            printf "\nlean_lib %s {\n}\n" "${module_name}" >> "${PANTOGRAPH_DIR}/lakefile.lean"
            echo "  Registered ${module_name} in lakefile.lean"
        fi
    fi

    BENCH_LIBS_TO_BUILD+=("${module_name}")
done

# Build oleans if lake is available
if [ ${#BENCH_LIBS_TO_BUILD[@]} -gt 0 ]; then
    if command -v lake &>/dev/null; then
        echo "  Building ${#BENCH_LIBS_TO_BUILD[@]} benchmark modules..."
        cd "${PANTOGRAPH_DIR}"
        if lake build "${BENCH_LIBS_TO_BUILD[@]}"; then
            echo "  Built oleans: ${BENCH_LIBS_TO_BUILD[*]}"
        else
            echo "  WARNING: lake build failed. Oleans can be built later with:"
            echo "    cd vendor/Pantograph && lake build ${BENCH_LIBS_TO_BUILD[*]}"
        fi
        cd "$REPO_ROOT"
    else
        echo "  lake not found — skipping olean build."
        echo "  Build later with: cd vendor/Pantograph && lake build ${BENCH_LIBS_TO_BUILD[*]}"
    fi
else
    echo "  No benchmark JSON files found — skipping .lean generation."
fi

# ── Step 5: Validate outputs ───────────────────────────────────────────────
echo ""
echo "=== Step 5: Validate outputs ==="

PASS=0
FAIL=0
WARN=0

check_dir() {
    local path="$1"
    local required="$2"
    local desc="$3"

    if [ ! -d "$path" ] || [ -z "$(ls -A "$path" 2>/dev/null)" ]; then
        if [ "$required" = "required" ]; then
            echo "  FAIL  ${desc}: directory empty or not found"
            FAIL=$((FAIL + 1))
        else
            echo "  WARN  ${desc}: directory empty or not found (optional)"
            WARN=$((WARN + 1))
        fi
    else
        local count
        count=$(find "$path" -type f | head -100 | wc -l)
        echo "  PASS  ${desc}: ${count}+ files"
        PASS=$((PASS + 1))
    fi
}

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

check_lean_file() {
    local path="$1"
    local desc="$2"

    if [ -f "$path" ]; then
        echo "  PASS  ${desc}: exists"
        PASS=$((PASS + 1))
    else
        echo "  WARN  ${desc}: not generated (optional)"
        WARN=$((WARN + 1))
    fi
}

# HF datasets (required unless --skip-datasets)
if [ $SKIP_DATASETS -eq 0 ]; then
    check_dir "${LEAN_DIR}/workbook"       "required" "data/lean/workbook (Lean Workbook)"
    check_dir "${LEAN_DIR}/goedel_proofs"  "required" "data/lean/goedel_proofs (Goedel proofs)"
    check_dir "${LEAN_DIR}/lean_github"    "required" "data/lean/lean_github (LEAN-GitHub)"
    check_dir "${LEAN_DIR}/numinamath"     "required" "data/lean/numinamath (NuminaMath)"
else
    echo "  SKIP  HF datasets (--skip-datasets)"
fi

# miniF2F v1 (required)
check_json_theorems "${BENCH_DIR}/minif2f_test.json"  1 "required" "minif2f_test.json (v1 test)"
check_json_theorems "${BENCH_DIR}/minif2f_valid.json" 1 "required" "minif2f_valid.json (v1 valid)"

# miniF2F v2 — valid splits required, test splits optional
check_json_theorems "${BENCH_DIR}/minif2f_v2s_valid.json" 1 "required" "minif2f_v2s_valid.json (v2s valid)"
check_json_theorems "${BENCH_DIR}/minif2f_v2c_valid.json" 1 "required" "minif2f_v2c_valid.json (v2c valid)"
check_json_theorems "${BENCH_DIR}/minif2f_v2s_test.json"  1 "optional" "minif2f_v2s_test.json (v2s test)"
check_json_theorems "${BENCH_DIR}/minif2f_v2c_test.json"  1 "optional" "minif2f_v2c_test.json (v2c test)"

# Benchmark .lean files
for pair in "${BENCHMARKS[@]}"; do
    read -r json_file module_name <<< "$pair"
    if [ -f "${BENCH_DIR}/${json_file}" ]; then
        check_lean_file "${PANTOGRAPH_DIR}/${module_name}.lean" "${module_name}.lean"
    fi
done

# ── Summary ─────────────────────────────────────────────────────────────────
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
    "${BENCH_DIR}/minif2f_test.json" \
    "${BENCH_DIR}/minif2f_valid.json" \
    "${BENCH_DIR}/minif2f_v2s_valid.json" \
    "${BENCH_DIR}/minif2f_v2c_valid.json" \
    "${BENCH_DIR}/minif2f_v2s_test.json" \
    "${BENCH_DIR}/minif2f_v2c_test.json"; do
    if [ -f "$f" ]; then
        local_path="${f#"$REPO_ROOT/"}"
        size=$(du -h "$f" 2>/dev/null | cut -f1)
        echo "    ${size}  ${local_path}"
    fi
done
for pair in "${BENCHMARKS[@]}"; do
    read -r _ module_name <<< "$pair"
    lean_file="${PANTOGRAPH_DIR}/${module_name}.lean"
    if [ -f "$lean_file" ]; then
        local_path="${lean_file#"$REPO_ROOT/"}"
        size=$(du -h "$lean_file" 2>/dev/null | cut -f1)
        echo "    ${size}  ${local_path}"
    fi
done

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "  Next steps:"
    echo "    1. Inspect datasets:   python python/data/inspect_datasets.py"
    echo "    2. Pantograph validation: (see Phase 0 tasks 0.3d-0.3f)"
    echo "    3. Build oleans (if skipped): cd vendor/Pantograph && lake build"
fi
echo "================================================================"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
