#!/bin/bash
# Copy Goedel data to local NVMe and run LeanDojo tracing.
#
# Network FUSE storage (/workspace) is too slow for the I/O-intensive
# compilation and tracing that LeanDojo performs. This script:
#   1. Rsyncs essential goedel_migration files to local overlay FS
#   2. Runs trace_goedel.py with all cache/tmp on local FS
#   3. Copies output JSONL back to data/traced/
#
# Usage:
#   scripts/trace_goedel_local.sh --test 5     # small test run
#   scripts/trace_goedel_local.sh --test 20    # medium test
#   scripts/trace_goedel_local.sh --full       # full 28K trace
#
# Environment:
#   LOCAL_BASE   - local working directory (default: /root/goedel_trace)
#   WORKERS      - LeanDojo NUM_PROCS (default: 32)

set -euo pipefail

source "$(dirname "$0")/_lib.sh"

# ── Configuration ──────────────────────────────────────────────────────
LOCAL_BASE="${LOCAL_BASE:-/root/goedel_trace}"
WORKERS="${WORKERS:-32}"

GOEDEL_SRC="${LEAN_DIR}/goedel_migration"
LOCAL_REPO="${LOCAL_BASE}/repo"
LOCAL_CACHE="${LOCAL_BASE}/cache"
LOCAL_TMP="${LOCAL_BASE}/tmp"
TRACED_DIR="${DATA_ROOT}/traced"

TRACE_SCRIPT="${REPO_ROOT}/python/data/goedel_migration/trace_goedel.py"

# ── Parse arguments ───────────────────────────────────────────────────
MODE=""
N_PROOFS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            MODE="test"
            N_PROOFS="${2:?--test requires N}"
            shift 2
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "Usage: $0 --test N | --full [--workers N]"
    exit 1
fi

# ── Step 1: Prepare local directory ───────────────────────────────────
echo "=== Setting up local directories ==="
echo "  LOCAL_BASE: ${LOCAL_BASE}"
echo "  WORKERS:    ${WORKERS}"

mkdir -p "$LOCAL_REPO" "$LOCAL_CACHE" "$LOCAL_TMP"

# Check source exists
if [[ ! -d "${GOEDEL_SRC}/GoedelMigration" ]]; then
    echo "ERROR: Goedel source not found at ${GOEDEL_SRC}/GoedelMigration"
    exit 1
fi

# Check local disk space (need ~2 GB for data + cache + oleans)
AVAIL_GB=$(df --output=avail -BG "$LOCAL_BASE" | tail -1 | tr -d ' G')
echo "  Available disk: ${AVAIL_GB} GB"
if [[ "$AVAIL_GB" -lt 5 ]]; then
    echo "ERROR: Less than 5 GB available on local FS. Need at least 5 GB."
    exit 1
fi

# ── Step 2: Copy essential files to local FS ──────────────────────────
echo ""
echo "=== Copying goedel_migration to local FS ==="
echo "  Source: ${GOEDEL_SRC}"
echo "  Dest:   ${LOCAL_REPO}"

# Only copy what LeanDojo needs: .git, lean config, proof sources.
# Skip .lake/ (11 GB of build artifacts — LeanDojo downloads its own via
# `lake exe cache get`) and artifacts/ (357 MB, not needed for tracing).
# Skip copy if GoedelMigration/ already exists locally (previous run)
if [[ -d "${LOCAL_REPO}/GoedelMigration" ]]; then
    echo "  Local copy already exists, skipping copy"
    COPY_ELAPSED=0
else
    COPY_START=$(date +%s)

    if command -v rsync &>/dev/null; then
        rsync -a --info=progress2 \
            --exclude '.lake/' \
            --exclude 'artifacts/' \
            --exclude '*.olean' \
            --exclude '*.ilean' \
            --exclude '*.trace' \
            --exclude '__pycache__/' \
            "${GOEDEL_SRC}/" "${LOCAL_REPO}/"
    else
        # Fallback: selective cp (skip heavy dirs)
        echo "  (rsync not available, using cp)"
        for item in .git .gitignore lakefile.lean lean-toolchain lake-manifest.json \
                    GoedelMigration.lean GoedelMigration goedel_manifest.json; do
            src="${GOEDEL_SRC}/${item}"
            if [[ -e "$src" ]]; then
                cp -a "$src" "${LOCAL_REPO}/"
            fi
        done
    fi

    COPY_END=$(date +%s)
    COPY_ELAPSED=$((COPY_END - COPY_START))
fi

LOCAL_SIZE=$(du -sh "$LOCAL_REPO" | cut -f1)
echo "  Local repo: ${LOCAL_SIZE} (${COPY_ELAPSED}s)"

# Verify or create git repo. LeanDojo requires a valid git repo with at
# least one commit. The source may have an incomplete .git (e.g. from a
# slow FUSE-backed init that timed out), so we re-init locally if needed.
if ! git -C "$LOCAL_REPO" rev-parse HEAD >/dev/null 2>&1; then
    echo "  Git repo invalid or missing — initializing locally..."
    rm -rf "${LOCAL_REPO}/.git"
    git -C "$LOCAL_REPO" init -q
    git -C "$LOCAL_REPO" config user.email "trace@local"
    git -C "$LOCAL_REPO" config user.name "trace"
    echo "  Staging files (this may take a moment for 28K+ files)..."
    git -C "$LOCAL_REPO" add \
        lakefile.lean lean-toolchain GoedelMigration.lean GoedelMigration/
    # Also add lake-manifest.json and .gitignore if present
    for f in lake-manifest.json .gitignore goedel_manifest.json; do
        [[ -f "${LOCAL_REPO}/${f}" ]] && git -C "$LOCAL_REPO" add "$f"
    done
    git -C "$LOCAL_REPO" commit -q -m "Goedel proofs migrated to Lean 4.27"
    echo "  Git repo initialized with fresh commit"
fi
LOCAL_COMMIT=$(git -C "$LOCAL_REPO" rev-parse --short HEAD)
echo "  Git commit: ${LOCAL_COMMIT}"

# ── Step 3: Run trace ─────────────────────────────────────────────────
echo ""
echo "=== Running LeanDojo trace ==="
TRACE_START=$(date +%s)

if [[ "$MODE" == "test" ]]; then
    echo "  Mode: test (${N_PROOFS} proofs)"
    python3 "$TRACE_SCRIPT" \
        --test "$N_PROOFS" \
        --local-dir "$LOCAL_REPO" \
        --cache-dir "$LOCAL_CACHE" \
        --tmp-dir "$LOCAL_TMP" \
        --no-build-deps \
        --workers "$WORKERS" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
elif [[ "$MODE" == "full" ]]; then
    echo "  Mode: full (all passing proofs)"
    python3 "$TRACE_SCRIPT" \
        --full \
        --local-dir "$LOCAL_REPO" \
        --cache-dir "$LOCAL_CACHE" \
        --tmp-dir "$LOCAL_TMP" \
        --no-build-deps \
        --workers "$WORKERS" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi

TRACE_END=$(date +%s)
TRACE_ELAPSED=$((TRACE_END - TRACE_START))

# ── Step 4: Report ────────────────────────────────────────────────────
echo ""
echo "=== Trace complete ==="
echo "  Copy time:   ${COPY_ELAPSED}s"
echo "  Trace time:  ${TRACE_ELAPSED}s"
echo "  Total time:  $((COPY_ELAPSED + TRACE_ELAPSED))s"

# Show cache size
CACHE_SIZE=$(du -sh "$LOCAL_CACHE" 2>/dev/null | cut -f1 || echo "0")
TMP_SIZE=$(du -sh "$LOCAL_TMP" 2>/dev/null | cut -f1 || echo "0")
echo "  Cache size:  ${CACHE_SIZE}"
echo "  Tmp size:    ${TMP_SIZE}"

# Show output
if [[ "$MODE" == "test" ]]; then
    OUTPUT_FILE="${TRACED_DIR}/goedel_427_test_pairs.jsonl"
else
    OUTPUT_FILE="${TRACED_DIR}/goedel_427_pairs.jsonl"
fi

if [[ -f "$OUTPUT_FILE" ]]; then
    PAIRS=$(wc -l < "$OUTPUT_FILE")
    echo "  Output:      ${OUTPUT_FILE} (${PAIRS} pairs)"
else
    echo "  Output:      (not found at ${OUTPUT_FILE})"
fi

echo ""
echo "Local data preserved at ${LOCAL_BASE} for re-runs."
echo "To clean up: rm -rf ${LOCAL_BASE}"
