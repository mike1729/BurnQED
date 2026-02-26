#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PANTOGRAPH_DIR="$REPO_ROOT/vendor/Pantograph"
MATHLIB_VERSION="${MATHLIB_VERSION:-v4.27.0}"
SKIP_MATHLIB="${SKIP_MATHLIB:-0}"

echo "=== BurnQED: Pantograph Setup ==="

# Check for elan (Lean version manager)
if ! command -v elan &>/dev/null; then
    echo "ERROR: elan not found. Install it from https://github.com/leanprover/elan"
    exit 1
fi
echo "elan: $(elan --version)"

# Initialize submodule if needed
if [ ! -f "$PANTOGRAPH_DIR/lakefile.lean" ]; then
    echo "Initializing Pantograph submodule..."
    cd "$REPO_ROOT"
    git submodule update --init vendor/Pantograph
fi

# Build base Pantograph
echo "Building Pantograph (this may take a few minutes on first run)..."
cd "$PANTOGRAPH_DIR"
lake build

# Verify the REPL binary exists
if [ -f "$PANTOGRAPH_DIR/.lake/build/bin/repl" ] || [ -f "$PANTOGRAPH_DIR/.lake/build/bin/repl.exe" ]; then
    echo "SUCCESS: Pantograph REPL binary built."
else
    echo "WARNING: REPL binary not found at expected path. 'lake exe repl' may still work."
fi

# Add Mathlib dependency (needed for theorem_index search)
if [ "$SKIP_MATHLIB" -eq 0 ]; then
    echo ""
    echo "=== Adding Mathlib ${MATHLIB_VERSION} ==="

    if ! grep -q "require.*mathlib" "$PANTOGRAPH_DIR/lakefile.lean"; then
        echo "" >> "$PANTOGRAPH_DIR/lakefile.lean"
        echo "require mathlib from git" >> "$PANTOGRAPH_DIR/lakefile.lean"
        echo "  \"https://github.com/leanprover-community/mathlib4\" @ \"${MATHLIB_VERSION}\"" >> "$PANTOGRAPH_DIR/lakefile.lean"
        echo "Added Mathlib ${MATHLIB_VERSION} to lakefile.lean"
    else
        echo "Mathlib already present in lakefile.lean"
    fi

    cd "$PANTOGRAPH_DIR"

    # Check if Mathlib oleans are already present (skip expensive cache get + rebuild)
    if [ -d "$PANTOGRAPH_DIR/.lake/packages/mathlib/.lake/build/lib" ] && \
       [ -n "$(ls -A "$PANTOGRAPH_DIR/.lake/packages/mathlib/.lake/build/lib"/*.olean 2>/dev/null)" ]; then
        echo "Mathlib oleans already present, skipping cache get + rebuild."
    else
        echo "Resolving Mathlib dependency..."
        lake update mathlib

        echo "Downloading prebuilt Mathlib oleans (~2GB, one-time)..."
        lake exe cache get || echo "WARNING: cache get failed, will build from source (slower)"

        echo "Rebuilding with Mathlib..."
        lake build
    fi

    echo "SUCCESS: Pantograph built with Mathlib support."
else
    echo ""
    echo "Skipping Mathlib setup (SKIP_MATHLIB=1)"
fi

echo ""
echo "Setup complete. You can now run integration tests:"
echo "  cargo test -p lean-repl -- --ignored --nocapture"
