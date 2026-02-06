#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PANTOGRAPH_DIR="$REPO_ROOT/vendor/Pantograph"

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

# Build Pantograph
echo "Building Pantograph (this may take a few minutes on first run)..."
cd "$PANTOGRAPH_DIR"
lake build

# Verify the REPL binary exists
if [ -f "$PANTOGRAPH_DIR/.lake/build/bin/repl" ] || [ -f "$PANTOGRAPH_DIR/.lake/build/bin/repl.exe" ]; then
    echo "SUCCESS: Pantograph REPL binary built."
else
    echo "WARNING: REPL binary not found at expected path. 'lake exe repl' may still work."
fi

echo ""
echo "Setup complete. You can now run integration tests:"
echo "  cargo test -p lean-repl -- --ignored --nocapture"
