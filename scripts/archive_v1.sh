#!/usr/bin/env bash
# archive_v1.sh — Idempotent archival of v1 git-ignored artifacts + v2 directory setup.
#
# Safe to run on any machine (GPU box or local dev). Moves only generated artifacts,
# NOT source code or base model weights (models/ stays — reusable in v2).
#
# Usage: bash scripts/archive_v1.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Archiving v1 artifacts ==="

# Archive v1 generated artifacts (only if source exists and dest doesn't)
archive_dir() {
    local src="$1"
    local dst="$2"
    if [ -d "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        if [ -d "$dst" ]; then
            echo "  SKIP $src → $dst (already archived)"
        else
            mv "$src" "$dst"
            echo "  MOVED $src → $dst"
        fi
    else
        echo "  SKIP $src (not found)"
    fi
}

archive_dir "checkpoints/ebm"    "archive/v1/checkpoints/ebm"
archive_dir "trajectories"       "archive/v1/trajectories"
archive_dir "data/tactic_pairs"  "archive/v1/data/tactic_pairs"
archive_dir "logs"               "archive/v1/logs"
archive_dir "baselines"          "archive/v1/baselines"

echo ""
echo "=== Creating v2 directory structure ==="

# Data directories (downloads from HuggingFace)
mkdir -p data/{workbook,numinamath,traced}

# Iteration directories
mkdir -p iterations/iter_0/{model,tactic_pairs,trajectories,embeddings,ebm,baselines}
mkdir -p iterations/iter_1/{model,ebm,trajectories,embeddings,baselines,analysis}

echo "  Created data/{workbook,numinamath,traced}"
echo "  Created iterations/iter_0/{model,tactic_pairs,trajectories,embeddings,ebm,baselines}"
echo "  Created iterations/iter_1/{model,ebm,trajectories,embeddings,baselines,analysis}"

echo ""
echo "Done. models/ left in place (reusable for v2)."
