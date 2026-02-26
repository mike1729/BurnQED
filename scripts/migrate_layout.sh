#!/bin/bash
# Migrate BurnQED project layout: consolidate all generated artifacts under data/.
#
# This script is idempotent — safe to run multiple times. It skips moves
# where the source doesn't exist or the destination already has content.
#
# Usage:
#   ./scripts/migrate_layout.sh           # Dry run (shows what would happen)
#   ./scripts/migrate_layout.sh --apply   # Actually move files

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=1
if [ "${1:-}" = "--apply" ]; then
    DRY_RUN=0
fi

MOVED=0
SKIPPED=0
ERRORS=0

# Move a file or directory from old location to new location.
# Skips if source doesn't exist or destination already exists with content.
migrate() {
    local src="$1"
    local dst="$2"
    local desc="${3:-}"

    if [ ! -e "$src" ]; then
        return
    fi

    # If destination already has content, skip
    if [ -e "$dst" ]; then
        if [ -d "$dst" ] && [ -n "$(ls -A "$dst" 2>/dev/null)" ]; then
            echo "  SKIP  ${src} → ${dst} (destination already has content)"
            SKIPPED=$((SKIPPED + 1))
            return
        elif [ -f "$dst" ]; then
            echo "  SKIP  ${src} → ${dst} (destination file exists)"
            SKIPPED=$((SKIPPED + 1))
            return
        fi
    fi

    if [ $DRY_RUN -eq 1 ]; then
        echo "  WOULD MOVE  ${src} → ${dst}  ${desc}"
    else
        mkdir -p "$(dirname "$dst")"
        mv "$src" "$dst" 2>/dev/null || {
            echo "  ERROR  Failed to move ${src} → ${dst}"
            ERRORS=$((ERRORS + 1))
            return
        }
        echo "  MOVED  ${src} → ${dst}"
    fi
    MOVED=$((MOVED + 1))
}

echo "================================================================"
if [ $DRY_RUN -eq 1 ]; then
    echo "  BurnQED Layout Migration (DRY RUN)"
    echo "  Run with --apply to actually move files"
else
    echo "  BurnQED Layout Migration (APPLYING)"
fi
echo "================================================================"
echo ""

# ── Create target directory structure ─────────────────────────────────────
echo "=== Creating directory structure ==="
DIRS=(
    data/lean/workbook
    data/lean/goedel_proofs
    data/lean/lean_github
    data/lean/numinamath
    data/benchmarks
    data/sft
    data/traced
    data/models/base
    data/models/merged
    data/checkpoints/lora
    data/checkpoints/ebm
    data/trajectories
    data/embeddings
    data/evals
    data/logs
    data/archive/v1
)
if [ $DRY_RUN -eq 0 ]; then
    for d in "${DIRS[@]}"; do
        mkdir -p "$d"
    done
    echo "  Created ${#DIRS[@]} directories under data/"
else
    echo "  Would create ${#DIRS[@]} directories under data/"
fi
echo ""

# ── Move data/ subdirectories to data/lean/ ───────────────────────────────
echo "=== HF datasets: data/* → data/lean/* ==="
migrate "data/workbook"       "data/lean/workbook"       "(Lean Workbook)"
migrate "data/goedel_proofs"  "data/lean/goedel_proofs"  "(Goedel proofs)"
migrate "data/lean_github"    "data/lean/lean_github"    "(LEAN-GitHub)"
migrate "data/numinamath"     "data/lean/numinamath"     "(NuminaMath)"
echo ""

# ── Move miniF2F JSONs to data/benchmarks/ ────────────────────────────────
echo "=== Benchmarks: data/*.json → data/benchmarks/ ==="
for f in data/minif2f_*.json data/theorem_index.json data/test_theorems.json; do
    [ -f "$f" ] && migrate "$f" "data/benchmarks/$(basename "$f")"
done
echo ""

# ── Move SFT data ────────────────────────────────────────────────────────
echo "=== SFT data: data/sft_*.jsonl → data/sft/ ==="
migrate "data/sft_train.jsonl" "data/sft/train.jsonl"
migrate "data/sft_val.jsonl"   "data/sft/val.jsonl"
# Legacy tactic_pairs (v1 formatted data)
if [ -d "data/tactic_pairs" ] && [ -n "$(ls -A data/tactic_pairs 2>/dev/null)" ]; then
    for f in data/tactic_pairs/*; do
        [ -f "$f" ] && migrate "$f" "data/sft/$(basename "$f")"
    done
    # Remove empty dir
    if [ $DRY_RUN -eq 0 ] && [ -d "data/tactic_pairs" ] && [ -z "$(ls -A data/tactic_pairs 2>/dev/null)" ]; then
        rmdir "data/tactic_pairs"
    fi
fi
echo ""

# ── Move model weights ───────────────────────────────────────────────────
echo "=== Models: models/* → data/models/ ==="
migrate "models/deepseek-prover-v2-7b" "data/models/base/deepseek-prover-v2-7b"
migrate "models/tinyllama-1.1b"        "data/models/base/tinyllama-1.1b"
# Merged LLM exports
if [ -d "models/llm" ]; then
    for d in models/llm/iter_*; do
        [ -d "$d" ] && migrate "$d" "data/models/merged/$(basename "$d")"
    done
fi
# Remove empty models/ tree
if [ $DRY_RUN -eq 0 ]; then
    find models/ -type d -empty -delete 2>/dev/null || true
    [ -d "models" ] && [ -z "$(ls -A models 2>/dev/null)" ] && rmdir models 2>/dev/null || true
fi
echo ""

# ── Move checkpoints ─────────────────────────────────────────────────────
echo "=== Checkpoints: checkpoints/* → data/checkpoints/ ==="
if [ -d "checkpoints/llm" ]; then
    for d in checkpoints/llm/iter_*; do
        [ -d "$d" ] && migrate "$d" "data/checkpoints/lora/$(basename "$d")"
    done
fi
if [ -d "checkpoints/ebm" ]; then
    for d in checkpoints/ebm/*; do
        [ -d "$d" ] && migrate "$d" "data/checkpoints/ebm/$(basename "$d")"
    done
fi
if [ $DRY_RUN -eq 0 ]; then
    find checkpoints/ -type d -empty -delete 2>/dev/null || true
    [ -d "checkpoints" ] && [ -z "$(ls -A checkpoints 2>/dev/null)" ] && rmdir checkpoints 2>/dev/null || true
fi
echo ""

# ── Move trajectories ────────────────────────────────────────────────────
echo "=== Trajectories: trajectories/* → data/trajectories/ ==="
if [ -d "trajectories" ]; then
    for f in trajectories/*.parquet; do
        [ -f "$f" ] && migrate "$f" "data/trajectories/$(basename "$f")"
    done
    if [ $DRY_RUN -eq 0 ]; then
        [ -d "trajectories" ] && [ -z "$(ls -A trajectories 2>/dev/null)" ] && rmdir trajectories 2>/dev/null || true
    fi
fi
echo ""

# ── Move baselines ───────────────────────────────────────────────────────
echo "=== Baselines: baselines/* → data/evals/baselines/ ==="
if [ -d "baselines" ]; then
    for f in baselines/*; do
        [ -e "$f" ] && migrate "$f" "data/evals/baselines/$(basename "$f")"
    done
    if [ $DRY_RUN -eq 0 ]; then
        [ -d "baselines" ] && [ -z "$(ls -A baselines 2>/dev/null)" ] && rmdir baselines 2>/dev/null || true
    fi
fi
echo ""

# ── Move eval_results ────────────────────────────────────────────────────
echo "=== Eval results: eval_results/* → data/evals/ ==="
if [ -d "eval_results" ]; then
    for f in eval_results/*; do
        [ -e "$f" ] && migrate "$f" "data/evals/$(basename "$f")"
    done
    if [ $DRY_RUN -eq 0 ]; then
        [ -d "eval_results" ] && [ -z "$(ls -A eval_results 2>/dev/null)" ] && rmdir eval_results 2>/dev/null || true
    fi
fi
echo ""

# ── Move iterations/ artifacts ────────────────────────────────────────────
echo "=== Iterations: iterations/iter_N/* → data/ subdirs ==="
for iter_dir in iterations/iter_*/; do
    [ -d "$iter_dir" ] || continue
    iter_name=$(basename "$iter_dir")
    migrate "${iter_dir}embeddings" "data/embeddings/${iter_name}"
    migrate "${iter_dir}baselines"  "data/evals/${iter_name}"
    migrate "${iter_dir}analysis"   "data/evals/${iter_name}/analysis"
    migrate "${iter_dir}model"      "data/models/merged/${iter_name}"
    migrate "${iter_dir}ebm"        "data/checkpoints/ebm/${iter_name}"
    migrate "${iter_dir}trajectories" "data/trajectories/${iter_name}"
    migrate "${iter_dir}tactic_pairs" "data/sft/${iter_name}_tactic_pairs"
done
if [ $DRY_RUN -eq 0 ]; then
    find iterations/ -type d -empty -delete 2>/dev/null || true
    [ -d "iterations" ] && [ -z "$(ls -A iterations 2>/dev/null)" ] && rmdir iterations 2>/dev/null || true
fi
echo ""

# ── Move logs ─────────────────────────────────────────────────────────────
echo "=== Logs: logs/* → data/logs/ ==="
if [ -d "logs" ]; then
    for f in logs/*; do
        [ -f "$f" ] && migrate "$f" "data/logs/$(basename "$f")"
    done
    if [ $DRY_RUN -eq 0 ]; then
        [ -d "logs" ] && [ -z "$(ls -A logs 2>/dev/null)" ] && rmdir logs 2>/dev/null || true
    fi
fi
echo ""

# ── Move archive ─────────────────────────────────────────────────────────
echo "=== Archive: archive/v1/* → data/archive/v1/ ==="
migrate "archive/v1" "data/archive/v1"
if [ $DRY_RUN -eq 0 ]; then
    [ -d "archive" ] && [ -z "$(ls -A archive 2>/dev/null)" ] && rmdir archive 2>/dev/null || true
fi
echo ""

# ── Move stray top-level files ────────────────────────────────────────────
echo "=== Stray files ==="
migrate "eval_clean_100.json" "data/evals/eval_clean_100.json"
echo ""

# ── Clean up empty directories ────────────────────────────────────────────
echo "=== Cleanup ==="
EMPTY_DIRS=(local_test_output plans)
for d in "${EMPTY_DIRS[@]}"; do
    if [ -d "$d" ]; then
        if [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
            if [ $DRY_RUN -eq 1 ]; then
                echo "  WOULD DELETE empty dir: ${d}/"
            else
                rmdir "$d"
                echo "  DELETED empty dir: ${d}/"
            fi
        else
            echo "  SKIP  ${d}/ not empty"
        fi
    fi
done
# Clean up empty data/traced/ .gitkeep placeholder
if [ -d "data/traced" ] && [ -f "data/traced/.gitkeep" ] && [ "$(ls -A data/traced)" = ".gitkeep" ]; then
    if [ $DRY_RUN -eq 0 ]; then
        rm "data/traced/.gitkeep"
    fi
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────
echo "================================================================"
if [ $DRY_RUN -eq 1 ]; then
    echo "  DRY RUN complete"
    echo "  Would move: ${MOVED} items, skipped: ${SKIPPED}"
    echo ""
    echo "  To apply: ./scripts/migrate_layout.sh --apply"
else
    echo "  Migration complete!"
    echo "  Moved: ${MOVED} items, skipped: ${SKIPPED}, errors: ${ERRORS}"
fi
echo "================================================================"

if [ $ERRORS -gt 0 ]; then
    exit 1
fi
