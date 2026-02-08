#!/usr/bin/env python3
"""Extract theorem statements and tactic-level training data from Mathlib4 using LeanDojo.

Usage:
    python python/data/trace_mathlib.py --output-dir data/
    python python/data/trace_mathlib.py --skip-trace --output-dir data/
    python python/data/trace_mathlib.py --fallback --output-dir data/
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Known-good Mathlib4 commit compatible with Lean v4.27.0
DEFAULT_MATHLIB_COMMIT = "v4.27.0"


def trace_mathlib(commit: str, cache_dir: str):
    """Trace Mathlib4 at the given commit using LeanDojo."""
    from lean_dojo import LeanGitRepo, trace

    repo = LeanGitRepo(
        "https://github.com/leanprover-community/mathlib4",
        commit,
    )
    logger.info("Tracing Mathlib4 at %s (this may take hours on first run)...", commit)
    traced_repo = trace(repo, dst_dir=cache_dir)
    logger.info("Trace complete. Cached at %s", cache_dir)
    return traced_repo


def load_cached_trace(cache_dir: str):
    """Load an existing traced repo from the LeanDojo cache."""
    from lean_dojo import LeanGitRepo, TracedRepo

    logger.info("Loading cached trace from %s", cache_dir)
    # LeanDojo stores traces in subdirectories keyed by repo+commit
    # TracedRepo.load expects the cache dir used during tracing
    traced_repo = TracedRepo.load_from_disk(cache_dir)
    return traced_repo


def extract_theorems(traced_repo) -> list:
    """Extract theorem names and statements from a traced repo.

    Returns list of dicts: {name, statement, file_path}
    """
    theorems = []
    num_skipped = 0

    for traced_file in traced_repo.traced_files:
        for thm in traced_file.traced_theorems:
            # Skip theorems without tactic proofs (term-mode, axioms, etc.)
            if not thm.has_tactic_proof:
                num_skipped += 1
                continue

            theorems.append({
                "name": thm.full_name,
                "statement": str(thm.statement),
                "file_path": str(traced_file.path),
            })

    logger.info(
        "Extracted %d theorems (%d skipped, no tactic proof)",
        len(theorems),
        num_skipped,
    )
    return theorems


def extract_tactic_pairs(traced_repo) -> list:
    """Extract (state, tactic) pairs from a traced repo.

    Returns list of dicts: {state, tactic, theorem, depth}
    """
    pairs = []
    errors = 0

    for traced_file in traced_repo.traced_files:
        for thm in traced_file.traced_theorems:
            if not thm.has_tactic_proof:
                continue

            try:
                for i, step in enumerate(thm.get_tactic_steps()):
                    state_before = str(step.state_before)
                    tactic = str(step.tactic)

                    if not state_before.strip() or not tactic.strip():
                        continue

                    pairs.append({
                        "state": state_before,
                        "tactic": tactic,
                        "theorem": thm.full_name,
                        "depth": i,
                    })
            except Exception as e:
                errors += 1
                if errors <= 10:
                    logger.warning("Error extracting tactics from %s: %s", thm.full_name, e)

    logger.info("Extracted %d tactic pairs (%d errors)", len(pairs), errors)
    return pairs


def extract_minif2f(traced_repo) -> tuple:
    """Extract miniF2F-test and miniF2F-valid theorem lists.

    Returns (test_theorems, valid_theorems) in TheoremIndex format.
    """
    test_theorems = []
    valid_theorems = []

    for traced_file in traced_repo.traced_files:
        path_str = str(traced_file.path)

        # miniF2F theorems are typically in Mathlib's test directory or
        # in a separate miniF2F import. Check path patterns.
        is_minif2f = "minif2f" in path_str.lower() or "MiniF2F" in path_str

        if not is_minif2f:
            continue

        for thm in traced_file.traced_theorems:
            entry = {
                "name": thm.full_name,
                "statement": str(thm.statement),
            }

            if "test" in path_str.lower():
                test_theorems.append(entry)
            else:
                valid_theorems.append(entry)

    logger.info(
        "miniF2F: %d test theorems, %d valid theorems",
        len(test_theorems),
        len(valid_theorems),
    )
    return test_theorems, valid_theorems


def download_fallback(output_dir: str):
    """Download pre-traced data as fallback when LeanDojo tracing is unavailable.

    Uses LeanDojo's pre-traced benchmark data from GitHub releases.
    """
    import subprocess
    import tempfile

    logger.info("Downloading pre-traced LeanDojo data (fallback mode)...")

    release_url = (
        "https://github.com/lean-dojo/LeanDojo/releases/download/"
        "v2.1.0/leandojo_benchmark_4.tar.gz"
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive = os.path.join(tmpdir, "benchmark.tar.gz")
        logger.info("Downloading %s ...", release_url)
        subprocess.run(
            ["curl", "-L", "-o", archive, release_url],
            check=True,
        )
        logger.info("Extracting...")
        subprocess.run(
            ["tar", "-xzf", archive, "-C", tmpdir],
            check=True,
        )

        # Convert extracted data to our format
        benchmark_dir = Path(tmpdir)
        convert_benchmark_to_our_format(benchmark_dir, output_path)

    logger.info("Fallback download complete.")


def convert_benchmark_to_our_format(benchmark_dir: Path, output_dir: Path):
    """Convert LeanDojo benchmark format to our TheoremIndex + tactic pairs format."""
    theorems = []
    tactic_pairs = []

    # LeanDojo benchmark has train/val/test splits with JSON files
    for split_file in sorted(benchmark_dir.rglob("*.json")):
        try:
            with open(split_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        if not isinstance(data, list):
            continue

        for entry in data:
            if not isinstance(entry, dict):
                continue

            if "full_name" in entry and "statement" in entry:
                theorems.append({
                    "name": entry["full_name"],
                    "statement": entry["statement"],
                })

            # Extract tactic pairs if available
            if "traced_tactics" in entry:
                for i, tactic_info in enumerate(entry["traced_tactics"]):
                    if "state_before" in tactic_info and "tactic" in tactic_info:
                        tactic_pairs.append({
                            "state": tactic_info["state_before"],
                            "tactic": tactic_info["tactic"],
                            "theorem": entry.get("full_name", "unknown"),
                            "depth": i,
                        })

    # Write outputs
    write_theorem_index(theorems, output_dir / "theorem_index.json")
    write_tactic_pairs(tactic_pairs, output_dir, val_fraction=0.05)

    logger.info(
        "Converted: %d theorems, %d tactic pairs", len(theorems), len(tactic_pairs)
    )


def write_theorem_index(theorems: list, path: Path):
    """Write theorem index in the format expected by prover-core.

    Format: {"theorems": [{"name": "...", "statement": "..."}]}
    """
    # Only keep name + statement (drop file_path if present)
    index_entries = [{"name": t["name"], "statement": t["statement"]} for t in theorems]
    index = {"theorems": index_entries}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %d theorems to %s", len(index_entries), path)


def write_tactic_pairs(pairs: list, output_dir: Path, val_fraction: float):
    """Write tactic pairs as JSONL, split into train/val."""
    random.shuffle(pairs)
    val_size = int(len(pairs) * val_fraction)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    pairs_dir = output_dir / "tactic_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train_pairs), ("val", val_pairs)]:
        path = pairs_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for pair in split_data:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info("Wrote %d pairs to %s", len(split_data), path)


def split_and_write(
    theorems: list,
    tactic_pairs: list,
    val_fraction: float,
    output_dir: str,
    minif2f_test: list | None = None,
    minif2f_valid: list | None = None,
):
    """Write all outputs to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Theorem index
    write_theorem_index(theorems, output_path / "theorem_index.json")

    # Tactic pairs (train/val split)
    write_tactic_pairs(tactic_pairs, output_path, val_fraction)

    # miniF2F splits
    if minif2f_test:
        minif2f_path = output_path / "minif2f_test.json"
        with open(minif2f_path, "w") as f:
            json.dump({"theorems": minif2f_test}, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %d miniF2F test theorems to %s", len(minif2f_test), minif2f_path)

    if minif2f_valid:
        minif2f_path = output_path / "minif2f_valid.json"
        with open(minif2f_path, "w") as f:
            json.dump({"theorems": minif2f_valid}, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %d miniF2F valid theorems to %s", len(minif2f_valid), minif2f_path)


def print_summary(theorems, tactic_pairs, minif2f_test, minif2f_valid, output_dir):
    """Print final summary statistics."""
    print("\n" + "=" * 60)
    print("  Mathlib Trace Summary")
    print("=" * 60)
    print(f"  Theorems:          {len(theorems):>8}")
    print(f"  Tactic pairs:      {len(tactic_pairs):>8}")
    print(f"  miniF2F test:      {len(minif2f_test):>8}")
    print(f"  miniF2F valid:     {len(minif2f_valid):>8}")
    print(f"  Output directory:  {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract theorem statements and tactic training data from Mathlib4.",
    )
    parser.add_argument(
        "--mathlib-commit",
        default=DEFAULT_MATHLIB_COMMIT,
        help="Mathlib4 commit or tag (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.expanduser("~/.cache/leandojo"),
        help="LeanDojo cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Fraction of tactic pairs for validation (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-trace",
        action="store_true",
        help="Load existing cached trace instead of re-tracing",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Skip LeanDojo tracing; download pre-traced data from releases",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: %(default)s)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Fallback mode: download pre-traced data
    if args.fallback:
        download_fallback(args.output_dir)
        return

    # Normal mode: trace with LeanDojo
    try:
        import lean_dojo  # noqa: F401
    except ImportError:
        logger.error(
            "lean_dojo not installed. Install with: pip install lean-dojo>=2.1.0\n"
            "Or use --fallback to download pre-traced data."
        )
        sys.exit(1)

    if args.skip_trace:
        traced_repo = load_cached_trace(args.cache_dir)
    else:
        traced_repo = trace_mathlib(args.mathlib_commit, args.cache_dir)

    # Extract data
    theorems = extract_theorems(traced_repo)
    tactic_pairs = extract_tactic_pairs(traced_repo)
    minif2f_test, minif2f_valid = extract_minif2f(traced_repo)

    # Write outputs
    split_and_write(
        theorems,
        tactic_pairs,
        args.val_fraction,
        args.output_dir,
        minif2f_test,
        minif2f_valid,
    )

    print_summary(theorems, tactic_pairs, minif2f_test, minif2f_valid, args.output_dir)


if __name__ == "__main__":
    main()
