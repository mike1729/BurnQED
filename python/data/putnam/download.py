#!/usr/bin/env python3
"""Download PutnamBench Lean 4 theorems and convert to BurnQED TheoremIndex format.

PutnamBench contains 672 Lean 4 competition math problems (Putnam exam, 1962–2025)
targeting Lean 4.27.0 / Mathlib v4.27.0 — identical to our Pantograph version.

Clones trishullab/PutnamBench (shallow), parses each .lean file in lean4/src/,
extracts theorem statements, and writes TheoremIndex JSON.

Usage:
    python python/data/putnam/download.py --output-dir data/benchmarks
    python python/data/putnam/download.py --output-dir data/benchmarks --force
"""

import argparse
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

# Reuse signature_to_expression from the miniF2F download script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data.minif2f.download import signature_to_expression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PUTNAM_REPO = "https://github.com/trishullab/PutnamBench.git"


def clone_repo(url: str, dest: Path) -> Path:
    """Shallow-clone a git repository."""
    logger.info("Cloning %s to %s...", url, dest)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        logger.error("git clone failed: %s", stderr)
        raise
    return dest


def parse_putnam_file(lean_file: Path) -> list[dict]:
    """Parse theorem declarations from a PutnamBench .lean file.

    Each file typically contains one theorem `putnam_XXXX_XN` plus possibly
    an `abbrev putnam_XXXX_XN_solution` (answer placeholder with sorry).
    We skip solution abbrevs and extract only the theorem statement.

    Returns list of {"name": ..., "statement": ...} dicts.
    """
    content = lean_file.read_text(encoding="utf-8", errors="replace")
    theorems = []

    # Strip docstrings (/-  ... -/)
    content_clean = re.sub(r'/\-.*?\-/', '', content, flags=re.DOTALL)

    # Match `theorem <name>` followed by everything up to `:= by` or `:= sorry`
    pattern = re.compile(
        r"theorem\s+([\w'.]+)\s*(.*?)\s*:=\s*(?:by\s+)?sorry",
        re.DOTALL,
    )

    for match in pattern.finditer(content_clean):
        name = match.group(1)
        signature = match.group(2).strip()

        statement = signature_to_expression(signature)

        if statement:
            theorems.append({
                "name": name,
                "statement": statement,
            })

    return theorems


def download_putnam(output_dir: Path) -> list[dict]:
    """Download and extract PutnamBench theorems.

    Returns list of theorem dicts.
    """
    all_theorems = []
    seen_names: set[str] = set()
    skipped_solutions = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = clone_repo(PUTNAM_REPO, Path(tmpdir) / "PutnamBench")

        # PutnamBench stores .lean files in lean4/src/
        src_dir = repo_dir / "lean4" / "src"
        if not src_dir.exists():
            # Fallback: search for .lean files anywhere
            logger.warning("lean4/src/ not found, searching entire repo...")
            lean_files = sorted(repo_dir.rglob("*.lean"))
        else:
            lean_files = sorted(src_dir.rglob("*.lean"))

        logger.info("Found %d .lean files", len(lean_files))

        for lean_file in lean_files:
            if lean_file.name in ("lakefile.lean",):
                continue

            content = lean_file.read_text(encoding="utf-8", errors="replace")

            # Skip solution abbrevs — files named *_solution* or containing
            # only `abbrev ... := sorry` (answer placeholders)
            if "_solution" in lean_file.stem.lower():
                skipped_solutions += 1
                continue

            theorems = parse_putnam_file(lean_file)
            for thm in theorems:
                if thm["name"] in seen_names:
                    continue
                # Skip solution abbrev names that snuck through
                if "_solution" in thm["name"].lower():
                    skipped_solutions += 1
                    continue
                seen_names.add(thm["name"])
                all_theorems.append(thm)

    logger.info(
        "Extracted %d PutnamBench theorems (skipped %d solution abbrevs)",
        len(all_theorems),
        skipped_solutions,
    )

    return all_theorems


def write_output(theorems: list, path: Path):
    """Write theorems in TheoremIndex JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"theorems": theorems}, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d theorems to %s", len(theorems), path)


def main():
    parser = argparse.ArgumentParser(description="Download PutnamBench for BurnQED")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks",
        help="Output directory (default: data/benchmarks)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output file exists",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_path = output_dir / "putnam.json"

    if not args.force and output_path.exists():
        logger.info("PutnamBench file already exists at %s (use --force to re-download)", output_path)
        return

    theorems = download_putnam(output_dir)

    if theorems:
        write_output(theorems, output_path)
        print(f"\nPutnamBench download complete: {len(theorems)} theorems")
    else:
        logger.error("No PutnamBench theorems extracted")


if __name__ == "__main__":
    main()
