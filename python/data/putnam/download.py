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

# PutnamBench .lean files use `open Set Filter MeasureTheory Metric ...` which
# allows bare identifiers like `Icc` instead of `Set.Icc`.  Pantograph's REPL
# has no `open` command, so goal.start(expr) fails on unqualified names.
# This table maps bare → fully-qualified for identifiers we've confirmed appear.
_QUALIFY_MAP: dict[str, str] = {
    # Set intervals (open Set)
    "Icc": "Set.Icc",
    "Ico": "Set.Ico",
    "Ioo": "Set.Ioo",
    "Ioc": "Set.Ioc",
    "Ici": "Set.Ici",
    "Ioi": "Set.Ioi",
    "Iic": "Set.Iic",
    "Iio": "Set.Iio",
    # Filter (open Filter)
    "Tendsto": "Filter.Tendsto",
    "atTop": "Filter.atTop",
    "atBot": "Filter.atBot",
    # MeasureTheory (open MeasureTheory)
    "volume": "MeasureTheory.volume",
    # Metric (open Metric)
    "ball": "Metric.ball",
    "closedBall": "Metric.closedBall",
}

# Build a single regex: match bare identifiers not preceded by a dot or word char.
_QUALIFY_PATTERN = re.compile(
    r"(?<!\w)(?<!\.)(" + "|".join(re.escape(k) for k in _QUALIFY_MAP) + r")(?!\w)"
)


def qualify_statement(stmt: str) -> str:
    """Replace bare Mathlib identifiers with fully-qualified names."""
    return _QUALIFY_PATTERN.sub(lambda m: _QUALIFY_MAP[m.group(1)], stmt)


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


def parse_putnam_file(lean_file: Path) -> tuple[list[dict], list[str]]:
    """Parse theorem and solution abbrev declarations from a PutnamBench .lean file.

    Each file typically contains one theorem `putnam_XXXX_XN` plus possibly
    an `abbrev putnam_XXXX_XN_solution` (answer placeholder with sorry).
    Solution abbrevs are extracted separately since theorems may reference them.

    Returns (theorems, solution_abbrevs) where:
      - theorems: list of {"name": ..., "statement": ...} dicts
      - solution_abbrevs: list of full abbrev declaration strings (with := sorry)
    """
    content = lean_file.read_text(encoding="utf-8", errors="replace")

    # Strip docstrings (/- ... -/)
    content_clean = re.sub(r'/\-.*?\-/', '', content, flags=re.DOTALL)
    # Strip line comments
    content_clean = re.sub(r'--.*$', '', content_clean, flags=re.MULTILINE)

    theorems = []
    solution_abbrevs = []

    # Extract solution abbrevs: `[noncomputable] abbrev name_solution : Type := sorry`
    abbrev_pattern = re.compile(
        r"((?:noncomputable\s+)?abbrev\s+[\w'.]+_solution\s*:.*?):=\s*sorry",
        re.DOTALL,
    )
    for match in abbrev_pattern.finditer(content_clean):
        decl = match.group(1).strip()
        # Normalize whitespace
        decl = re.sub(r'\s+', ' ', decl)
        solution_abbrevs.append(f"{decl} := sorry")

    # Extract theorems
    theorem_pattern = re.compile(
        r"theorem\s+([\w'.]+)\s*(.*?)\s*:=\s*(?:by\s+)?sorry",
        re.DOTALL,
    )
    for match in theorem_pattern.finditer(content_clean):
        name = match.group(1)
        signature = match.group(2).strip()

        statement = signature_to_expression(signature)
        statement = qualify_statement(statement)

        if statement:
            theorems.append({
                "name": name,
                "statement": statement,
            })

    return theorems, solution_abbrevs


def download_putnam(output_dir: Path) -> tuple[list[dict], list[str]]:
    """Download and extract PutnamBench theorems and solution abbrevs.

    Returns (theorems, preamble) where preamble contains solution abbrev declarations.
    """
    all_theorems = []
    all_abbrevs = []
    seen_names: set[str] = set()
    seen_abbrevs: set[str] = set()

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = clone_repo(PUTNAM_REPO, Path(tmpdir) / "PutnamBench")

        # PutnamBench stores .lean files in lean4/src/
        src_dir = repo_dir / "lean4" / "src"
        if not src_dir.exists():
            logger.warning("lean4/src/ not found, searching entire repo...")
            lean_files = sorted(repo_dir.rglob("*.lean"))
        else:
            lean_files = sorted(src_dir.rglob("*.lean"))

        logger.info("Found %d .lean files", len(lean_files))

        for lean_file in lean_files:
            if lean_file.name in ("lakefile.lean",):
                continue

            theorems, abbrevs = parse_putnam_file(lean_file)

            for abbrev in abbrevs:
                # Deduplicate by abbrev name
                abbrev_name = re.search(r'abbrev\s+([\w\'.]+)', abbrev)
                if abbrev_name and abbrev_name.group(1) not in seen_abbrevs:
                    seen_abbrevs.add(abbrev_name.group(1))
                    all_abbrevs.append(abbrev)

            for thm in theorems:
                if thm["name"] in seen_names:
                    continue
                seen_names.add(thm["name"])
                all_theorems.append(thm)

    logger.info(
        "Extracted %d PutnamBench theorems + %d solution abbrevs",
        len(all_theorems),
        len(all_abbrevs),
    )

    return all_theorems, all_abbrevs


def write_output(theorems: list, path: Path, preamble: list[str] | None = None):
    """Write theorems in TheoremIndex JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"theorems": theorems}
    if preamble:
        data["preamble"] = preamble
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
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

    theorems, preamble = download_putnam(output_dir)

    if theorems:
        write_output(theorems, output_path, preamble=preamble)
        print(f"\nPutnamBench download complete: {len(theorems)} theorems, {len(preamble)} solution abbrevs")
    else:
        logger.error("No PutnamBench theorems extracted")


if __name__ == "__main__":
    main()
