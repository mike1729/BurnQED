#!/usr/bin/env python3
"""Download miniF2F Lean 4 theorems and convert to BurnQED TheoremIndex format.

Clones the miniF2F-lean4 repository, parses .lean files to extract theorem
names and statements, and writes minif2f_test.json / minif2f_valid.json.

Usage:
    python python/data/download_minif2f.py --output-dir data/
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MINIF2F_REPO = "https://github.com/yangky11/miniF2F-lean4.git"


def clone_minif2f(dest: Path) -> Path:
    """Shallow-clone the miniF2F-lean4 repository."""
    logger.info("Cloning miniF2F-lean4 to %s...", dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", MINIF2F_REPO, str(dest)],
        check=True,
        capture_output=True,
    )
    return dest


def parse_lean_theorems(lean_file: Path) -> list[dict]:
    """Parse theorem declarations from a Lean 4 file.

    Extracts theorem name and statement (type signature) from patterns like:
        theorem name (args) : statement := by sorry
        theorem name : statement := by sorry

    Returns list of {"name": ..., "statement": ...} dicts.
    """
    content = lean_file.read_text(encoding="utf-8", errors="replace")
    theorems = []

    # Match `theorem <name>` followed by everything up to `:= by` or `:= sorry`
    # The statement is between the last `:` before `:=` and `:=`
    # This regex handles multi-line theorems
    pattern = re.compile(
        r"theorem\s+([\w'.]+)\s*(.*?)\s*:=\s*(?:by\s+)?sorry",
        re.DOTALL,
    )

    for match in pattern.finditer(content):
        name = match.group(1)
        signature = match.group(2).strip()

        # Convert signature to a Lean expression: params become ∀, goal stays as-is
        # "(x : T) (h : P) : goal" → "∀ (x : T) (h : P), goal"
        statement = signature_to_expression(signature)

        if statement:
            theorems.append({
                "name": name,
                "statement": statement,
            })

    return theorems


def signature_to_expression(signature: str) -> str:
    """Convert a Lean theorem signature to an expression suitable for goal.start.

    Given "(x : Nat) (h : x > 0) : x ≥ 1",
    returns "∀ (x : Nat) (h : x > 0), x ≥ 1".

    This is a valid Lean type expression that Pantograph's goal.start can accept.
    """
    # Find the last top-level ":" that separates params from the return type
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    last_colon = -1

    for i, ch in enumerate(signature):
        if ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren -= 1
        elif ch == '[':
            depth_bracket += 1
        elif ch == ']':
            depth_bracket -= 1
        elif ch == '{':
            depth_brace += 1
        elif ch == '}':
            depth_brace -= 1
        elif ch == ':' and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            last_colon = i

    if last_colon == -1:
        return ""

    params = signature[:last_colon].strip()
    goal = signature[last_colon + 1:].strip()

    # Clean up whitespace (multi-line signatures)
    params = re.sub(r'\s+', ' ', params)
    goal = re.sub(r'\s+', ' ', goal)

    if params:
        return f"∀ {params}, {goal}"
    else:
        return goal


def classify_file(path: Path) -> str | None:
    """Classify a .lean file as 'test', 'valid', or None."""
    name = path.stem.lower()
    path_str = str(path).lower()

    if "test" in name or "/test/" in path_str or "\\test\\" in path_str:
        return "test"
    elif "valid" in name or "/valid/" in path_str or "\\valid\\" in path_str:
        return "valid"
    return None


def download_minif2f(output_dir: Path) -> tuple[list, list]:
    """Download and extract miniF2F theorems.

    Returns (test_theorems, valid_theorems).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = clone_minif2f(Path(tmpdir) / "miniF2F-lean4")

        test_theorems = []
        valid_theorems = []
        seen_names: set[str] = set()

        # Find all .lean files under MiniF2F/
        lean_dir = repo_dir / "MiniF2F"
        if not lean_dir.exists():
            # Try alternate locations
            for candidate in repo_dir.rglob("*.lean"):
                if candidate.name in ("lakefile.lean",):
                    continue
                lean_dir = candidate.parent
                break

        lean_files = sorted(repo_dir.rglob("*.lean"))

        for lean_file in lean_files:
            if lean_file.name == "lakefile.lean":
                continue

            split = classify_file(lean_file)
            if split is None:
                # Try parent directory
                split = classify_file(lean_file.parent)

            theorems = parse_lean_theorems(lean_file)
            for thm in theorems:
                if thm["name"] in seen_names:
                    continue
                seen_names.add(thm["name"])

                if split == "test":
                    test_theorems.append(thm)
                elif split == "valid":
                    valid_theorems.append(thm)
                else:
                    # Default to test if can't classify
                    test_theorems.append(thm)

        logger.info(
            "Extracted %d test + %d valid miniF2F theorems",
            len(test_theorems),
            len(valid_theorems),
        )

    return test_theorems, valid_theorems


def write_output(theorems: list, path: Path):
    """Write theorems in TheoremIndex JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"theorems": theorems}, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d theorems to %s", len(theorems), path)


def main():
    parser = argparse.ArgumentParser(description="Download miniF2F for BurnQED")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (e.g., data/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output files exist",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    test_path = output_dir / "minif2f_test.json"
    valid_path = output_dir / "minif2f_valid.json"

    if not args.force and test_path.exists() and valid_path.exists():
        logger.info("miniF2F files already exist (use --force to re-download)")
        return

    test_theorems, valid_theorems = download_minif2f(output_dir)

    if test_theorems:
        write_output(test_theorems, test_path)
    else:
        logger.warning("No test theorems extracted")

    if valid_theorems:
        write_output(valid_theorems, valid_path)
    else:
        logger.warning("No valid theorems extracted")

    total = len(test_theorems) + len(valid_theorems)
    if total == 0:
        logger.error("Failed to extract any miniF2F theorems")
        return

    print(f"\nminiF2F download complete: {len(test_theorems)} test, {len(valid_theorems)} valid")


if __name__ == "__main__":
    main()
