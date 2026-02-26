#!/usr/bin/env python3
"""Download miniF2F Lean 4 theorems and convert to BurnQED TheoremIndex format.

Supports both v1 (yangky11/miniF2F-lean4, .lean parsing) and v2
(roozbeh-yz/miniF2F_v2, JSON datasets with v2s/v2c variants).

Usage:
    python python/data/download_minif2f.py --output-dir data/benchmarks       # all versions
    python python/data/download_minif2f.py --output-dir data/benchmarks --version v1    # v1 only
    python python/data/download_minif2f.py --output-dir data/benchmarks --version v2    # v2 only
"""

import argparse
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MINIF2F_REPO = "https://github.com/yangky11/miniF2F-lean4.git"
MINIF2F_V2_REPO = "https://github.com/roozbeh-yz/miniF2F_v2.git"

# v2 variant → JSON filename in the repo's datasets/ directory
V2_VARIANTS = {
    "v2s": "miniF2F_v2s.json",
    "v2c": "miniF2F_v2c.json",
}


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
        repo_dir = clone_repo(MINIF2F_REPO, Path(tmpdir) / "miniF2F-lean4")

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


def download_minif2f_v2(output_dir: Path) -> dict[str, dict[str, list]]:
    """Download and extract miniF2F v2 theorems (v2s and v2c variants).

    Clones roozbeh-yz/miniF2F_v2, reads JSON datasets, and converts each entry
    to TheoremIndex format using signature_to_expression().

    Returns dict mapping variant → split → theorem list, e.g.:
        {"v2s": {"valid": [...], "test": [...]}, "v2c": {"valid": [...]}}
    """
    results: dict[str, dict[str, list]] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = clone_repo(MINIF2F_V2_REPO, Path(tmpdir) / "miniF2F_v2")

        datasets_dir = repo_dir / "datasets"
        if not datasets_dir.exists():
            logger.error("datasets/ directory not found in miniF2F_v2 repo")
            return results

        for variant, json_filename in V2_VARIANTS.items():
            json_path = datasets_dir / json_filename
            if not json_path.exists():
                logger.warning("v2 dataset file not found: %s", json_path)
                continue

            with open(json_path) as f:
                entries = json.load(f)

            if not isinstance(entries, list):
                logger.warning("Expected list in %s, got %s", json_filename, type(entries).__name__)
                continue

            # Group entries by split
            by_split: dict[str, list] = {}
            seen_names: set[str] = set()

            for entry in entries:
                name = entry.get("name", "").strip()
                split = entry.get("split", "valid").strip()
                formal = entry.get("formal_statement", "").strip()

                if not name or not formal:
                    continue
                if name in seen_names:
                    continue
                seen_names.add(name)

                # Parse formal_statement: strip "theorem <name>" prefix and ":= by" suffix
                # formal_statement looks like: "theorem name (args) : goal := by"
                signature = _extract_signature_from_formal(formal)
                if not signature:
                    logger.debug("Could not parse formal_statement for %s", name)
                    continue

                statement = signature_to_expression(signature)
                if not statement:
                    logger.debug("signature_to_expression returned empty for %s", name)
                    continue

                by_split.setdefault(split, []).append({
                    "name": name,
                    "statement": statement,
                })

            results[variant] = by_split

            for split, theorems in by_split.items():
                logger.info(
                    "v2 %s %s: %d theorems", variant, split, len(theorems)
                )

    return results


def _extract_signature_from_formal(formal: str) -> str:
    """Extract the type signature from a v2 formal_statement.

    Input:  "theorem mathd_algebra_478 (b h v u : ℝ) ... : v = 65 * u := by"
    Output: "(b h v u : ℝ) ... : v = 65 * u"

    Strips the leading "theorem <name>" and trailing ":= by" (or ":= by\n  sorry").
    """
    # Strip trailing ":= by sorry", ":= by", ":= sorry", etc.
    text = re.sub(r'\s*:=\s*(?:by\s+)?(?:sorry\s*)?$', '', formal, flags=re.DOTALL).strip()

    # Strip leading "theorem <name>" (name can contain . and ')
    text = re.sub(r'^theorem\s+[\w\'.]+\s*', '', text).strip()

    return text


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
        default="data/benchmarks",
        help="Output directory (default: data/benchmarks)",
    )
    parser.add_argument(
        "--version",
        choices=["v1", "v2", "all"],
        default="all",
        help="Which miniF2F version(s) to download (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output files exist",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    total = 0

    # ── v1: yangky11/miniF2F-lean4 ──
    if args.version in ("v1", "all"):
        test_path = output_dir / "minif2f_test.json"
        valid_path = output_dir / "minif2f_valid.json"

        if not args.force and test_path.exists() and valid_path.exists():
            logger.info("v1 miniF2F files already exist (use --force to re-download)")
        else:
            test_theorems, valid_theorems = download_minif2f(output_dir)

            if test_theorems:
                write_output(test_theorems, test_path)
            else:
                logger.warning("No v1 test theorems extracted")

            if valid_theorems:
                write_output(valid_theorems, valid_path)
            else:
                logger.warning("No v1 valid theorems extracted")

            total += len(test_theorems) + len(valid_theorems)

    # ── v2: roozbeh-yz/miniF2F_v2 (v2s + v2c) ──
    if args.version in ("v2", "all"):
        # Check if v2 valid files already exist. Only check valid splits since
        # test splits may not exist in the upstream repo (all entries are "valid").
        v2_valid_files = [
            output_dir / f"minif2f_{variant}_valid.json"
            for variant in V2_VARIANTS
        ]
        if not args.force and all(p.exists() for p in v2_valid_files):
            logger.info("v2 miniF2F files already exist (use --force to re-download)")
        else:
            v2_results = download_minif2f_v2(output_dir)

            for variant, by_split in v2_results.items():
                for split, theorems in by_split.items():
                    if theorems:
                        out_path = output_dir / f"minif2f_{variant}_{split}.json"
                        write_output(theorems, out_path)
                        total += len(theorems)

            if not v2_results:
                logger.error("Failed to extract any v2 miniF2F theorems")

    if total > 0:
        print(f"\nminiF2F download complete: {total} total theorems")


if __name__ == "__main__":
    main()
