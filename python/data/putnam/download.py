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
    # Set (open Set)
    "univ": "Set.univ",
    # Filter (open Filter)
    "Tendsto": "Filter.Tendsto",
    "atTop": "Filter.atTop",
    "atBot": "Filter.atBot",
    # MeasureTheory (open MeasureTheory)
    "volume": "MeasureTheory.volume",
    "IntegrableOn": "MeasureTheory.IntegrableOn",
    # Metric (open Metric)
    "ball": "Metric.ball",
    "closedBall": "Metric.closedBall",
    # Complex (can't open Complex globally — clashes with Real)
    "cexp": "Complex.exp",
    # Real analysis functions (open Real / open Complex — default to Real;
    # the few Complex cases will get a type error instead of "Unknown identifier",
    # which is no worse).  Note: bare `I` (Complex.I) is NOT included because
    # PutnamBench also uses `I` as a variable name in several theorems.
    "exp": "Real.exp",
    "cos": "Real.cos",
    "sin": "Real.sin",
    "log": "Real.log",
    "sqrt": "Real.sqrt",
    # Function (open Function)
    "Injective": "Function.Injective",
    # Note: `coeff` removed from global map — handled context-aware below
    # (Polynomial.coeff vs MvPolynomial.coeff).
}

# Build a single regex: match bare identifiers not preceded by a dot or word char.
_QUALIFY_PATTERN = re.compile(
    r"(?<!\w)(?<!\.)(" + "|".join(re.escape(k) for k in _QUALIFY_MAP) + r")(?!\w)"
)


def qualify_statement(stmt: str) -> str:
    """Replace bare Mathlib identifiers with fully-qualified names."""
    # Topology notation: 𝓝 → nhds (bare function name, no open needed)
    stmt = stmt.replace("𝓝", "nhds")

    # nhds bracket notation: nhds[≠] x → nhdsWithin x ({x}ᶜ) etc.
    # Uses _fix_nhds_brackets from generate_lean.py which handles both simple
    # and parenthesized arguments like `nhds[>] (ts k.1)`.
    from data.generate_lean import _fix_nhds_brackets
    stmt = _fix_nhds_brackets(stmt)

    # Apply the main qualification map
    stmt = _QUALIFY_PATTERN.sub(lambda m: _QUALIFY_MAP[m.group(1)], stmt)

    # Context-aware Polynomial/MvPolynomial qualification: bare X, C, coeff, eval
    # are ambiguous globally, but unambiguous when the statement mentions a specific type.
    # Check MvPolynomial FIRST since "MvPolynomial" also contains "Polynomial".
    if "MvPolynomial" in stmt:
        # Qualify X used as function: `X c`, `X 0`, `X (expr)`
        stmt = re.sub(r"(?<!\w)(?<!\.)X(?=\s+[\w\d(])", "MvPolynomial.X", stmt)
        # Qualify C used as function: `C a`, `C (expr)`
        stmt = re.sub(r"(?<!\w)(?<!\.)C(?=\s+[\w(])", "MvPolynomial.C", stmt)
        stmt = re.sub(r"(?<!\w)(?<!\.)eval(?=\s*[\(!\[])", "MvPolynomial.eval", stmt)
        stmt = re.sub(r"(?<!\w)(?<!\.)aeval\b", "MvPolynomial.aeval", stmt)
        stmt = re.sub(r"(?<!\w)(?<!\.)coeff(?!\w)", "MvPolynomial.coeff", stmt)
    elif "Polynomial" in stmt:
        # Don't qualify X in binder position: `(X :` means X is a variable name
        stmt = re.sub(r"(?<!\w)(?<!\.)(?<!\()X(?!\w)(?!\s*:)", "Polynomial.X", stmt)
        stmt = re.sub(r"(?<!\w)(?<!\.)C(?=\s+[\w(])", "Polynomial.C", stmt)
        stmt = re.sub(r"(?<!\w)(?<!\.)coeff(?!\w)", "Polynomial.coeff", stmt)

    # Complex analysis context: when statement mentions ℂ and has bare I in
    # multiplicative context (I *, * I), qualify I → Complex.I and switch
    # Real.exp/sin/cos → Complex.exp/sin/cos for expressions involving Complex.I.
    if "ℂ" in stmt and re.search(r"(?<!\w)I\s*\*|\*\s*I(?!\w)", stmt):
        stmt = re.sub(r"(?<!\w)(?<!\.)I(?!\w)(?!\s*:)", "Complex.I", stmt)
        # Fix exp/sin/cos applied to complex arguments
        stmt = stmt.replace("Real.exp (Complex.I", "Complex.exp (Complex.I")
        stmt = stmt.replace("Real.sin (Complex.I", "Complex.sin (Complex.I")
        stmt = stmt.replace("Real.cos (Complex.I", "Complex.cos (Complex.I")

    return stmt


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

    # Extract custom helper definitions: `[noncomputable] def name ...` that are NOT
    # solution abbrevs. These define problem-specific functions (dist_to_int, tetration,
    # etc.) that theorems reference. We include them in the preamble so theorems compile.
    def_pattern = re.compile(
        r"((?:noncomputable\s+)?def\s+[\w'.]+\b[^:=]*(?::(?!=)[^:=]*)?):=(.*?)(?=\n(?:theorem|noncomputable\s+def|def\s|abbrev|end\b|#)\s|\Z)",
        re.DOTALL,
    )
    for match in def_pattern.finditer(content_clean):
        full_sig = match.group(1).strip()
        body = match.group(2).strip()
        # Skip solution defs (already extracted as abbrevs)
        if "_solution" in full_sig:
            continue
        # Normalize whitespace in signature
        full_sig = re.sub(r'\s+', ' ', full_sig)
        body = re.sub(r'\s+', ' ', body)
        solution_abbrevs.append(f"{full_sig} := {body}")

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
                # Deduplicate by declaration name (abbrev or def)
                abbrev_name = re.search(r'(?:abbrev|def)\s+([\w\'.]+)', abbrev)
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
