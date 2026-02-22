#!/usr/bin/env python3
"""Generate a Lean 4 file with sorry-proofs from a TheoremIndex JSON file.

This compiles theorem statements into .olean files so that Pantograph can
use `copyFrom(name)` instead of re-elaborating expressions from scratch.
Without this, complex expressions (involving ℂ, Finset, etc.) can take 60s+
to elaborate, exceeding the tactic timeout.

Usage:
    python generate_benchmark_lean.py \
        --input data/minif2f_v2s_test.json \
        --output vendor/Pantograph/BenchMinIF2FV2STest.lean \
        --module-name BenchMinIF2FV2STest

    # Then build:
    #   cd vendor/Pantograph && lake build BenchMinIF2FV2STest

Input format (TheoremIndex JSON):
    {
      "theorems": [
        {"name": "thm_name", "statement": "∀ (x : ℕ), x = x"},
        ...
      ]
    }

Output: A .lean file with `import Mathlib` and each theorem proved by `sorry`.
"""

import argparse
import json
import sys
from pathlib import Path


def fix_statement(statement: str) -> str:
    """Apply syntax fixes to make a statement valid in theorem-type position."""
    import re

    # Fix: Replace `∑/∏ VAR in S,` with `∑/∏ VAR ∈ S,` because in theorem-type
    # position, `in` can confuse the parser (conflicts with binder syntax).
    statement = re.sub(r"([∑∏]) (\w+) in ", r"\1 \2 ∈ ", statement)

    # Fix: `∃ x, ℤ,` → `∃ x : ℤ,` (malformed binder in v2s data)
    statement = re.sub(r"∃ (\w+(?:\s+\w+)*), (ℤ|ℝ|ℕ|ℂ|ℚ),", r"∃ \1 : \2,", statement)

    # Fix: `∃ x : ℤ,` → `∃ x,` — remove type annotations in ∃ binders
    # to avoid parser ambiguity. Lean infers these well.
    statement = re.sub(r"∃ (\w+) : (ℤ|ℝ|ℕ|ℂ|ℚ),", r"∃ \1,", statement)
    statement = re.sub(r"∃! (\w+) : (ℤ|ℝ|ℕ|ℂ|ℚ),", r"∃! \1,", statement)

    # Fix: In `∀ (args) (h : ...) :\n  conclusion`, the `:` conclusion separator
    # after the last `)` is valid in expression context (goal.start) but ambiguous
    # in `theorem name : TYPE` context. Replace with `,`.
    # Pattern: `)` followed by optional whitespace and `:` at end-of-binders position
    statement = re.sub(r"\)\s*:\s*\n", "),\n", statement)
    # Also handle single-line: `...) : conclusion`
    statement = re.sub(r"\)\s*:\s+(?=[A-Z∃∀¬↑⟨])", "), ", statement)

    return statement


def statement_to_theorem(name: str, statement: str) -> str:
    """Convert a ∀-expression statement to a theorem declaration with sorry proof."""
    statement = fix_statement(statement)
    return f"theorem {name} : {statement} := by sorry"


def generate_lean_file(theorems: list[dict], module_name: str) -> str:
    """Generate a complete Lean 4 file with sorry proofs for all theorems."""
    lines = [
        f"-- Auto-generated benchmark file: {module_name}",
        "-- Do not edit manually. Regenerate with:",
        "--   python python/data/generate_benchmark_lean.py",
        "",
        "import Mathlib",
        "",
        "set_option maxHeartbeats 400000",
        "",
        "open Real Nat",
        "",
    ]

    skipped = []
    for thm in theorems:
        name = thm["name"]
        statement = thm["statement"]

        # Skip theorems with obviously broken statements
        if not statement.strip():
            skipped.append((name, "empty statement"))
            continue

        # Skip theorems with set-builder notation {x : T | ...} — these don't
        # parse correctly in theorem-type position. They'll fall back to
        # goal.start(expr) at runtime with the longer timeout.
        if "{" in statement and "|" in statement:
            skipped.append((name, "set-builder notation"))
            continue

        # Skip theorems with `let` bindings — not valid in theorem type position
        if "let " in statement:
            skipped.append((name, "let binding"))
            continue

        # Write theorem
        lines.append(statement_to_theorem(name, statement))
        lines.append("")

    if skipped:
        lines.append(f"-- Skipped {len(skipped)} theorems:")
        for name, reason in skipped:
            lines.append(f"--   {name}: {reason}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Lean 4 sorry-file from TheoremIndex JSON"
    )
    parser.add_argument(
        "--input", required=True, help="Path to TheoremIndex JSON file"
    )
    parser.add_argument(
        "--output", required=True, help="Output .lean file path"
    )
    parser.add_argument(
        "--module-name",
        help="Lean module name (default: derived from output filename)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    theorems = data.get("theorems", [])
    if not theorems:
        print(f"ERROR: No theorems found in {input_path}", file=sys.stderr)
        sys.exit(1)

    module_name = args.module_name or output_path.stem

    lean_content = generate_lean_file(theorems, module_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(lean_content)

    print(f"Generated {output_path}: {len(theorems)} theorems, module {module_name}")


if __name__ == "__main__":
    main()
