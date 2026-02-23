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

    # Fix: `𝓝` notation requires `open Topology`; use `nhds` instead.
    statement = statement.replace("𝓝", "nhds")

    # Fix: Strip inline Lean comments `-- ...` that appear in some v1 statements.
    # Comments run to end-of-line, but may be flattened onto one line.
    # Match `--` through to the next `(h` (start of next hypothesis binder).
    statement = re.sub(r" --.*?(?=\(h)", " ", statement)

    # Fix: Malformed `∀ : ∃ f, T → U,` → `∃ f : T → U,` (v1 data bug)
    statement = re.sub(r"^∀ : (∃ \w+), ([\w→ ]+),", r"\1 : \2,", statement)

    # Fix: Integer division in exponents `^ (1 / 3)` → `^ ((1 : ℝ) / 3)` to
    # avoid ℕ division truncation (1/3 = 0 in ℕ).
    statement = re.sub(r"\^ \((\d+) / (\d+)\)", r"^ ((\1 : ℝ) / \2)", statement)

    # Fix: Replace `∑/∏ VAR in S,` with `∑/∏ VAR ∈ S,` — `in` notation was
    # deprecated in Mathlib 4 in favor of `∈`.
    statement = re.sub(r"([∑∏]) (\w+) in ", r"\1 \2 ∈ ", statement)

    # Fix: `∃ x, ℤ,` → `∃ x : ℤ,` (malformed binder in v2s data)
    statement = re.sub(r"∃ (\w+(?:\s+\w+)*), (ℤ|ℝ|ℕ|ℂ|ℚ),", r"∃ \1 : \2,", statement)

    # Fix: In `∀ (args) (h : ...) :\n  conclusion`, the `:` conclusion separator
    # after the last `)` is valid in expression context (goal.start) but ambiguous
    # in `theorem name : TYPE` context. Replace with `,`.
    # Pattern: `)` followed by optional whitespace and `:` at end-of-binders position
    statement = re.sub(r"\)\s*:\s*\n", "),\n", statement)
    # Also handle single-line: `...) : conclusion`
    statement = re.sub(r"\)\s*:\s+(?=[A-Z∃∀¬↑⟨])", "), ", statement)

    # Fix: Qualify bare Nat/Int names when `open Nat` is not used.
    # These are common in IMO-Steps data. Negative lookbehind avoids
    # double-qualifying already-qualified names like Nat.succ.
    statement = re.sub(r"(?<!\w\.)(?<!\w)succ\b", "Nat.succ", statement)
    statement = re.sub(r"(?<!\w\.)(?<!\w)natAbs\b", "Int.natAbs", statement)
    statement = re.sub(r"(?<!\w\.)(?<!\w)choose\b", "Nat.choose", statement)

    # Fix: Factorial notation `n!` or `n !` is not valid in Lean 4 theorem-type
    # position. Replace with `(Nat.factorial n)`. Parentheses needed so the result
    # works as an argument, e.g. `Nat.gcd (Nat.factorial 20) 200000`.
    statement = re.sub(r"\b(\d+)\s*!", r"(Nat.factorial \1)", statement)
    statement = re.sub(r"\b([a-z_]\w*)\s*!", r"(Nat.factorial \1)", statement)
    # Also handle parenthesized expressions like `(n)!` → `(Nat.factorial n)`
    statement = re.sub(r"\(([a-z_]\w*)\)\s*!", r"(Nat.factorial \1)", statement)

    # Fix: Set-builder notation `{x : T | P}` in theorem-type position confuses
    # the Lean parser (conflicts with implicit binder `{x : T}`). Wrapping in
    # parentheses `({x : T | P})` disambiguates. Handles nested braces.
    statement = _parenthesize_set_builders(statement)

    return statement


def _parenthesize_set_builders(s: str) -> str:
    """Wrap top-level set-builder `{...|...}` in parens, handling nesting."""
    result = []
    i = 0
    while i < len(s):
        if s[i] == '{' and (i == 0 or s[i-1] != '('):
            # Find the matching closing brace (handling nesting)
            depth = 1
            j = i + 1
            has_pipe = False
            while j < len(s) and depth > 0:
                if s[j] == '{':
                    depth += 1
                elif s[j] == '}':
                    depth -= 1
                elif s[j] == '|' and depth == 1:
                    has_pipe = True
                j += 1
            if depth == 0 and has_pipe:
                # Wrap {…|…} in parens
                result.append('(')
                result.append(s[i:j])
                result.append(')')
                i = j
                continue
        result.append(s[i])
        i += 1
    return ''.join(result)


def statement_to_theorem(name: str, statement: str) -> str:
    """Convert a ∀-expression statement to a theorem declaration with sorry proof."""
    statement = fix_statement(statement)
    return f"theorem {name} : {statement} := by sorry"


def generate_lean_file(
    theorems: list[dict], module_name: str, skip_names: set[str] | None = None
) -> str:
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
        "open Real",
        "",
    ]

    skipped = []
    for thm in theorems:
        name = thm["name"]
        statement = thm["statement"]

        # Skip theorems explicitly listed (e.g., Mathlib version incompatibilities)
        if skip_names and name in skip_names:
            skipped.append((name, "in skip list"))
            continue

        # Skip theorems with obviously broken statements
        if not statement.strip():
            skipped.append((name, "empty statement"))
            continue

        # Skip theorems with `let` bindings — not valid in theorem type position
        if "let " in statement:
            skipped.append((name, "let binding"))
            continue

        # Skip theorems using APIs that changed between Mathlib versions
        # (e.g., NNReal.IsConjExponent was renamed/removed in v4.26.0)
        if "NNReal.IsConjExponent" in statement:
            skipped.append((name, "NNReal.IsConjExponent (Mathlib version mismatch)"))
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
    parser.add_argument(
        "--skip-file",
        help="File with theorem names to skip (one per line)",
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

    skip_names = set()
    if args.skip_file:
        skip_path = Path(args.skip_file)
        if skip_path.exists():
            skip_names = {line.strip() for line in skip_path.read_text().splitlines() if line.strip()}
            print(f"Loaded {len(skip_names)} skip names from {skip_path}")

    lean_content = generate_lean_file(theorems, module_name, skip_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(lean_content)

    print(f"Generated {output_path}: {len(theorems)} theorems, module {module_name}")


if __name__ == "__main__":
    main()
