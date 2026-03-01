#!/usr/bin/env python3
"""Generate a Lean 4 file with sorry-proofs from a TheoremIndex JSON file.

This compiles theorem statements into .olean files so that Pantograph can
use `copyFrom(name)` instead of re-elaborating expressions from scratch.
Without this, complex expressions (involving ℂ, Finset, etc.) can take 60s+
to elaborate, exceeding the tactic timeout.

Usage:
    python python/data/minif2f/generate_lean.py \
        --input data/benchmarks/minif2f_v2s_test.json \
        --output vendor/Pantograph/BenchMinIF2FV2sTest.lean \
        --module-name BenchMinIF2FV2sTest

    # Then compile:
    #   cd vendor/Pantograph && lake env lean BenchMinIF2FV2sTest.lean

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
    # Handle `(expr)!` first (before simple patterns eat inner variables).
    statement = _fix_paren_factorials(statement)
    statement = re.sub(r"\b(\d+)\s*!", r"(Nat.factorial \1)", statement)
    statement = re.sub(r"\b([a-z_]\w*)\s*!", r"(Nat.factorial \1)", statement)

    # Fix: `{x, ℝ | P}` → `{x : ℝ | P}` (malformed set-builder type annotation in v2c)
    statement = re.sub(r"\{(\w+), (ℤ|ℝ|ℕ|ℂ|ℚ) \|", r"{\1 : \2 |", statement)

    # Fix: Set-builder notation `{x : T | P}` in theorem-type position confuses
    # the Lean parser (conflicts with implicit binder `{x : T}`). Wrapping in
    # parentheses `({x : T | P})` disambiguates. Handles nested braces.
    statement = _parenthesize_set_builders(statement)

    # Fix: `![a, b]` doesn't coerce to `EuclideanSpace ℝ (Fin n)` in Mathlib
    # v4.27.0+. Wrap with `(EuclideanSpace.equiv _ ℝ).symm ![...]`.
    if "EuclideanSpace" in statement:
        statement = re.sub(
            r"= !\[",
            r"= (EuclideanSpace.equiv _ ℝ).symm ![",
            statement,
        )

    # Fix: `let c, =` typo in v2s data → `let c :=`
    statement = statement.replace("let c, =", "let c :=")

    # Fix: `let` bindings flattened onto one line need `;` separators.
    # "let a := expr let b := expr" → "let a := expr; let b := expr"
    # Only insert `;` when preceded by a non-comma token (avoid breaking ∀ ..., let ...)
    statement = re.sub(r'(?<=[^\s,])\s+(let\s+\w+\s*:=)', r'; \1', statement)

    return statement


def _fix_paren_factorials(s: str) -> str:
    """Replace `(expr)!` with `(Nat.factorial (expr))`, handling nested parens."""
    import re
    while True:
        m = re.search(r'\)\s*!(?!\w)', s)
        if not m:
            break
        close_pos = m.start()  # position of `)`
        # Find matching `(`
        depth = 1
        j = close_pos - 1
        while j >= 0 and depth > 0:
            if s[j] == ')':
                depth += 1
            elif s[j] == '(':
                depth -= 1
            j -= 1
        j += 1
        if depth == 0:
            inner = s[j:close_pos + 1]
            s = s[:j] + f'(Nat.factorial {inner})' + s[m.end():]
        else:
            break
    return s


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


def _comma_to_colon_in_binders(body: str) -> str:
    """Convert the last ∀-style `,` conclusion separator to theorem-style `:`.

    In `∀ (x : T) (y : U), conclusion` the `,` separates binders from body.
    In `theorem name (x : T) (y : U) : conclusion` the `:` does the same.
    This finds the last `)` at paren-depth 0 followed by `,` and replaces with `:`.
    """
    depth = 0
    last_comma_pos = -1
    for i, c in enumerate(body):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                # Look ahead past whitespace for `,`
                j = i + 1
                while j < len(body) and body[j] in ' \t':
                    j += 1
                if j < len(body) and body[j] == ',':
                    last_comma_pos = j

    if last_comma_pos >= 0:
        return body[:last_comma_pos] + ' :' + body[last_comma_pos + 1:]
    return body


def _handle_compound_statement(name: str, statement: str) -> str:
    """Handle v2c compound statements with abbrev definitions + theorem.

    v2c data encodes some theorems as (often on a single line):
        ∀ abbrev solution_name : Type := sorry theorem thm_name (args...), conclusion
        abbrev solution_name : Type := sorry theorem thm_name : conclusion

    This emits the abbrev(s) as standalone declarations and the theorem
    with a sorry proof.
    """
    import re

    # Strip leading `∀ ` artifact
    statement = re.sub(r"^∀\s+", "", statement)

    # Split on `sorry` followed by `theorem` or `abbrev` — handles both
    # single-line and multi-line compound statements.
    parts = re.split(r"(?::=\s*sorry\s+)(?=(?:noncomputable\s+)?(?:abbrev|theorem)\s)", statement)

    result_lines = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        if part.startswith("theorem "):
            # Extract theorem body (everything after `theorem name`)
            m = re.match(r"(theorem\s+\S+)\s*(:?)\s*(.*)", part, re.DOTALL)
            if m:
                header = m.group(1)
                had_colon = bool(m.group(2))
                body = fix_statement(m.group(3))
                if had_colon:
                    result_lines.append(f"{header} : {body} := by sorry")
                else:
                    body = _comma_to_colon_in_binders(body)
                    result_lines.append(f"{header} {body} := by sorry")
            else:
                result_lines.append(f"{part} := by sorry")
        elif part.startswith(("abbrev ", "noncomputable ")):
            # abbrev declaration — add `:= sorry` since we split on it
            result_lines.append(f"{part} := sorry")
        else:
            # Unknown part — skip
            continue

    return "\n\n".join(result_lines)


def statement_to_theorem(name: str, statement: str) -> str:
    """Convert a ∀-expression statement to a theorem declaration with sorry proof."""
    # Handle compound v2c statements with abbrev + theorem
    s = statement.lstrip()
    if any(s.startswith(p) for p in ("∀ abbrev", "∀ noncomputable", "abbrev ", "noncomputable abbrev")):
        return _handle_compound_statement(name, statement)

    statement = fix_statement(statement)
    return f"theorem {name} : {statement} := by sorry"


def generate_lean_file(
    theorems: list[dict], module_name: str, skip_names: set[str] | None = None
) -> str:
    """Generate a complete Lean 4 file with sorry proofs for all theorems."""
    lines = [
        f"-- Auto-generated benchmark file: {module_name}",
        "-- Do not edit manually. Regenerate with:",
        "--   python python/data/minif2f/generate_lean.py",
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

        # Clean up name (some v2s entries have trailing colons)
        name = name.rstrip(":")

        # Fix malformed `∀ IsGreatest/IsLeast` (missing binder variable)
        if statement.startswith("∀ IsGreatest ") or statement.startswith("∀ IsLeast "):
            statement = statement[2:]  # strip leading `∀ `

        # Skip theorems with obviously broken statements
        if not statement.strip():
            skipped.append((name, "empty statement"))
            continue

        # Skip theorems using APIs that changed between Mathlib versions
        # (e.g., NNReal.IsConjExponent was renamed/removed in v4.27.0)
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
