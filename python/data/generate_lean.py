#!/usr/bin/env python3
"""Generate a Lean 4 file with sorry-proofs from a TheoremIndex JSON file.

This compiles theorem statements into .olean files so that Pantograph can
use `copyFrom(name)` instead of re-elaborating expressions from scratch.
Without this, complex expressions (involving ℂ, Finset, etc.) can take 60s+
to elaborate, exceeding the tactic timeout.

Usage:
    python python/data/generate_lean.py \
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

    # Fix: nhds bracket notation `nhds[≠] x` → `nhdsWithin x ({x}ᶜ)` etc.
    # These are PutnamBench shorthand for punctured/one-sided neighborhoods.
    # Handle both simple args (\w+) and parenthesized args like (ts k.1).
    statement = _fix_nhds_brackets(statement)

    # Fix: bare `cexp` → `Complex.exp` (removed from global namespace in Mathlib 4.27)
    statement = re.sub(r"(?<!\w)cexp\b", "Complex.exp", statement)

    # Fix: bare `sqrt` → `Real.sqrt` (not in bare namespace in Mathlib 4.27)
    statement = re.sub(r"(?<!\w\.)(?<!\w)sqrt\b", "Real.sqrt", statement)

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
    # Only apply at brace depth 0 to avoid corrupting set builders like `{(x,y) : F × F | ...}`.
    statement = _colon_to_comma_outside_braces(statement)

    # Fix: Qualify bare Nat/Int names when `open Nat` is not used.
    # These are common in IMO-Steps data. Negative lookbehind avoids
    # double-qualifying already-qualified names like Nat.succ.
    # Also exclude `.succ`/`.choose` (method syntax like `).choose`) with `(?<!\.)`.
    statement = re.sub(r"(?<!\.)(?<!\w)succ\b", "Nat.succ", statement)
    statement = re.sub(r"(?<!\.)(?<!\w)natAbs\b", "Int.natAbs", statement)
    statement = re.sub(r"(?<!\.)(?<!\w)choose\b", "Nat.choose", statement)

    # Fix: `id` is ambiguous when `open RingHom` is active (RingHom.id vs _root_.id).
    # Qualify as `_root_.id` to resolve. But skip `(id X)` pattern used in RatFunc.eval
    # where it means a RingHom identity, not _root_.id.
    statement = re.sub(r"(?<!\.)(?<!\w)id(?!\w)(?!\s+[ℝℂℕℤℚ])", "_root_.id", statement)

    # Fix: `lcm` is ambiguous between Nat.lcm and GCDMonoid.lcm.
    statement = re.sub(r"(?<!\.)(?<!\w)lcm\b", "Nat.lcm", statement)

    # Fix: Factorial notation `n!` or `n !` is not valid in Lean 4 theorem-type
    # position. Replace with `(Nat.factorial n)`. Parentheses needed so the result
    # works as an argument, e.g. `Nat.gcd (Nat.factorial 20) 200000`.
    # Handle `(expr)!` first (before simple patterns eat inner variables).
    statement = _fix_paren_factorials(statement)
    statement = re.sub(r"\b(\d+)\s*!(?!\[|=|₂|\w|\()", r"(Nat.factorial \1)", statement)
    # Single-letter var: n! or n ! → (Nat.factorial n)
    statement = re.sub(r"\b([a-z])\s*!(?!\[|=|₂|\w|\()", r"(Nat.factorial \1)", statement)
    # Multi-char var: require space before ! to avoid matching force-unwrap (getLast!)
    statement = re.sub(r"\b([a-z]\w+)\s+!(?!\[|=|₂|\w|\()", r"(Nat.factorial \1)", statement)

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

    # Fix: `let (a, b, c), =` destructuring with trailing comma → `let (a, b, c) :=`
    statement = re.sub(r'let\s*(\([^)]+\))\s*,\s*=', r'let \1 :=', statement)

    # Fix: `let` bindings flattened onto one line need `;` separators.
    # "let a := expr let b := expr" → "let a := expr; let b := expr"
    # Only insert `;` when preceded by a non-comma token (avoid breaking ∀ ..., let ...)
    statement = re.sub(r'(?<=[^\s,])\s+(let\s+\w+\s*:=)', r'; \1', statement)

    # Also handle `letI` bindings that need `;` separators.
    # Exclude `:` and logical operators (∧∨→↔) — `letI` after these is part of
    # an expression (type annotation or logical chain), not a separate binding.
    statement = re.sub(r'(?<=[^\s,:;∧∨→↔])\s+(letI\s)', r'; \1', statement)

    # Fix: `letI x := expr ∃` needs `;` between the binding body and a following
    # quantifier (∃/∀) that starts the continuation expression.
    # Only applies when there's a preceding `letI ... :=` (to avoid false positives).
    if 'letI' in statement:
        statement = re.sub(r'(?<=\))\s+(∃\s)', r'; \1', statement)

    # Fix: `∀ :` with no binder variable → strip the `∀ :` prefix.
    # Produced when signature_to_expression mistakes `:` in `:=` as a separator.
    statement = re.sub(r'^∀\s*:\s+', '', statement)

    # Fix: bare `rexp` → `Real.exp` (PutnamBench shorthand)
    statement = re.sub(r"(?<!\w\.)(?<!\w)rexp\b", "Real.exp", statement)

    # Note: `∆` (U+2206, INCREMENT) is the correct char for symmDiff notation
    # in Mathlib (with `open symmDiff`). Do NOT convert to U+0394 (Greek Delta).

    # Fix: bare `Simplex` → `Affine.Simplex` (lives in Affine namespace in Mathlib)
    statement = re.sub(r"(?<!\w\.)(?<!\w)Simplex\b", "Affine.Simplex", statement)

    # Fix: `Prime` is ambiguous when `open Nat` is active (_root_.Prime vs Nat.Prime).
    # Use `_root_.Prime` which works for both ℕ and ℤ (Nat.Prime only works for ℕ).
    statement = re.sub(r"(?<!\.)(?<!\w)Prime\b", "_root_.Prime", statement)

    # Fix: `φ` (U+03C6) is not a valid identifier in Lean 4.
    # Replace with `ϕ` (U+03D5, GREEK PHI SYMBOL) which IS valid.
    statement = statement.replace("\u03C6", "\u03D5")

    # Fix: `(Polynomial.X :` — dotted name in binder position is invalid.
    # Rename to `(X_var :` and update all references.
    if "(Polynomial.X :" in statement:
        statement = statement.replace("(Polynomial.X :", "(X_var :")
        statement = re.sub(r"(?<!\w)Polynomial\.X(?!\w)(?!\s*:)", "X_var", statement)

    # Fix: `Polynomial.coeff` in MvPolynomial context → `MvPolynomial.coeff`
    # Use regex with negative lookbehind to avoid double-qualifying `MvPolynomial.coeff`.
    if "MvPolynomial" in statement:
        statement = re.sub(r"(?<!Mv)Polynomial\.coeff", "MvPolynomial.coeff", statement)

    # Fix: bare `eval` in MvPolynomial context → `MvPolynomial.eval`
    # Match `eval (` (not just `eval ![`). Use negative lookbehind to skip `.eval`.
    if "MvPolynomial" in statement:
        statement = re.sub(r"(?<!\w)(?<!\.)eval(?=\s*[\(!\[])", "MvPolynomial.eval", statement)

    # Fix: bare `Perm` → `Equiv.Perm` (lives in Equiv namespace)
    statement = re.sub(r"(?<!\.)(?<!\w)Perm\b", "Equiv.Perm", statement)

    # Fix: bare `dist` is ambiguous between Nat.dist and Dist.dist when `open Nat`.
    # Qualify as `Dist.dist` for geometric contexts. Skip method syntax `.dist`.
    statement = re.sub(r"(?<!\.)(?<!\w)dist\b", "Dist.dist", statement)

    # Fix: bare `esymm` → `Multiset.esymm` (not opened in our context)
    statement = re.sub(r"(?<!\.)(?<!\w)esymm\b", "Multiset.esymm", statement)

    # Fix: bare `card` in Multiset context → `Multiset.card`
    # (open Finset makes bare `card` resolve to Finset.card)
    if "Multiset" in statement:
        statement = re.sub(r"(?<!\.)(?<!\w)card\b", "Multiset.card", statement)

    # Fix: `]]` (double close bracket) — Lean 4 tokenizes as single token.
    # Insert space: `] ]` so each bracket is parsed separately.
    statement = statement.replace("]]", "] ]")

    # Fix: `ℝ≥0∞` (ENNReal) is scoped notation requiring `open scoped ENNReal`.
    # Replace with the unscoped type name `ENNReal`.
    statement = statement.replace("ℝ≥0∞", "ENNReal")

    # Fix: bare `X` and `C` in MvPolynomial context → `MvPolynomial.X`, `MvPolynomial.C`
    # These are shadowed by `open Polynomial` which brings in Polynomial.X and Polynomial.C.
    if "MvPolynomial" in statement:
        # Qualify bare X used as function: `X c`, `X 0` (followed by word char or digit)
        # Skip X in binder position: `(X :` and already-qualified `MvPolynomial.X`
        statement = re.sub(r"(?<!\w)(?<!\.)X(?=\s+[\w\d(])", "MvPolynomial.X", statement)
        # Qualify bare C used as function: `C a`, `C (expr)`
        statement = re.sub(r"(?<!\w)(?<!\.)C(?=\s+[\w(])", "MvPolynomial.C", statement)
        # Qualify bare aeval
        statement = re.sub(r"(?<!\w)(?<!\.)aeval\b", "MvPolynomial.aeval", statement)

    # Fix: variable named `X` collides with `Polynomial.X` from `open Polynomial`.
    # Only trigger when `(X : Set ...)` — i.e., X is bound as a Set variable.
    # Avoids corrupting `(X : ℚ[X])` type casts.
    if re.search(r'\(X\s*:\s*Set\b', statement) and 'MvPolynomial' not in statement:
        statement = re.sub(r'\(X\s*:', '(X_set :', statement)
        # Also rename references: `∈ X`, `|X]`, bare `X ↔` etc.
        # Use word boundary to avoid renaming Polynomial.X
        statement = re.sub(r'(?<!\.)(?<!\w)X(?!\w)(?!\s*:)', 'X_set', statement)

    return statement


def _extract_nhds_arg(s: str, start: int) -> tuple[str, int]:
    """Extract the argument after nhds[X] starting at position `start`.

    Handles simple args (`x`), parenthesized args (`(ts k.1)`),
    and angle-bracket args (`⟨1,1⟩`).
    Returns (arg_text, end_position).
    """
    # Skip whitespace
    i = start
    while i < len(s) and s[i] in ' \t\n\r':
        i += 1
    if i >= len(s):
        return '', start
    if s[i] == '(':
        # Parenthesized argument — find matching close paren
        depth = 1
        j = i + 1
        while j < len(s) and depth > 0:
            if s[j] == '(':
                depth += 1
            elif s[j] == ')':
                depth -= 1
            j += 1
        return s[i:j], j
    elif s[i] == '⟨':
        # Angle-bracket argument — find matching ⟩
        depth = 1
        j = i + 1
        while j < len(s) and depth > 0:
            if s[j] == '⟨':
                depth += 1
            elif s[j] == '⟩':
                depth -= 1
            j += 1
        return s[i:j], j
    else:
        # Simple word argument
        j = i
        while j < len(s) and (s[j].isalnum() or s[j] in "_."):
            j += 1
        return s[i:j], j


def _fix_nhds_brackets(statement: str) -> str:
    """Rewrite nhds bracket notation to nhdsWithin calls.

    Handles both `nhds[>] x` and `nhds[>] (expr)` forms.
    Also handles `nhds[S] x` where S is a set variable (uppercase letter).
    """
    import re
    # Comparison operators: nhds[>] x → nhdsWithin x (Set.Ioi x)
    _nhds_cmp_map = {
        '≠': lambda arg: f'nhdsWithin {arg} ({{{arg}}}ᶜ)',
        '>': lambda arg: f'nhdsWithin {arg} (Set.Ioi {arg})',
        '<': lambda arg: f'nhdsWithin {arg} (Set.Iio {arg})',
        '≥': lambda arg: f'nhdsWithin {arg} (Set.Ici {arg})',
        '≤': lambda arg: f'nhdsWithin {arg} (Set.Iic {arg})',
    }
    result = []
    i = 0
    while i < len(statement):
        m = re.match(r'nhds\[(.)\]', statement[i:])
        if m:
            bracket_char = m.group(1)
            arg, end_pos = _extract_nhds_arg(statement, i + m.end())
            if arg:
                if bracket_char in _nhds_cmp_map:
                    # Comparison operator: point = arg, set = f(arg)
                    result.append(_nhds_cmp_map[bracket_char](arg))
                    i = end_pos
                    continue
                elif bracket_char.isupper():
                    # Set variable: nhds[S] point → nhdsWithin point S
                    result.append(f'nhdsWithin {arg} {bracket_char}')
                    i = end_pos
                    continue
        result.append(statement[i])
        i += 1
    return ''.join(result)


def _fix_paren_factorials(s: str) -> str:
    """Replace `(expr)!` with `(Nat.factorial (expr))`, handling nested parens."""
    import re
    while True:
        m = re.search(r'\)\s*!(?!\[|=|₂|\w|\()', s)
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


def _colon_to_comma_outside_braces(s: str) -> str:
    """Replace `) : CONCLUSION` with `), CONCLUSION` only at top level.

    Only applies when the `)` closes the last open paren (paren depth goes to 0)
    AND we're not inside braces (brace depth 0). This avoids corrupting:
    - Set builders like `{(x,y) : F × F | ...}` (inside braces)
    - Type casts like `((expr) : ℝ)` (inner `)` is at paren depth > 0)
    """
    import re
    result = []
    brace_depth = 0
    paren_depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
        elif ch == '(':
            paren_depth += 1
        elif ch == ')':
            paren_depth -= 1
            if brace_depth == 0 and paren_depth == 0:
                # Top-level `)` — check if followed by `: CONCLUSION`
                m = re.match(r'\)\s*:\s+(?=\S)', s[i:])
                if m:
                    result.append(')')
                    result.append(', ')
                    i += m.end()
                    continue
        result.append(ch)
        i += 1
    return ''.join(result)


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
    theorems: list[dict],
    module_name: str,
    skip_names: set[str] | None = None,
    extra_opens: list[str] | None = None,
    preamble: list[str] | None = None,
) -> str:
    """Generate a complete Lean 4 file with sorry proofs for all theorems."""
    open_line = "open Real"
    if extra_opens:
        all_opens = ["Real"] + [o for o in extra_opens if o != "Real"]
        open_line = "open " + " ".join(all_opens)

    lines = [
        f"-- Auto-generated benchmark file: {module_name}",
        "-- Do not edit manually. Regenerate with:",
        "--   python python/data/generate_lean.py",
        "",
        "import Mathlib",
        "",
        "set_option maxHeartbeats 800000",
        "",
        open_line,
        "",
    ]

    if preamble:
        lines.append("-- Solution abbrevs (referenced by theorems)")
        for decl in preamble:
            lines.append(decl)
            lines.append("")

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
    parser.add_argument(
        "--extra-opens",
        nargs="*",
        default=[],
        help="Additional Lean namespaces to open (e.g. Filter Topology Set)",
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

    preamble = data.get("preamble", [])

    module_name = args.module_name or output_path.stem

    skip_names = set()
    if args.skip_file:
        skip_path = Path(args.skip_file)
        if skip_path.exists():
            skip_names = {line.strip() for line in skip_path.read_text().splitlines() if line.strip()}
            print(f"Loaded {len(skip_names)} skip names from {skip_path}")

    lean_content = generate_lean_file(
        theorems, module_name, skip_names,
        extra_opens=args.extra_opens or None,
        preamble=preamble or None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(lean_content)

    print(f"Generated {output_path}: {len(theorems)} theorems, module {module_name}")


if __name__ == "__main__":
    main()
