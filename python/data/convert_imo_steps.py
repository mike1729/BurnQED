#!/usr/bin/env python3
"""Convert IMO-Steps Lean 4 files to TheoremIndex JSON.

Supports three evaluation modes:
  1. Full theorems (imo_proofs/)   — 13-20 complete IMO problem proofs
  2. Lemma steps (Lemmas/)         — 1,329 incremental proof steps
  3. Extended archive (Lean_v20/)  — 37 files, broader problem coverage

Usage:
    # Full theorems
    python convert_imo_steps.py \
        --input-dir /path/to/IMO-Steps/imo_proofs \
        --output data/imo_steps_theorems.json

    # Lemma steps
    python convert_imo_steps.py \
        --input-dir /path/to/IMO-Steps/Lemmas \
        --output data/imo_steps_lemmas.json

    # Extended archive
    python convert_imo_steps.py \
        --input-dir /path/to/IMO-Steps/Lean_v20/imo_proofs \
        --output data/imo_steps_v20.json

Output format (TheoremIndex JSON):
    {
      "theorems": [
        {"name": "imo_1959_p1", "statement": "∀ (n : ℕ) (h₀ : 0 < n), ..."},
        ...
      ]
    }
"""

import argparse
import json
import re
import sys
from pathlib import Path


def extract_declarations(lean_source: str) -> list[dict]:
    """Extract theorem/lemma declarations from Lean 4 source.

    Returns list of dicts with 'name' and 'raw_type' (everything between
    the name and ':= by', representing the full type signature).
    """
    # Split on theorem/lemma boundaries
    blocks = re.split(r'(?=^(?:theorem|lemma)\b)', lean_source, flags=re.MULTILINE)

    declarations = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Match theorem/lemma name
        match = re.match(r'^(?:theorem|lemma)\s+(\S+)', block)
        if not match:
            continue

        name = match.group(1)

        # Extract everything between name and ':= by'
        # The type signature is: (params...) : conclusion
        after_name = block[match.end():]
        idx = after_name.find(':= by')
        if idx < 0:
            # Try ':= by\n' or just ':= by' at end
            idx = after_name.find(':=')
            if idx < 0:
                continue

        raw_type = after_name[:idx].strip()
        # Strip single-line comments (-- ...)
        raw_type = re.sub(r'--[^\n]*', '', raw_type)
        # Clean up blank lines left by comment removal
        raw_type = re.sub(r'\n\s*\n', '\n', raw_type).strip()
        declarations.append({'name': name, 'raw_type': raw_type})

    return declarations


def type_to_forall(raw_type: str) -> str:
    """Convert a Lean type signature to a ∀-expression.

    Input:  '(n : ℕ) (h₀ : 0 < n) :\\n  Nat.gcd (21*n + 4) (14*n + 3) = 1'
    Output: '∀ (n : ℕ) (h₀ : 0 < n), Nat.gcd (21*n + 4) (14*n + 3) = 1'

    Strategy: consume binders (parenthesized groups at depth 0) from left to right.
    The first ':' at depth 0 after all binders is the conclusion separator.
    """
    # Scan from the start, consuming whitespace and binder groups '(...)'.
    # The first non-whitespace, non-binder character should be ':'.
    pos = 0
    n = len(raw_type)

    while pos < n:
        # Skip whitespace
        if raw_type[pos] in ' \t\n\r':
            pos += 1
            continue

        # If we see '(' or '{' at depth 0, consume the entire binder group
        if raw_type[pos] in '({':
            close = ')' if raw_type[pos] == '(' else '}'
            depth = 1
            pos += 1
            while pos < n and depth > 0:
                if raw_type[pos] in '({':
                    depth += 1
                elif raw_type[pos] in ')}':
                    depth -= 1
                pos += 1
            continue

        # First non-whitespace, non-binder character
        break

    if pos >= n:
        # All content was binders — no conclusion
        return raw_type.strip()

    if raw_type[pos] == ':':
        # Found the conclusion separator
        binders = raw_type[:pos].strip()
        conclusion = raw_type[pos + 1:].strip()
        if not binders:
            return conclusion
        return f"∀ {binders}, {conclusion}"

    if raw_type[pos] == ',':
        # Already in ∀-form (e.g., lemma with trailing comma)
        binders = raw_type[:pos].strip()
        conclusion = raw_type[pos + 1:].strip()
        if not binders:
            return conclusion
        return f"∀ {binders}, {conclusion}"

    # No binders found — the entire expression is the conclusion
    return raw_type.strip()


def convert_directory(input_dir: Path) -> list[dict]:
    """Convert all Lean files in a directory to TheoremIndex entries."""
    theorems = []
    lean_files = sorted(input_dir.glob('*.lean'))

    for lean_file in lean_files:
        # Skip module declaration files
        if lean_file.name == 'ImoSteps.lean':
            continue

        source = lean_file.read_text()
        declarations = extract_declarations(source)

        for decl in declarations:
            statement = type_to_forall(decl['raw_type'])
            # Clean up whitespace: normalize multi-line to single spaces
            # but preserve newlines within expressions for readability
            statement = re.sub(r'[ \t]+', ' ', statement)
            # Remove leading/trailing whitespace per line
            lines = [line.strip() for line in statement.split('\n')]
            statement = '\n  '.join(lines)

            theorems.append({
                'name': decl['name'],
                'statement': statement,
            })

    return theorems


def main():
    parser = argparse.ArgumentParser(
        description='Convert IMO-Steps Lean files to TheoremIndex JSON'
    )
    parser.add_argument(
        '--input-dir', required=True,
        help='Directory containing IMO-Steps .lean files'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output TheoremIndex JSON file path'
    )
    parser.add_argument(
        '--main-only', action='store_true',
        help='Only include main IMO theorems (imo_YYYY_pN), skip helpers'
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: Not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    theorems = convert_directory(input_dir)
    if not theorems:
        print(f"ERROR: No theorems found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.main_only:
        before = len(theorems)
        theorems = [t for t in theorems if re.match(r'imo_\d{4}_p\d+(_\d+)?$', t['name'])]
        print(f"Filtered to {len(theorems)} main theorems (skipped {before - len(theorems)} helpers)")

    # Deduplicate names by appending source file suffix
    seen = {}
    for t in theorems:
        name = t['name']
        if name in seen:
            seen[name] += 1
            t['name'] = f"{name}__dup{seen[name]}"
        else:
            seen[name] = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'theorems': theorems}, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(theorems)} theorems from {input_dir} → {output_path}")


if __name__ == '__main__':
    main()
