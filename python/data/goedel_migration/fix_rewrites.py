#!/usr/bin/env python3
"""Fix Goedel rewrite failures from Mathlib v4.8→v4.27 field_simp changes.

Root cause: field_simp in v4.27 normalizes goals more aggressively, so
subsequent `rw [div_le_div_iff₀]` (etc.) can't find the expected pattern.

Fix strategy: Replace the rewrite with a `first` combinator that tries all
division-clearing rewrites, plus `skip` for when field_simp already cleared
everything. Use `<;>` to broadcast the subsequent tactic to all goals.

Usage:
  python3 fix_goedel_rewrites.py --data-dir /path/to/goedel_data
  python3 fix_goedel_rewrites.py --data-dir /path/to/goedel_data --dry-run
  python3 fix_goedel_rewrites.py --data-dir /path/to/goedel_data --files Proof_00002.lean Proof_00048.lean
"""

import argparse
import re
from pathlib import Path


# The rewrite lemma names we need to fix
DIV_LEMMAS = [
    "div_le_div_iff₀",
    "le_div_iff₀",
    "div_le_iff₀",
    "div_lt_div_iff₀",
    "lt_div_iff₀",
    "div_lt_iff₀",
    "le_div_iff₀'",
    "div_le_iff₀'",
]

# Build regex that matches rw [...div_lemma...]
# Match: rw [lemma_name optional_args] optional_at optional_semicolon
RW_PATTERN = re.compile(
    r'^(\s*)'                              # capture indent
    r'rw\s*\['                             # rw [
    r'('                                   # capture the full rw contents
    r'(?:' + '|'.join(re.escape(l) for l in DIV_LEMMAS) + r')'  # lemma name
    r'(?:\s*\([^)]*\))*'                   # optional args like (by positivity) (by nlinarith)
    r')'
    r'('                                   # capture rest-of-rw (other rewrites in same rw call)
    r'(?:\s*,\s*[^]]*)?'                   # e.g., , ← sub_nonneg
    r')'
    r'\]'                                  # ]
    r'('                                   # capture suffix
    r'(?:\s+at\s+\w+)?'                    # optional: at H
    r')'
    r'(\s*<;>)?'                           # optional <;>
    r'(.*)',                               # rest of line
    re.MULTILINE
)

# The three division-clearing rewrites (without args — let Lean create goals)
FIRST_BLOCK = "(first | rw [div_le_div_iff₀] | rw [le_div_iff₀] | rw [div_le_iff₀] | skip)"
FIRST_BLOCK_LT = "(first | rw [div_lt_div_iff₀] | rw [lt_div_iff₀] | rw [div_lt_iff₀] | skip)"


def fix_file(filepath: Path, dry_run: bool = False) -> bool:
    """Fix rewrite-div patterns in a single file. Returns True if modified."""
    text = filepath.read_text()
    lines = text.split('\n')
    new_lines = []
    modified = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this line has a div rewrite
        m = RW_PATTERN.match(line)
        if m:
            indent = m.group(1)
            rw_contents = m.group(2)  # e.g., div_le_div_iff₀ (by positivity) (by positivity)
            extra_rws = m.group(3)    # e.g., , ← sub_nonneg
            at_clause = m.group(4)    # e.g., at H
            has_semicolon = m.group(5) is not None
            rest = m.group(6).strip()

            # Determine if it's a < variant
            is_lt = any(lt in rw_contents for lt in ["div_lt_div_iff₀", "lt_div_iff₀", "div_lt_iff₀"])
            block = FIRST_BLOCK_LT if is_lt else FIRST_BLOCK

            # Handle "at H" — keep original rewrite with try wrapper
            if at_clause and at_clause.strip():
                # For hypothesis rewrites, just wrap in try
                new_line = f"{indent}(try rw [{rw_contents}{extra_rws}]{at_clause})"
                if has_semicolon:
                    new_line += " <;>"
                if rest:
                    new_line += f" {rest}"
                new_lines.append(new_line)
                modified = True
                i += 1
                continue

            # Handle extra rewrites in same rw call (e.g., , ← sub_nonneg)
            if extra_rws and extra_rws.strip():
                # Complex case: multiple rewrites in one call
                # Use try for the div rewrite, keep the rest
                new_line = f"{indent}(try rw [{rw_contents}])"
                new_lines.append(new_line)
                new_line2 = f"{indent}try rw [{extra_rws.strip().lstrip(','). strip()}]"
                new_lines.append(new_line2)
                modified = True
                i += 1
                continue

            # Standard case: rw [div_lemma args] possibly <;> rest
            if has_semicolon and rest:
                # Pattern: rw [...] <;> tactic
                # → (first | ...) <;> tactic
                new_lines.append(f"{indent}{block} <;>")
                new_lines.append(f"{indent}  {rest}")
                modified = True
                i += 1
            elif has_semicolon:
                # Pattern: rw [...] <;>
                # next line has the tactic
                new_lines.append(f"{indent}{block} <;>")
                modified = True
                i += 1
            else:
                # Sequential pattern: rw [...] alone, followed by tactic lines
                # The old rw [div_le_div_iff₀] (no args) creates 2 positivity side
                # goals; rw [le_div_iff₀] creates 1. The subsequent tactic lines
                # handle main goal + side goals. With (first | ... | skip) <;>,
                # a single tactic handles ALL goals. Leftover lines need `try`.

                # Helper: consume a possibly multi-line tactic (track bracket balance)
                def consume_tactic(start_idx):
                    """Return (tactic_text_lines, next_idx) for a possibly multi-line tactic."""
                    tactic_lines = []
                    j = start_idx
                    balance = 0
                    while j < len(lines):
                        ln = lines[j]
                        s = ln.strip()
                        if not tactic_lines and (not s or s.startswith('--')):
                            j += 1
                            continue
                        tactic_lines.append(ln)
                        balance += s.count('[') + s.count('(') - s.count(']') - s.count(')')
                        j += 1
                        if balance <= 0 and tactic_lines:
                            break
                    return tactic_lines, j

                # Consume the first tactic after the rw
                main_tactic_lines, after_main = consume_tactic(i + 1)

                if main_tactic_lines:
                    rw_indent_len = len(indent)

                    # Emit: (first | ...) <;> tactic (possibly multi-line)
                    new_lines.append(f"{indent}{block} <;>")
                    first_line = main_tactic_lines[0].strip()
                    new_lines.append(f"{indent}  {first_line}")
                    for extra_line in main_tactic_lines[1:]:
                        new_lines.append(extra_line)
                    modified = True

                    # Now scan subsequent lines: wrap any nlinarith/linarith/
                    # positivity calls at the same indent in `try` (they were
                    # handling positivity side goals from the old rewrite)
                    SIDE_GOAL_TACTICS = {'nlinarith', 'linarith', 'positivity',
                                         'norm_num', 'omega', 'ring_nf', 'simp'}
                    i = after_main
                    while i < len(lines):
                        candidate = lines[i]
                        cstripped = candidate.strip()

                        # Skip blanks and comments
                        if not cstripped or cstripped.startswith('--'):
                            new_lines.append(candidate)
                            i += 1
                            continue

                        # Check if this line is at the same indent level
                        line_indent = len(candidate) - len(candidate.lstrip())
                        if line_indent < rw_indent_len:
                            break  # Different proof block

                        # Check if it's a side-goal-handling tactic
                        first_word = cstripped.split()[0].split('[')[0].split('(')[0]
                        if first_word in SIDE_GOAL_TACTICS:
                            # Consume entire multi-line tactic and wrap in try
                            side_lines, after_side = consume_tactic(i)
                            line_indent_str = candidate[:line_indent]
                            side_first = side_lines[0].strip()
                            new_lines.append(f"{line_indent_str}try {side_first}")
                            for extra_line in side_lines[1:]:
                                new_lines.append(extra_line)
                            i = after_side
                        else:
                            break  # Different kind of tactic, stop wrapping

                    continue
                else:
                    # No subsequent tactic found, just wrap in first
                    new_lines.append(f"{indent}{block}")
                    modified = True
                    i += 1
        else:
            new_lines.append(line)
            i += 1

    if modified and not dry_run:
        filepath.write_text('\n'.join(new_lines))

    return modified


def main():
    parser = argparse.ArgumentParser(description="Fix Goedel rewrite-div failures")
    parser.add_argument("--data-dir", required=True, help="Directory with .lean proof files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed")
    parser.add_argument("--files", nargs="*", help="Specific files to fix (default: all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.files:
        files = [data_dir / f for f in args.files]
    else:
        files = sorted(data_dir.glob("Proof_*.lean"))

    modified_count = 0
    for filepath in files:
        if fix_file(filepath, dry_run=args.dry_run):
            modified_count += 1
            if args.dry_run:
                print(f"  WOULD modify: {filepath.name}")

    print(f"\n{'Would modify' if args.dry_run else 'Modified'}: {modified_count}/{len(files)} files")


if __name__ == "__main__":
    main()
