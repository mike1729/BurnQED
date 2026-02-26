# miniF2F Benchmark Compilation: Lean 4.27

## Overview

miniF2F is the primary evaluation benchmark for BurnQED. It contains 488 competition-level math problems (244 test + 244 valid) formalized as Lean 4 theorem statements. We use three variants:

| Variant | Source | Description | Theorems |
|---------|--------|-------------|----------|
| **v1** | [yangky11/miniF2F-lean4](https://github.com/yangky11/miniF2F-lean4) | Original miniF2F, single correct answer | 244 test + 244 valid |
| **v2s** | [roozbeh-yz/miniF2F_v2](https://github.com/roozbeh-yz/miniF2F_v2) | Single answer (like v1, sometimes different) | 244 test + 244 valid |
| **v2c** | [roozbeh-yz/miniF2F_v2](https://github.com/roozbeh-yz/miniF2F_v2) | Multiple-choice (disjunction of 5 candidates) | 244 test + 244 valid |

## Pipeline

All scripts live in `python/data/minif2f/`. Run from the BurnQED project root.

### Step 1: Download datasets

```bash
python python/data/minif2f/download.py --output-dir data/benchmarks
```

Clones both GitHub repos, extracts theorem statements, converts to TheoremIndex JSON format (`{"theorems": [{"name": ..., "statement": ...}]}`).

Outputs:
- `data/benchmarks/minif2f_test.json` / `minif2f_valid.json` (v1)
- `data/benchmarks/minif2f_v2s_test.json` / `minif2f_v2s_valid.json`
- `data/benchmarks/minif2f_v2c_test.json` / `minif2f_v2c_valid.json`

### Step 2: Generate Lean files

```bash
python python/data/minif2f/generate_lean.py \
    --input data/benchmarks/minif2f_v2s_test.json \
    --output vendor/Pantograph/BenchMinIF2FV2sTest.lean \
    --module-name BenchMinIF2FV2sTest
```

Converts TheoremIndex JSON to a `.lean` file where each theorem is proved by `sorry`. The generator applies syntax fixes for Lean 4.27 / Mathlib v4.27 compatibility:

**Statement fixes (`fix_statement`):**
- `nhds` for `𝓝` notation (requires `open Topology`)
- BigOperators: `∑/∏ x in S` → `∑/∏ x ∈ S`
- Integer exponent division: `^ (1 / 3)` → `^ ((1 : ℝ) / 3)`
- Factorial notation: `n!` → `(Nat.factorial n)`
- Qualified names: `succ` → `Nat.succ`, `natAbs` → `Int.natAbs`
- Set-builder disambiguation: `{x : T | P}` → `({x : T | P})`
- Malformed binders from v2s/v2c data
- `EuclideanSpace` coercion for `![a, b]` notation
- `NNReal.IsConjExponent` skip (removed in Mathlib v4.27)

**v2c compound statements:**
Some v2c theorems include `abbrev` definitions before the theorem. These are emitted as standalone `abbrev ... := sorry` declarations followed by the theorem.

### Step 3: Compile

```bash
# Compile a single variant (from Pantograph directory)
cd vendor/Pantograph
lake env lean BenchMinIF2FV2sTest.lean

# Or use the orchestration script (from project root)
python python/data/minif2f/compile.py --variant v2s_test --timeout 300

# Compile all 6 variants
python python/data/minif2f/compile.py
```

The compile script uses `lake env lean` (not `lake build`), which injects `LEAN_PATH` from the Pantograph project's pre-compiled Mathlib `.olean` cache. Each file is compiled independently — no `lean_lib` targets needed in `lakefile.lean`.

Output: `data/benchmarks/compile_results/{variant}_results.json` with per-theorem pass/fail.

### Full pipeline (one command)

```bash
python python/data/minif2f/compile.py
```

This runs download → generate → compile → report for all 6 variants.

## File Layout

```
python/data/minif2f/
├── __init__.py
├── download.py       # Clone repos, extract theorems to TheoremIndex JSON
├── generate_lean.py  # Convert JSON → .lean sorry-files with syntax fixes
└── compile.py        # Orchestrate pipeline: download → generate → compile → report

data/benchmarks/
├── minif2f_test.json          # v1 test (244 theorems)
├── minif2f_valid.json         # v1 valid (244 theorems)
├── minif2f_v2s_test.json      # v2s test (244 theorems)
├── minif2f_v2s_valid.json     # v2s valid (244 theorems)
├── minif2f_v2c_test.json      # v2c test (244 theorems)
├── minif2f_v2c_valid.json     # v2c valid (244 theorems)
└── compile_results/           # Per-variant compilation results

vendor/Pantograph/
├── BenchMinIF2FTest.lean      # Generated v1 test
├── BenchMinIF2FValid.lean     # Generated v1 valid
├── BenchMinIF2FV2sTest.lean   # Generated v2s test (primary benchmark)
├── BenchMinIF2FV2sValid.lean  # Generated v2s valid
├── BenchMinIF2FV2cTest.lean   # Generated v2c test
└── BenchMinIF2FV2cValid.lean  # Generated v2c valid
```

## Key Decisions

1. **`lake env lean` over `lake build`:** Individual file compilation avoids modifying `lakefile.lean` (which would trigger dependency re-resolution and fail without network). Uses the same Mathlib `.olean` cache as Goedel compilation.

2. **Syntax fixes in generator, not source data:** The TheoremIndex JSONs store clean `∀`-expression statements. Lean-specific fixes (set-builder parenthesization, factorial notation, etc.) are applied during `.lean` generation so the JSONs remain reusable for other tools (Pantograph `goal.start`, etc.).

3. **v2s as primary benchmark:** v2s has single correct answers (like v1) but with improved formalization quality. v2c (multiple-choice) is secondary — it's easier (just prove one of 5 disjuncts) but useful for measuring whether the prover can at least get the right ballpark.

## Bug Fixes Applied

### `_extract_signature_from_formal` regex (download.py)
The v2 `formal_statement` ends with `:= by` (no trailing whitespace/sorry). The original regex `(?:by\s+)?` required whitespace after `by`, failing to strip the suffix. Fixed to `(?:by\s*)?`.

### `signature_to_expression` false colon match (download.py)
For no-param theorems like `theorem name : ∃ a b : ℕ, P`, the `:` in `a b : ℕ` was incorrectly identified as the param-goal separator. Fixed by requiring the separator `:` be preceded by a closing delimiter (`)`, `]`, `}`) at depth 0. Also strip leading `:` from no-param theorem signatures.

### v2c compound statements (generate_lean.py)
v2c data encodes some theorems as `abbrev solution : T := sorry theorem name (args), goal`. The handler wasn't splitting these correctly because it expected newline separators, but the data is on a single line. Fixed by splitting on `:= sorry` followed by `theorem`/`abbrev`. Also fixed detection of bare `abbrev` (without `∀` prefix).

### `let` binding chain separation (generate_lean.py + download.py)
Multi-line `let` bindings like `let a := expr\nlet b := expr\nbody` were flattened onto one line during whitespace normalization in `signature_to_expression`. Fixed by preserving newlines in the goal portion (`[ \t]+` instead of `\s+`), and inserting `;` between consecutive `let` bindings when they appear on the same line.

## Compilation Results

All 1,464 theorems (6 variants × 244) compile against **Lean 4.27.0 / Mathlib v4.27.0**:

| Variant | Split | Theorems | Status |
|---------|-------|----------|--------|
| v1      | test  | 244      | 244/244 pass |
| v1      | valid | 244      | 244/244 pass |
| v2s     | test  | 244      | 244/244 pass |
| v2s     | valid | 244      | 244/244 pass |
| v2c     | test  | 244      | 244/244 pass |
| v2c     | valid | 244      | 244/244 pass |
