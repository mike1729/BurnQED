# Goedel Dataset Analysis (Lean 4.27, Traced via Pantograph)

**Date:** 2026-03-06
**Source:** `data/traced/pantograph_pairs/goedel_427_pairs.jsonl`
**Origin:** Goedel-LM/Lean-workbook-proofs, migrated to Lean 4.27 (Phase M)

## Overview

| Metric | Value |
|--------|-------|
| Total tactic pairs | 60,341 |
| Total theorems | 24,879 |
| Mean tactics/theorem | 2.43 |
| Median tactics/theorem | 2 |
| p95 tactics/theorem | 7 |
| p99 tactics/theorem | 11 |
| Max tactics/theorem | 37 |
| Unique tactic heads | 73 |
| Tactic head entropy | 3.89 bits |

All theorems are named `lean_workbook_*` (competition math from Lean Workbook, proved by DeepSeek-Prover-V1.5).

## Proof Depth Distribution (max depth per theorem)

| Depth | Theorems | % | Cumulative % |
|-------|----------|---|--------------|
| 0 (single-tactic) | 11,512 | 46.3% | 46.3% |
| 1 | 6,344 | 25.5% | 71.7% |
| 2-3 | 3,735 | 15.0% | 86.7% |
| 4-6 | 2,178 | 8.8% | 95.5% |
| 7-10 | 931 | 3.7% | 99.3% |
| 11+ | 179 | 0.7% | 100.0% |

**Key takeaway:** 46% of proofs are single-tactic (trivial for search), 72% are depth <= 1. Only 13% have depth >= 4 (the interesting range for search training). The deep tail (7+) has 1,110 theorems.

## Goal Type Distribution

| Type | Count | % |
|------|-------|---|
| Inequality (>=, <=, >, <) | 17,561 | 70.6% |
| Equality (=) | 5,968 | 24.0% |
| Other | 520 | 2.1% |
| Divisibility (∣) | 399 | 1.6% |
| Universal (∀) | 271 | 1.1% |
| Existential (∃) | 131 | 0.5% |
| Iff (↔) | 29 | 0.1% |

**Heavily dominated by inequalities** — expected for competition algebra dataset.

## Tactic Usage (Top 20)

| Tactic | Count | % | Cumulative |
|--------|-------|---|------------|
| have | 16,209 | 26.9% | 26.9% |
| nlinarith | 10,702 | 17.7% | 44.6% |
| intro | 4,068 | 6.7% | 51.3% |
| field_simp | 3,575 | 5.9% | 57.3% |
| ring_nf | 3,018 | 5.0% | 62.3% |
| first | 2,433 | 4.0% | 66.3% |
| rw | 2,099 | 3.5% | 69.8% |
| constructor | 1,827 | 3.0% | 72.8% |
| simp | 1,766 | 2.9% | 75.7% |
| norm_num | 1,757 | 2.9% | 78.7% |
| apply | 1,523 | 2.5% | 81.2% |
| try | 1,361 | 2.3% | 83.5% |
| <;> | 1,005 | 1.7% | 85.1% |
| rcases | 920 | 1.5% | 86.6% |
| linarith | 859 | 1.4% | 88.1% |
| ring | 854 | 1.4% | 89.5% |
| cases' | 815 | 1.4% | 90.8% |

Top 3 tactics cover 51%. Top 17 cover 90%.

### Single-Tactic Proof Breakdown (11,512 theorems)

| Tactic | Count | % of single-tactic |
|--------|-------|--------------------|
| nlinarith | 6,058 | 52.6% |
| have | 1,539 | 13.4% |
| norm_num | 672 | 5.8% |
| cases' | 483 | 4.2% |
| induction | 455 | 4.0% |
| rw | 400 | 3.5% |
| apply | 278 | 2.4% |
| ring_nf | 277 | 2.4% |
| simp | 206 | 1.8% |
| exact | 196 | 1.7% |

Over half of single-tactic proofs are just `nlinarith` — machine-generated proofs lean heavily on arithmetic solvers.

## Depth-Stratified Tactic Diversity

| Depth Bucket | Theorems | Steps | Unique Heads | Entropy (bits) | Top Tactic |
|--------------|----------|-------|-------------|----------------|------------|
| 0 (single) | 11,512 | 11,512 | 46 | 2.76 | nlinarith (53%) |
| 1 | 6,344 | 12,688 | 56 | 4.11 | nlinarith (19%) |
| 2-3 | 3,735 | 12,500 | 66 | 4.40 | have (17%) |
| 4-6 | 2,178 | 12,625 | 56 | 3.39 | have (42%) |
| 7+ | 1,110 | 11,016 | 45 | 2.61 | have (58%) |

**Pattern:** Shallow proofs dominated by `nlinarith`. Deeper proofs increasingly dominated by `have` (intermediate lemma introduction), with `field_simp` and `(first|...)` combinator as support. Entropy peaks at depth 2-3 (most diverse tactic mix) and drops for deep proofs (repetitive `have` chains).

### Depth 0 vs Deeper Tactic Shift

| Tactic | Depth 0 | Deeper | Deeper/D0 Ratio |
|--------|---------|--------|-----------------|
| have | 4,638 | 11,570 | 2.5x |
| field_simp | 1,041 | 2,534 | 2.4x |
| linarith | 83 | 776 | 9.4x |
| ring | 90 | 764 | 8.5x |
| try | 0 | 1,055 | ∞ (combinator) |
| <;> | 0 | 1,005 | ∞ (combinator) |
| nlinarith | 6,058 | 4,644 | 0.8x |
| constructor | 1,384 | 443 | 0.3x |
| intro | 2,846 | 1,203 | 0.4x |

Deeper steps use more `have` chains, `field_simp`, `linarith`/`ring` (simplification after setup), and tactical combinators (`try`, `<;>`). Root steps use more `nlinarith` (one-shot closers), `constructor`, and `intro`.

## Lemma References (Top 20)

| Lemma/Identifier | Count |
|------------------|-------|
| sub_nonneg.mpr | 5,516 |
| Real.sqrt | 3,383 |
| sub_pos.mpr | 1,598 |
| Real.sqrt_nonneg | 933 |
| ha.ne / hb.ne / hc.ne | 1,790 |
| ha.le / hb.le / hc.le | 1,510 |
| Int.ModEq | 610 |
| Int.mul_emod | 246 |
| Nat.pow_succ | 231 |
| Finset.sum_range_succ | 175 |
| Int.add_emod | 167 |
| Eq.symm | 159 |
| Real.le_sqrt_of_sq_le | 127 |
| Real.sq_sqrt | 102 |

Heavy use of `sub_nonneg`/`sub_pos` (inequality manipulation), `Real.sqrt*` (radical algebra), and modular arithmetic (`Int.ModEq`, `Int.mul_emod`). Hypothesis destructuring (`.ne`, `.le`, `.1`, `.2`) is pervasive.

## State and Tactic Size

| Metric | State (chars) | Tactic (chars) |
|--------|---------------|----------------|
| Mean | 155 | 71 |
| Median | 127 | 36 |
| p90 | 255 | 193 |
| p95 | 324 | 265 |
| p99 | 634 | 438 |
| Max | 83,726 | 1,900 |

States are compact (competition problems have few hypotheses). One extreme outlier at 83K chars.

### Hypothesis Count (root goals)

| Metric | Value |
|--------|-------|
| Mean | 3.8 |
| Median | 4 |
| p90 | 7 |
| p95 | 8 |
| Max | 24 |

## Recommendations for 10K Search Sample

Given the depth distribution and the goal of training an EBM that discriminates proof-state quality, the sample should **over-represent deeper proofs** (more search branching, more value signal) while including enough shallow proofs to maintain coverage.

### Suggested Depth-Stratified Sample (10,000 theorems)

| Depth | Available | Sample | % of Sample | Sampling Rate |
|-------|-----------|--------|-------------|---------------|
| 0 | 11,512 | 2,000 | 20% | 17% (subsample) |
| 1 | 6,344 | 2,000 | 20% | 32% (subsample) |
| 2-3 | 3,735 | 2,500 | 25% | 67% (subsample) |
| 4-6 | 2,178 | 2,178 | 22% | 100% (all) |
| 7+ | 1,110 | 1,110 | 11% | 100% (all) |
| **Total** | **24,879** | **9,788** | | |

Round up to 10K by adding ~212 from depth 2-3 bucket. This gives:
- All 3,288 deep proofs (depth >= 4) included — no waste
- 13,367 theorems at depth >= 2 well-represented (4,000 of 7,023)
- Depth 0-1 included for coverage but not over-represented

**Alternative: Uniform depth weighting** — sample equal counts per depth bucket (2,000 each for 5 buckets). This maximizes depth diversity but under-represents the natural distribution, which may hurt SFT if used for training.

**For pure search/EBM purposes:** The depth-stratified sample above is recommended. Shallow proofs provide little signal for tree-search training.
