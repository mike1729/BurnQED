# LEAN-GitHub Dataset Analysis (Pre-traced Tactic Pairs)

**Date:** 2026-03-06
**Source:** `data/traced/lean_github_pairs.jsonl`
**Origin:** internlm/Lean-Github (218K pairs from 28.6K theorems), filtered in Phase M.9

## Overview

| Metric | Value |
|--------|-------|
| Total tactic pairs | 196,853 |
| Total theorems | 18,778 |
| Total repos | 146 |
| Mean tactics/theorem | 10.48 |
| Median tactics/theorem | 4 |
| p95 tactics/theorem | 40 |
| p99 tactics/theorem | 92 |
| Max tactics/theorem | 625 |
| Unique tactic heads | 996 |
| Tactic head entropy | 5.10 bits |

Human-written proofs from diverse GitHub repos. Much deeper and more diverse than Goedel.

## Comparison with Goedel

| Metric | Goedel | LEAN-GitHub |
|--------|--------|-------------|
| Theorems | 24,879 | 18,778 |
| Tactic pairs | 60,341 | 196,853 |
| Mean tactics/thm | 2.43 | 10.48 |
| Median tactics/thm | 2 | 4 |
| Unique tactic heads | 73 | 996 |
| Entropy | 3.89 bits | 5.10 bits |
| Single-tactic % | 46.3% | 14.4% (2,816 / 15.0%) |
| Depth >= 4 | 13.3% | 48.4% |
| Depth >= 11 | 0.7% | 22.5% |
| Dominant goal type | Inequality (71%) | Equality (45%) |

LEAN-GitHub is **dramatically deeper, more diverse, and more structurally complex** than Goedel. It provides the proof complexity Goedel lacks.

## Proof Depth Distribution (max depth per theorem)

| Depth | Theorems | % | Cumulative % |
|-------|----------|---|--------------|
| 0 | 2,816 | 15.0% | 15.0% |
| 1 | 3,772 | 20.1% | 35.1% |
| 2 | 1,276 | 6.8% | 41.9% |
| 3 | 1,887 | 10.0% | 51.9% |
| 4 | 842 | 4.5% | 56.4% |
| 5 | 1,160 | 6.2% | 62.6% |
| 6 | 571 | 3.0% | 65.6% |
| 7 | 865 | 4.6% | 70.2% |
| 8 | 339 | 1.8% | 72.0% |
| 9 | 605 | 3.2% | 75.3% |
| 10 | 271 | 1.4% | 76.7% |
| 11 | 466 | 2.5% | 79.2% |
| 12 | 191 | 1.0% | 80.2% |
| 13 | 393 | 2.1% | 82.3% |
| 14 | 155 | 0.8% | 83.1% |
| 15+ | 3,169 | 16.9% | 100.0% |

**Key takeaway:** Only 15% single-tactic vs 46% in Goedel. Nearly half (48%) have depth >= 4. Long tail of 4,230 theorems at depth >= 11. Bimodal pattern at odd depths (3, 5, 7, 9, ...) suggesting structured proof patterns like case splits.

## Goal Type Distribution

| Type | Count | % |
|------|-------|---|
| Equality | 8,355 | 44.5% |
| Inequality | 4,937 | 26.3% |
| Other | 3,727 | 19.8% |
| Iff (↔) | 825 | 4.4% |
| Universal (∀) | 446 | 2.4% |
| Existential (∃) | 404 | 2.2% |
| Divisibility (∣) | 84 | 0.4% |

**Much more balanced than Goedel.** "Other" (20%) includes type-theoretic goals, set membership, propositions — reflecting real Lean library code vs pure competition math.

## Tactic Usage (Top 20)

| Tactic | Count | % | Cumulative |
|--------|-------|---|------------|
| rw | 24,577 | 12.5% | 12.5% |
| simp | 23,940 | 12.2% | 24.6% |
| exact | 20,518 | 10.4% | 35.1% |
| have | 17,870 | 9.1% | 44.1% |
| apply | 14,995 | 7.6% | 51.8% |
| intro | 11,994 | 6.1% | 57.9% |
| case | 4,562 | 2.3% | 60.2% |
| . | 4,400 | 2.2% | 62.4% |
| cases | 3,943 | 2.0% | 64.4% |
| rcases | 3,739 | 1.9% | 66.3% |
| refine | 3,629 | 1.8% | 68.2% |
| obtain | 3,502 | 1.8% | 69.9% |
| constructor | 2,848 | 1.4% | 71.4% |
| simpa | 2,470 | 1.3% | 72.6% |
| induction | 2,295 | 1.2% | 73.8% |
| unfold | 1,905 | 1.0% | 74.8% |
| linarith | 1,878 | 1.0% | 75.7% |
| use | 1,825 | 0.9% | 76.7% |
| rintro | 1,825 | 0.9% | 77.6% |
| let | 1,740 | 0.9% | 78.5% |

Top 5 cover 52%. Top 46 cover 90%. Much flatter distribution than Goedel — no single tactic dominates.

Notable tactics absent from Goedel: `calc` (951), `convert` (983), `suffices` (728), `specialize` (849), `congr` (759), `bound` (690), `generalize` (588), `funext` (582).

### Single-Tactic Proof Breakdown (2,711 theorems, 14.4%)

| Tactic | Count | % of single-tactic |
|--------|-------|--------------------|
| simp | 793 | 29.3% |
| rw | 695 | 25.6% |
| cases | 126 | 4.6% |
| ext | 103 | 3.8% |
| simpa | 94 | 3.5% |
| induction | 88 | 3.2% |
| bv_decide | 81 | 3.0% |
| apply | 59 | 2.2% |
| exact | 59 | 2.2% |

No `nlinarith` dominance (unlike Goedel's 53%). Single-tactic proofs use standard simplification (`simp`, `rw`) reflecting library-style lemmas.

## Depth-Stratified Tactic Diversity

| Depth Bucket | Theorems | Steps | Unique Heads | Entropy (bits) | Top Tactic |
|--------------|----------|-------|-------------|----------------|------------|
| 0 | 2,816 | 2,933 | 142 | 3.91 | simp (28%) |
| 1 | 3,772 | 7,682 | 247 | 4.61 | simp (23%) |
| 2-3 | 3,163 | 11,722 | 258 | 4.81 | rw (15%) |
| 4-6 | 2,573 | 15,704 | 261 | 5.01 | exact (12%) |
| 7-10 | 2,080 | 19,484 | 268 | 5.03 | rw (13%) |
| 11+ | 4,230 | 116,789 | 543 | 5.06 | rw (12%) |

**Entropy stays high across all depths** (3.9–5.1 bits), unlike Goedel where it drops to 2.6 at depth 7+. Deep LEAN-GitHub proofs use a rich vocabulary, not just repetitive `have` chains.

## Top Repos (by theorem count)

| Repository | Theorems |
|------------|----------|
| apnelson1/Matroid | 1,416 |
| iehality/lean4-logic | 1,406 |
| girving/ray | 969 |
| iehality/Arithmetization | 808 |
| leanprover/leansat | 723 |
| AntoineChambert-Loir/DividedPowers4 | 622 |
| AlexKontorovich/PrimeNumberTheoremAnd | 615 |
| lecopivo/SciLean | 605 |
| Junology/algdata | 504 |
| leanprover-community/flt-regular | 471 |

Diverse sources: matroid theory, logic, ray tracing, arithmetic, SAT, divided powers, PNT, scientific computing, algebra.

## Lemma References (Top 15 of 18,690 unique)

| Lemma/Identifier | Count |
|------------------|-------|
| Or.inr | 1,224 |
| Or.inl | 1,192 |
| And.intro | 780 |
| Exists.intro | 764 |
| fst.comp | 634 |
| Eq.symm | 489 |
| Rel.symm | 483 |
| Function.updateITE | 472 |
| Function.comp | 459 |
| Nat.add_comm | 421 |
| Nat.succ | 414 |
| Rel.refl | 383 |
| Function.comp_apply | 333 |
| List.length_cons | 326 |
| Nat.succ_eq_add_one | 324 |

Core logic constructors (`Or.inl/inr`, `And.intro`, `Exists.intro`) dominate — structural proof building. Contrast with Goedel's `sub_nonneg.mpr` / `Real.sqrt` arithmetic focus.

## State and Tactic Size

| Metric | State (chars) | Tactic (chars) |
|--------|---------------|----------------|
| Mean | 394 | 51 |
| Median | 276 | 27 |
| p90 | 838 | 97 |
| p95 | 1,125 | 155 |
| p99 | 1,911 | 415 |
| Max | 4,090 | 11,150 |

States are 2.5x larger than Goedel (more hypotheses, more complex types). Tactics are shorter (less `nlinarith`-with-hints verbosity).

### Hypothesis Count (root goals)

| Metric | Value |
|--------|-------|
| Mean | 7.4 |
| Median | 6 |
| p90 | 15 |
| p95 | 20 |
| Max | 128 |

Nearly 2x the hypotheses of Goedel (mean 7.4 vs 3.8). Reflects real library theorems with richer contexts.
