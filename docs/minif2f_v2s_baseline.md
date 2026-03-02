# miniF2F iter_0 Baseline Evaluation

**Date:** 2026-03-02
**Model:** DeepSeek-Prover-V2-7B (base, no LoRA)
**Mode:** LLM-only (EBM scores all 0.0)
**Config:** `configs/search_minif2f.toml` — n=16 proofs, 30 rounds, 1800s timeout, fp8 quantization, temp=1.4

## Headline Numbers

| Split | Proved | Total | Rate |
|-------|--------|-------|------|
| v2s_test | 155 | 241 | **64.3%** |
| v2c_test | 150 | 240 | **62.5%** |
| Union | **167** | **242** | **69.0%** |
| Both | 138 | — | — |
| Only v2s | 17 | — | — |
| Only v2c | 12 | — | — |

### vs Prior Baseline (2026-02-28)

Old config: max_nodes=600, timeout=600s, 60 hybrid rounds, T=0.8, hybrid_max_tokens=512.
New config: n=16, 30 rounds, 1800s timeout, T=1.4, fp8, parallel sub-batches.

| Metric | Old | New (v2s) | Delta |
|--------|-----|-----------|-------|
| v2s rate | 148/244 (60.7%) | 155/241 (64.3%) | +3.6pp |
| Union rate | — | 167/242 (69.0%) | — |
| 1-node solves | 110/148 (74%) | — (hybrid-only) | — |

The +3.6pp improvement comes from higher temperature (1.4 vs 0.8) driving more exploration and the longer timeout (1800s vs 600s).

## Subject Breakdown

| Subject | v2s | v2c |
|---------|-----|-----|
| algebra (standalone) | 16/18 (89%) | 17/18 (94%) |
| induction | 7/8 (88%) | 7/8 (88%) |
| mathd_algebra | 55/70 (79%) | 56/70 (80%) |
| numbertheory | 6/8 (75%) | 5/8 (63%) |
| mathd_numbertheory | 44/60 (73%) | 44/60 (73%) |
| amc12 | 18/43 (42%) | 13/43 (30%) |
| imo | 7/20 (35%) | 5/19 (26%) |
| aime | 2/14 (14%) | 3/14 (21%) |

Strengths: algebra, induction, mathd (73-94%). Weaknesses: AIME (14-21%), IMO (26-35%), AMC12 (30-42%).

## Search Depth Profile

- Median proof depth: 3 (both splits)
- Mean proof depth: ~3.5
- 40-45 theorems solved at depth 1 (whole-proof hybrid mode)
- 10-13 theorems only solved at depth 1 — hybrid search is essential
- Deepest proofs: depth 22 (mathd_algebra_131), depth 21 (mathd_algebra_267, mathd_algebra_327)

### Proof depth distribution (v2s)

| Depth | Theorems proved |
|-------|-----------------|
| 1 | 40 |
| 2 | 21 |
| 3 | 27 |
| 4 | 19 |
| 5 | 10 |
| 6 | 12 |
| 7 | 3 |
| 8 | 6 |
| 9 | 3 |
| 10 | 4 |
| 11 | 1 |
| 12 | 5 |
| 13 | 3 |
| 15 | 1 |

### Node depth distribution (all nodes, v2s)

| Depth | Nodes | Branching factor |
|-------|-------|------------------|
| 0 | 241 | 5.0 |
| 1 | 1207 | 1.7 |
| 2 | 1999 | 1.5 |
| 3 | 2922 | 1.3 |
| 4 | 3914 | 1.0 |
| 5 | 3937 | 0.9 |
| 6 | 3705 | 0.9 |
| 7 | 3386 | 0.8 |
| 8 | 2725 | 0.7 |
| 9 | 1938 | 0.7 |
| 10+ | 4147 | — |

Branching factor 5.0 at root (depth 0->1 expansion via hybrid proofs), then rapidly decays below 1.0 by depth 5 — most branches die.

## Search Efficiency

- Median nodes to first proof: 8 (both splits)
- P90: ~63 nodes
- Max: ~290 nodes
- Proved theorems use median 24-25 nodes; unproved explore 255-266 nodes
- Total nodes explored: 30K (v2s), 31K (v2c)
- Avg nodes/theorem: 125 (v2s), 130 (v2c)

## Tactic Profile

| Tactic | % of all | Proof-path rate |
|--------|----------|-----------------|
| have | 50-53% | 1.0% |
| intro | 7-9% | 4-6% |
| simp_all | 6% | 2.0% |
| norm_num | 6% | 2.7-3.3% |
| simp | 4-5% | 2.9-3.2% |
| rw | 4% | 2.8-4.2% |
| exact | 0.7% | 15.6-15.9% |
| nlinarith | 0.6-0.7% | 13.5% |
| ring_nf | 1.1-1.4% | 4.3-5.3% |

`have` dominates (50-53%) but only 1% land on proof paths — the LLM introduces many intermediate lemmas that mostly don't pan out. Most effective tactics: `exact` (16%), `nlinarith` (14%).

## Proof Multiplicity

| Metric | v2s | v2c |
|--------|-----|-----|
| Mean proofs/theorem | 4.5 | 4.4 |
| Median | 3 | 3 |
| Max | 15 | 15 |
| 1 proof only | 41 | 41 |
| 2-5 proofs | 68 | 58 |
| 6-10 proofs | 29 | 42 |
| >10 proofs | 17 | 9 |

## Theorems Proved Only in One Split

### Only v2s (17)

amc12_2000_p11, amc12_2001_p9, amc12a_2010_p10, amc12a_2013_p8, amc12a_2017_p2, amc12b_2003_p6, imo_1961_p1, imo_1966_p5, imo_1974_p5, imo_1978_p5, imo_1984_p2, mathd_algebra_151, mathd_algebra_421, mathd_algebra_509, mathd_numbertheory_43, mathd_numbertheory_48, numbertheory_xsqpysqintdenomeq

### Only v2c (12)

aime_1994_p4, algebra_amgm_faxinrrp2msqrt2geq2mxm1div2x, amc12b_2004_p3, imo_1965_p1, imo_1966_p4, imo_1990_p3, mathd_algebra_22, mathd_algebra_224, mathd_algebra_480, mathd_algebra_59, mathd_numbertheory_136, mathd_numbertheory_32

## Hardest Unproved (75 theorems)

Top 15 by total nodes explored across both splits:

| Theorem | Nodes | Max depth |
|---------|-------|-----------|
| amc12a_2016_p2 | 1121 | 30 |
| mathd_algebra_149 | 1012 | 24 |
| mathd_numbertheory_13 | 979 | 14 |
| amc12a_2002_p1 | 947 | 25 |
| aimeI_2000_p7 | 895 | 16 |
| aime_1984_p5 | 865 | 23 |
| imo_1988_p6 | 849 | 13 |
| mathd_numbertheory_405 | 811 | 27 |
| amc12b_2002_p3 | 793 | 20 |
| mathd_algebra_482 | 791 | 20 |
| mathd_numbertheory_200 | 744 | 12 |
| amc12a_2019_p9 | 733 | 38 |
| aime_1988_p3 | 712 | 17 |
| amc12b_2002_p6 | 704 | 28 |
| amc12a_2009_p25 | 702 | 27 |

## Key Takeaways

1. **69% union rate with base model** is a strong LLM-only baseline for iter_0 (target was >=35%).
2. **29 theorems unique to one split** — running both v2s and v2c adds ~5pp over either alone.
3. **EBM opportunity:** 75 theorems failed with 641-1121 nodes each. EBM scoring should help prune the ~99% of `have` tactics that lead nowhere.
4. **Hybrid search is load-bearing:** 10-13 theorems exclusively proved at depth 1, not reachable by tactic search alone.
5. **No timing data** recorded (all timestamp deltas are 0) — the pipeline doesn't embed wall-clock timestamps in the parquet.
