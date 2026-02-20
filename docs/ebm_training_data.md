# EBM Training Data Analysis

Generated: 2026-02-20

## Overview

EBM training (iter 3) uses 8 trajectory files from iterations 0-2, totaling **650,024 records** across **217,563 unique proof states** and **15,426 unique theorems**.

## File Summary

| File | Records | Positive | Negative | Pos% | Theorems | Proved |
|------|--------:|----------:|---------:|-----:|---------:|-------:|
| iter_0.parquet | 3,830 | 324 | 3,506 | 8.5% | 1,799 | 120 |
| iter_0_noisy.parquet | 4,619 | 575 | 4,044 | 12.4% | 1,799 | 175 |
| iter_1.parquet | 140,910 | 3,570 | 137,340 | 2.5% | 3,598 | 1,281 |
| iter_1_harvest.parquet | 37,861 | 1,460 | 36,401 | 3.9% | 1,797 | 614 |
| iter_1_negatives.parquet | 127,182 | 106,328 | 20,854 | 83.6% | 4,352 | 4,352 |
| iter_2.parquet | 250,275 | 4,177 | 246,098 | 1.7% | 8,848 | 1,749 |
| iter_2_broken_ebm.parquet | 80,314 | 1,310 | 79,004 | 1.6% | 3,019 | 566 |
| iter_2_search.parquet | 5,033 | 106 | 4,927 | 2.1% | 219 | 44 |
| **Total** | **650,024** | **117,850** | **532,174** | **18.1%** | **15,426** | **7,277** |

## Label Distribution

- **Positive (on proof path):** 117,850 (18.1%)
- **Negative (dead-end/off-path):** 532,174 (81.9%)
- **Neg:Pos ratio:** 4.5:1 overall

Note: `iter_1_negatives.parquet` is the outlier — 83.6% positive — because it comes from the `generate-negatives` pipeline which records ground-truth proof steps upfront (positive labels) alongside divergent tactics (negative labels). Without this file, the ratio is 11,522 pos / 511,320 neg = 2.2% positive.

## Positive State Difficulty (by remaining_depth)

`remaining_depth` = number of tactic steps remaining to QED on the proof path. Only meaningful for positive states (negatives have `remaining_depth = -1`).

**Excluding iter_1_negatives** (which has 72,895 positives with remaining_depth=-1 due to generate-negatives pipeline not tracking full proof paths):

| Difficulty | remaining_depth | Count | % of graded positives |
|-----------|----------------|------:|-----:|
| Trivial | 0 (next step = QED) | 4,549 | 10.1% |
| Easy | 1-2 | 16,583 | 36.9% |
| Medium | 3-5 | 13,813 | 30.7% |
| Hard | 6-10 | 6,858 | 15.3% |
| Very Hard | 11+ | 3,152 | 7.0% |

```
rem= 0:   4,549  ███
rem= 1:  10,102  ██████▌
rem= 2:   6,481  ████
rem= 3:   6,543  ████▎
rem= 4:   4,282  ██▊
rem= 5:   2,988  █▉
rem= 6:   2,193  █▍
rem= 7:   1,645  █
rem= 8:   1,262  ▊
rem= 9:     997  ▋
rem=10:     761  ▌
rem=11:     588  ▍
rem=12:     469  ▎
rem=13:     383  ▎
rem=14:     307  ▏
rem=15+:  1,405  ▉
```

Distribution is heavily right-skewed — most positives are near QED. Deep proof paths (10+ steps) are rare but present up to remaining_depth=54.

## Depth from Root Distribution

### Positive states
Peak at depth 2-3 (early proof steps), long tail to depth 54.

```
depth= 0:  10,811  ████████████▌
depth= 1:  10,507  ████████████▏
depth= 2:  21,095  ████████████████████████▌  (peak)
depth= 3:  19,564  ██████████████████████▋
depth= 4:  13,585  ███████████████▊
depth= 5:   9,810  ███████████▍
depth= 6:   7,300  ████████▍
depth= 7:   5,563  ██████▍
depth= 8:   4,308  █████
depth= 9:   3,285  ███▊
depth=10+: 12,022  ██████████████
```

### Negative states
Peak at depth 1-2, concentrated in shallow exploration.

```
depth= 0:  16,530  ████████
depth= 1:  78,493  ██████████████████████████████████████▋
depth= 2: 101,560  ████████████████████████████████████████████████▌ (peak)
depth= 3:  70,310  █████████████████████████████████▋
depth= 4:  74,580  ███████████████████████████████████▋
depth= 5:  53,749  █████████████████████████▊
depth= 6:  40,654  ███████████████████▍
depth= 7:  30,298  ██████████████▌
depth= 8:  22,490  ██████████▊
depth= 9:  15,286  ███████▎
depth=10+: 28,224  █████████████▌
```

## Contrastive Sampling Analysis

The ContrastiveSampler uses a 3-tier negative sampling strategy (30/40/30):
- **Hard negatives (30%):** Sibling negatives — share a `parent_state_id` with a positive (diverge at one tactic choice). These are the most informative examples for contrastive learning.
- **Medium negatives (40%):** Same-theorem negatives that are NOT siblings of any positive. Same proof context but from a different branch.
- **Easy negatives (30%):** Random cross-theorem states.

When a tier is exhausted for a given theorem, shortfall overflows into the next tier (hard → medium → easy → all_negatives).

### Hard Negative Availability

| Metric | Value |
|--------|------:|
| Eligible theorems (>=1 positive) | 7,277 |
| With hard negatives available | 7,276 (100.0%) |
| Without hard negatives | 1 (0.0%) |

Hard negatives per theorem (among those with any):

| Statistic | Count |
|-----------|------:|
| Min | 2 |
| P25 | 3 |
| Median | 6 |
| Mean | 19.6 |
| P75 | 13 |
| Max | 550 |

Most theorems have a modest number of hard negatives (median 6). A few large search trees produce hundreds.

## State Overlap

| Metric | Count |
|--------|------:|
| Unique positive states | 65,357 |
| Unique negative states | 162,660 |
| States in **both** pos and neg | 10,454 |
| Total unique states | 217,563 |

10,454 states (4.8% of total) appear as both positive and negative across different theorems or proof attempts. This creates contradictory training signal for those states — the EBM must learn theorem-conditioned scoring rather than absolute state quality.

## Per-File Characteristics

### iter_0.parquet / iter_0_noisy.parquet (iteration 0 search)
- Base model search, 1,799 theorems attempted
- Low prove rate: 120/1,799 (6.7%) base, 175/1,799 (9.7%) noisy
- Very few negatives per proved theorem (0.3-0.5 avg) — small search trees
- Noisy variant (temperature=1.2) proves more theorems but with more exploration

### iter_1.parquet (iteration 1 search)
- Largest search file: 140K records but only 2.5% positive
- 1,281/3,598 theorems proved (35.6%)
- Avg 4.0 negatives per proved theorem — deeper search trees
- Good source of hard negatives

### iter_1_harvest.parquet (iteration 1 harvest search)
- Secondary search pass: 614/1,797 proved (34.2%)
- 2.8 neg/pos ratio — moderate search trees

### iter_1_negatives.parquet (generate-negatives pipeline)
- **Fundamentally different structure:** mostly ground-truth proof steps (positive)
- 4,352 theorems, all with positives
- 72,895 positive states have remaining_depth=-1 (proof path not tracked)
- Primary purpose: provide depth-balanced positive training signal + hard divergent negatives
- Largest source of eligible theorems (4,352)

### iter_2.parquet (iteration 2 search)
- Largest single file: 250K records
- 8,848 theorems attempted, 1,749 proved (19.8%)
- 98.3% negative — very large search trees with many dead ends
- 2.7 neg/pos avg ratio

### iter_2_broken_ebm.parquet (iteration 2 with broken EBM)
- Search run where EBM was misconfigured/broken
- 566/3,019 proved (18.7%)
- Still useful: negative states from broken EBM guidance are informative failures

### iter_2_search.parquet (small iteration 2 search)
- Small supplementary search: 219 theorems, 44 proved
- Similar distribution to main iter_2

## Sibling-Based Hardness Analysis

True difficulty for contrastive learning isn't just about proof depth — it's about how similar a negative state is to a positive one. The hardest negatives are **siblings**: states that share the same parent but diverge at a single tactic choice. Only the positive sibling's tactic leads toward QED; the negative sibling's tactic leads to a dead end, but their states may be nearly identical.

### Hard Pair Counts (sibling negatives)

| File | Total Neg | Sibling-of-Pos | % Hard |
|------|----------:|---------------:|-------:|
| iter_0.parquet | 3,506 | 111 | 3.2% |
| iter_0_noisy.parquet | 4,044 | 259 | 6.4% |
| iter_1.parquet | 137,340 | 12,398 | 9.0% |
| iter_1_harvest.parquet | 36,401 | 3,880 | 10.7% |
| iter_1_negatives.parquet | 20,854 | 290 | 1.4% |
| iter_2.parquet | 246,098 | 11,160 | 4.5% |
| iter_2_broken_ebm.parquet | 79,004 | 3,374 | 4.3% |
| iter_2_search.parquet | 4,927 | 277 | 5.6% |
| **Total** | **532,174** | **31,749** | **6.0%** |

6% of all negatives are siblings of a positive — the rest are from unrelated branches or failed theorems entirely.

### State Text Similarity (pos vs neg sibling)

How similar are the resulting proof states when one tactic succeeds and the sibling fails?

| Similarity Range | Count | % | Description |
|-----------------|------:|--:|-------------|
| 0.95-1.00 | 9,154 | 28.8% | Near identical (incl. 7,863 both-empty) |
| 0.80-0.95 | 3,986 | 12.6% | Very similar |
| 0.50-0.80 | 4,742 | 14.9% | Moderately similar |
| 0.20-0.50 | 1,866 | 5.9% | Somewhat different |
| 0.00-0.20 | 12,001 | 37.8% | Very different (one empty, one not) |

**Key insight:** 41.4% of hard pairs have similarity >= 0.80. These are the states where the EBM must learn extremely subtle distinctions — the proof states look almost identical but one leads to QED and the other doesn't.

**Both-empty pairs:** 7,863 (24.8%) of hard pairs have both pos and neg states empty (`is_proof_complete = true`). Both tactics complete the proof, but only one is on the labeled proof path. This is a **labeling artifact** — both are valid proofs, yet the EBM is trained to prefer one over the other. These pairs add noise to training.

### Divergence Depth

At what depth in the proof tree do hard pairs diverge?

```
depth= 1:  18,882  ██████████████████████████████████████████████████ (59.5%)
depth= 2:   4,503  ███████████ (14.2%)
depth= 3:   3,063  ████████ (9.6%)
depth= 4:   2,366  ██████ (7.5%)
depth= 5:   1,282  ███ (4.0%)
depth= 6:     676  █▊
depth= 7:     386  █
depth= 8+:    591  █▌
```

59.5% of hard pairs diverge at depth 1 — the very first tactic choice. The EBM is mostly learning "which opening tactic is correct?" rather than deep proof planning.

### Tactic Divergence Patterns

Only 4.1% of hard pairs use the same tactic family (same first token). Top divergences:

```
        simp vs exfalso     3,388
        simp vs norm_num    3,376
        simp vs simp_all    3,319
    simp_all vs exfalso     1,358
    simp_all vs norm_num    1,341
       intro vs norm_num    1,233
       intro vs exfalso     1,230
        simp vs intro       1,147
       intro vs intro       1,014   (same family, different args)
        simp vs ring          645
```

`simp` is the dominant positive tactic — it appears on one side of 38% of all hard pairs. The EBM largely learns "simp works here, other tactics don't."

## Key Observations

1. **Severe class imbalance** (18% pos / 82% neg) is by design — search explores many dead ends. The contrastive loss (InfoNCE or margin ranking) handles this via negative sampling.

2. **iter_1_negatives dominates positive count** — 106,328 of 117,850 total positives (90.2%). Without it, there are only 11,522 positives from actual search trajectories.

3. **Difficulty skew toward easy** — 47% of graded positives are trivial/easy (remaining_depth 0-2). The EBM may learn to identify near-QED states well but struggle with deeper proof planning.

4. **Hard negative quality varies by iteration** — iter_0 has very few (0.3/theorem), while iter_1/2 have richer search trees (2.7-4.0/theorem).

5. **State overlap (10,454 dual-label states)** means the EBM cannot rely on state text alone — it must learn contextual provability signals from the embedding space.

6. **Sibling hardness is shallow** — 59.5% of hard pairs diverge at depth 1. The EBM is primarily learning first-tactic selection, not deep proof planning. This may limit its value for guiding multi-step search.

7. **Both-empty noise** — 24.8% of hard pairs are both valid proofs (empty goal states). Training the EBM to prefer one over the other is pure noise. Consider filtering these or labeling both as positive.

8. **simp dominance** — `simp` appears in 38% of hard pairs as the positive tactic. The EBM may overfit to "simp usually works" rather than learning structural proof state features.

9. **Scaling across iterations**: data grows substantially each iteration:
   - iter_0: 8,449 records
   - iter_1: 305,953 records (36x)
   - iter_2: 335,622 records (1.1x)
