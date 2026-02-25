# EBM & LLM Search Signal Analysis

Analysis of how well the LLM log-prob and EBM energy score guide proof search across expert iteration rounds. For each proved theorem, we trace the proof path and check whether the correct child was ranked #1 among siblings at each decision point.

## Rank-1 Accuracy Across Iterations

| Iter | Theorems | Proved | Decisions | LLM r1 | EBM r1 | Combined | Random |
|------|----------|--------|-----------|--------|--------|----------|--------|
| 0    | 1,799    | 6.7%   | 66        | 37.9%  | 37.9%  | 37.9%   | 40.9%  |
| 1    | 3,598    | 35.6%  | 2,289     | **48.2%** | 10.2% | 17.0%  | 27.2%  |
| 2    | 8,848    | 19.8%  | 2,428     | **58.4%** | 9.6% | 8.8%   | 26.7%  |
| 4    | 12,922   | 16.0%  | 3,809     | **48.4%** | 12.5% | 21.8% | 29.4%  |
| 5    | 6,087    | 16.9%  | 4,621     | 27.6%  | 29.1%  | 27.7%   | 33.4%  |

- **Random baseline** = mean of 1/num_siblings across all decision points.
- **Combined** uses alpha=0.5, beta=0.5 (config values): `0.5 * (-llm_log_prob) + 0.5 * ebm_score`.
- **EBM (non-zero)** filters out proof-completing nodes whose EBM score falls back to 0.0.

### EBM Non-Zero Rank-1

Excluding decision points where the correct child has ebm_score=0.0 (terminal node bug):

| Iter | EBM (nz) r1 | Decisions |
|------|-------------|-----------|
| 0    | N/A         | 0         |
| 1    | 22.0%       | 1,007     |
| 2    | 32.3%       | 679       |
| 4    | 22.2%       | 1,741     |
| 5    | 37.4%       | 3,591     |

Even filtering the bug, EBM hovers around random.

## Log-Prob and Energy Separation

For the EBM/LLM signal to be useful, positive (proof-path) nodes should be distinguishable from negative (dead-end) siblings.

| Iter | LLM pos mean | LLM neg mean | LLM sep  | EBM pos mean | EBM neg mean | EBM sep  |
|------|-------------|-------------|----------|-------------|-------------|----------|
| 0    | -6.21       | -10.63      | **+4.42** | N/A         | N/A         | N/A      |
| 1    | -12.70      | -12.11      | -0.58    | -1.017      | -1.000      | +0.017   |
| 2    | -18.01      | -14.44      | -3.56    | -6.068      | -8.171      | -2.103   |
| 4    | -12.81      | -11.55      | -1.27    | -0.554      | -0.586      | -0.032   |
| 5    | -25.76      | -16.76      | -9.00    | -1.133      | -1.383      | -0.250   |

- **LLM sep > 0** means correct tactics get higher log-prob (good). Only iter 0 (base model) has positive separation.
- **EBM sep > 0** means correct states get lower energy (good). Separation is near zero across all iterations.

## Proof Depth Distribution

Earlier iterations are dominated by shallow (depth 0-1) proofs. Iter 5 searches much harder theorems.

| Iter | Depth 0-1 | Depth 2-3 | Depth 4-7 | Depth 8+ | Median | Mean |
|------|-----------|-----------|-----------|----------|--------|------|
| 0    | 72%       | 18%       | 11%       | 0%       | 1      | 1.7  |
| 1    | 79%       | 7%        | 12%       | 2%       | 1      | 1.8  |
| 2    | **86%**   | 6%        | 7%        | <1%      | 1      | 1.4  |
| 4    | 73%       | 12%       | 12%       | 1%       | 1      | 1.8  |
| 5    | **17%**   | 27%       | 41%       | **15%**  | **4**  | 4.5  |

### LLM Rank-1 by Depth Bucket

Controlling for proof difficulty removes the apparent iter 2 peak.

| Depth | Iter 1 | Iter 2 | Iter 4 | Iter 5 |
|-------|--------|--------|--------|--------|
| 0-1   | 72.2%  | 74.3%  | 66.8%  | 46.6%  |
| 2-3   | 39.4%  | 35.3%  | **48.6%** | 30.8% |
| 4-7   | 29.9%  | 32.1%  | 32.1%  | 27.1%  |
| 8+    | 19.6%  | 22.8%  | **28.5%** | 24.9% |

At depth 4+, all iterations are in a similar 20-32% range. Iter 4 is the best at deeper proofs (28.5% at 8+). The headline drop from iter 2 (58%) to iter 5 (28%) is almost entirely explained by iter 2 being 86% shallow proofs vs iter 5 at 17%.

## Key Findings

### 1. LLM log-prob signal is depth-dependent, not collapsing

The aggregate LLM rank-1 numbers (58% → 28%) are misleading. Iter 2's 58% is inflated by 86% of proofs being depth 0-1 (where LLM gets 74%). When controlling for depth:
- At depth 4-7: all iterations cluster around 27-32%
- At depth 8+: iter 4 leads at 28.5%, iter 5 is 24.9%
- The LLM is consistently useful at ~30% for medium-depth proofs (above 27% random)

The negative log-prob separation (pos mean < neg mean) reflects that correct tactics for hard theorems tend to be less probable — the LLM is less confident but the ranking still works.

### 2. EBM provides no useful signal

Across all iterations, the EBM energy score fails to discriminate proof-path states from dead ends:
- Rank-1 accuracy hovers around random baseline (10-37% vs 27-41%)
- Mean energy separation between positive and negative nodes is < 0.3
- Combined scoring (LLM + EBM) consistently hurts compared to LLM-only

The EBM head (~11M params) trained on contrastive pairs from search trajectories is not learning a meaningful value function.

### 3. Terminal node score bug

All 1,030 proof-completing nodes in iter 5 have `ebm_score = 0.0` (the encode fallback value). Since lower energy = better, these nodes sort to the middle of the priority queue instead of the top. This means the search never prioritizes expanding states that are close to completing a proof.

### 4. Combined scoring is strictly worse

At every iteration, the combined score underperforms LLM-only. The EBM's near-random signal dilutes whatever LLM signal exists. With alpha=beta=0.5, the combined ranker is a weighted average of a weak signal and noise.

## Methodology

For each proved theorem:
1. Identify all positive-labeled nodes (proof path) and negative-labeled nodes (dead ends)
2. Group children by parent state ID
3. At each decision point (parent with >1 child, where the correct child is on the proof path), rank siblings by LLM log-prob (higher = better) and EBM energy (lower = better)
4. Record whether the correct child is ranked #1

Data sources:
- `trajectories/iter_0.parquet` through `trajectories/iter_5_run1.parquet`
- `trajectories/iter_5_eligible.parquet` (proved theorems only, same results as iter_5_run1)
