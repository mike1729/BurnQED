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

### EBM Rank-1 by Depth Bucket

| Depth | Iter 1 | Iter 2 | Iter 4 | Iter 5 |
|-------|--------|--------|--------|--------|
| 0-1   | 1.0%   | 0.7%   | 5.2%   | 0.0%   |
| 2-3   | 13.0%  | 18.4%  | 14.1%  | **22.5%** |
| 4-7   | 17.6%  | 25.9%  | 18.1%  | **31.2%** |
| 8+    | 21.1%  | 31.6%  | 21.5%  | **32.3%** |

### EBM Rank-1 by Depth (non-zero only, excluding terminal 0.0 bug)

| Depth | Iter 1 | Iter 2 | Iter 4 | Iter 5 |
|-------|--------|--------|--------|--------|
| 2-3   | 20.1%  | 30.8%  | 21.6%  | **36.7%** |
| 4-7   | 21.9%  | 32.4%  | 22.2%  | **39.4%** |
| 8+    | 23.6%  | 35.3%  | 23.3%  | **35.5%** |

The EBM is not pure noise — it improves with depth and iter 5 is the best at deep proofs (35-39% non-zero at depth 4+, above ~27% random baseline). At depth 0-1 it reads 0-5% because those decisions involve proof-completing nodes that fall back to `ebm_score=0.0`.

## Key Findings

### 1. Aggregate numbers are misleading — control for depth

Both LLM and EBM aggregate rank-1 numbers are dominated by the depth distribution of proved theorems. Iter 2's 58% LLM rank-1 is inflated by 86% depth 0-1 proofs; iter 5's 29% EBM rank-1 is depressed by only 17% depth 0-1.

### 2. LLM signal is stable at depth 4+

When controlling for depth, all iterations cluster at 27-32% for depth 4-7. The LLM provides a consistent ~30% rank-1 signal for medium-to-deep proofs across all iterations. The negative log-prob separation (pos mean < neg mean) reflects that correct tactics for hard theorems tend to be less probable — the LLM is less confident but the ranking still works.

### 3. EBM improves with iteration and depth

Contrary to the aggregate picture, depth-controlled EBM rank-1 shows clear improvement:
- Iter 1/4 cluster at 21-24% (non-zero, depth 4+)
- Iter 2/5 reach 32-39% (non-zero, depth 4+)
- The EBM is most useful precisely where search is hardest (deep proofs)

### 4. Terminal node score bug masks EBM value

All proof-completing nodes get `ebm_score = 0.0` (encode fallback). This has two effects:
- At depth 0-1 (where most decisions involve a proof-completing child), the EBM reads 0-5% — appearing useless
- The aggregate EBM rank-1 is dragged down by the large fraction of shallow proofs in earlier iterations
- Fixing this bug would likely improve both the aggregate numbers and the search itself (proof-completing states should rank highest, not middle)

### 5. Combined scoring hurts due to weight calibration

At every iteration, the combined score (alpha=0.5, beta=0.5) underperforms LLM-only. This is likely a calibration issue — the EBM and LLM scores are on different scales. With proper normalization or learned weights, combining a 30% LLM signal with a 35-39% EBM signal (at depth 4+) could outperform either alone.

## Methodology

For each proved theorem:
1. Identify all positive-labeled nodes (proof path) and negative-labeled nodes (dead ends)
2. Group children by parent state ID
3. At each decision point (parent with >1 child, where the correct child is on the proof path), rank siblings by LLM log-prob (higher = better) and EBM energy (lower = better)
4. Record whether the correct child is ranked #1

Data sources:
- `trajectories/iter_0.parquet` through `trajectories/iter_5_run1.parquet`
- `trajectories/iter_5_eligible.parquet` (proved theorems only, same results as iter_5_run1)
