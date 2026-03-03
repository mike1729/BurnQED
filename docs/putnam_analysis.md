# Putnam Trajectory Deep Analysis

**Dataset:** 604 theorems, 132 proved (21.9%), 169060 nodes (556 positive, 168504 negative)

## 1. Proof Path vs Dead Branch Tactic Distribution

Proof-path tactics: 424 | Dead-branch tactics: 9128

| Tactic      | Path # | Dead # | Path % | Dead % | Enrichment |
| :---------- | -----: | -----: | -----: | -----: | ---------: |
| exact?      |    111 |      8 |  26.2% |   0.1% |     298.7x |
| intro       |    106 |    612 |  25.0% |   6.7% |       3.7x |
| have        |     87 |   5557 |  20.5% |  60.9% |       0.3x |
| exact       |     24 |     27 |   5.7% |   0.3% |      19.1x |
| simp        |     14 |    335 |   3.3% |   3.7% |       0.9x |
| intros      |     12 |     93 |   2.8% |   1.0% |       2.8x |
| simp_all    |     11 |    745 |   2.6% |   8.2% |       0.3x |
| constructor |      9 |     92 |   2.1% |   1.0% |       2.1x |
| rw          |      9 |    189 |   2.1% |   2.1% |       1.0x |
| simpa       |      7 |      5 |   1.7% |   0.1% |      30.1x |
| aesop       |      7 |    173 |   1.7% |   1.9% |       0.9x |
| norm_num    |      5 |    205 |   1.2% |   2.2% |       0.5x |
| apply?      |      5 |      2 |   1.2% |   0.0% |      53.8x |
| omega       |      3 |      0 |   0.7% |   0.0% |       infx |
| dsimp       |      3 |     22 |   0.7% |   0.2% |       2.9x |
| use         |      3 |     71 |   0.7% |   0.8% |       0.9x |
| unfold      |      2 |     34 |   0.5% |   0.4% |       1.3x |
| apply       |      1 |     79 |   0.2% |   0.9% |       0.3x |
| rcases      |      1 |    225 |   0.2% |   2.5% |       0.1x |
| linarith    |      1 |     13 |   0.2% |   0.1% |       1.7x |


**Key findings:**
- **exact?**: 26.2% on proof paths vs 0.1% on dead branches (enrichment 299x — proof closer)
- **intro**: 25.0% on proof paths vs 6.7% on dead branches (enrichment 3.7x)
- **have**: 20.5% on proof paths vs 60.9% on dead branches (enrichment 0.34x — 3.0x overrepresented on dead branches)

### Proof-Path Tactic Bigrams

| Transition          | Count |
| :------------------ | ----: |
| intro → have        |    39 |
| intro → exact?      |    31 |
| have → exact?       |    29 |
| have → have         |    26 |
| have → exact        |    10 |
| intro → intro       |     9 |
| simp → exact?       |     8 |
| simp_all → exact?   |     7 |
| have → intro        |     7 |
| exact → exact       |     6 |
| intros → exact?     |     6 |
| aesop → exact?      |     6 |
| intro → simp_all    |     5 |
| constructor → intro |     5 |
| have → rw           |     4 |
| intros → have       |     4 |
| exact? → intro      |     4 |
| exact? → exact      |     3 |
| intro → simp        |     3 |
| have → aesop        |     3 |


## 2. Tactic Outcome at Sibling States

At 238 mixed branch points (both have and non-have children):
- **have/let won: 26.9%** (64/238), non-have won: **73.1%** (174/238)

**Per-tactic win rate at branch points** (min 5 occurrences):

| Tactic      | Wins | Total | Win Rate |
| :---------- | ---: | ----: | -------: |
| exact?      |   74 |    75 |    98.7% |
| constructor |    6 |    15 |    40.0% |
| intro       |   91 |   319 |    28.5% |
| dsimp       |    3 |    19 |    15.8% |
| intros      |   11 |    71 |    15.5% |
| haveI       |    1 |     7 |    14.3% |
| unfold      |    2 |    15 |    13.3% |
| aesop       |    7 |    62 |    11.3% |
| use         |    3 |    27 |    11.1% |
| rw          |    4 |    41 |     9.8% |
| simp        |   12 |   138 |     8.7% |
| norm_num    |    5 |    65 |     7.7% |
| have        |   80 |  1138 |     7.0% |
| apply       |    1 |    24 |     4.2% |
| simp_all    |    8 |   252 |     3.2% |
| rcases      |    1 |    48 |     2.1% |
| refine'     |    0 |    48 |     0.0% |
| rintro      |    0 |    50 |     0.0% |
| obtain      |    0 |     6 |     0.0% |
| let         |    0 |     5 |     0.0% |
| ext         |    0 |     5 |     0.0% |
| induction'  |    0 |    13 |     0.0% |
| cases'      |    0 |     5 |     0.0% |


## 3. Goal-Count Delta by Tactic Type

Delta = parent_goals - child_goals. Positive = goals decreased (closing). Negative = goals increased (opening).

### On proof paths only

| Tactic   | Count | Mean Δ | Closing % | Opening % | Neutral % |
| :------- | ----: | -----: | --------: | --------: | --------: |
| apply?   |     5 |  +1.00 |      100% |        0% |        0% |
| exact    |    24 |  +1.00 |      100% |        0% |        0% |
| exact?   |   111 |  +1.00 |      100% |        0% |        0% |
| omega    |     3 |  +1.00 |      100% |        0% |        0% |
| simpa    |     7 |  +1.00 |      100% |        0% |        0% |
| rw       |     9 |  +0.56 |       56% |        0% |       44% |
| simp_all |    11 |  +0.18 |       18% |        0% |       82% |
| intro    |   106 |  +0.03 |        3% |        0% |       97% |
| aesop    |     7 |  +0.00 |       14% |       14% |       71% |
| dsimp    |     3 |  +0.00 |        0% |        0% |      100% |
| intros   |    12 |  +0.00 |        0% |        0% |      100% |
| norm_num |     5 |  +0.00 |        0% |        0% |      100% |
| simp     |    14 |  +0.00 |        0% |        0% |      100% |
| use      |     3 |  +0.00 |        0% |        0% |      100% |
| have     |    87 |  -0.25 |        0% |       25% |       75% |


### On all nodes

| Tactic        | Count | Mean Δ | Closing % | Opening % | Neutral % |
| :------------ | ----: | -----: | --------: | --------: | --------: |
| .             |    10 |  +1.00 |      100% |        0% |        0% |
| assumption    |    36 |  +1.00 |      100% |        0% |        0% |
| case          |    92 |  +1.00 |      100% |        0% |        0% |
| contradiction |    56 |  +1.00 |      100% |        0% |        0% |
| linarith      |   256 |  +1.00 |      100% |        0% |        0% |
| nlinarith     |   112 |  +1.00 |      100% |        0% |        0% |
| omega         |    38 |  +1.00 |      100% |        0% |        0% |
| positivity    |    19 |  +1.00 |      100% |        0% |        0% |
| trivial       |    13 |  +1.00 |      100% |        0% |        0% |
| exact         |   818 |  +1.00 |      100% |        0% |        0% |
| exact?        |   168 |  +0.98 |       98% |        0% |        2% |
| tauto         |    23 |  +0.96 |       96% |        0% |        4% |
| simpa         |   114 |  +0.89 |       90% |        2% |        8% |
| rfl           |    59 |  +0.88 |       88% |        0% |       12% |
| ring          |    72 |  +0.60 |       60% |        0% |       40% |


## 4. State Size Trajectory

len(state_pp) by depth in proved theorems. Proof paths should stay compact; dead branches balloon.

| Depth | Path N | Path Mean | Path Med | Dead N | Dead Mean | Dead Med | P/D Ratio |
| :---- | -----: | --------: | -------: | -----: | --------: | -------: | --------: |
| 0     |    132 |       236 |      206 |      0 |         — |        — |         — |
| 1     |    132 |       224 |      202 |    722 |       377 |      285 |      0.59 |
| 2     |    113 |       217 |      181 |   1664 |       415 |      331 |      0.52 |
| 3     |     77 |       263 |      246 |   1953 |       459 |      363 |      0.57 |
| 4     |     53 |       210 |      157 |   1936 |       521 |      431 |      0.40 |
| 5     |     31 |       128 |        0 |   1180 |       575 |      485 |      0.22 |
| 6     |     11 |       144 |        0 |    802 |       663 |      542 |      0.22 |
| 7     |      4 |       140 |       68 |    464 |       753 |      588 |      0.19 |
| 8     |      2 |        46 |       46 |    213 |       851 |      641 |      0.05 |
| 9     |      1 |         0 |        0 |    115 |      1075 |      817 |         — |
| 10    |      0 |         — |        — |     49 |      1247 |      835 |         — |
| 11    |      0 |         — |        — |     21 |       889 |      743 |         — |


### Sample proof-path trajectories

- **putnam_1967_b3**: d0:(root)(255) → d1:intro(270) → d2:have(297) → d3:simp_all(282) → d4:exact?(0)
- **putnam_1968_a1**: d0:(root)(61) → d1:have(224) → d2:have(247) → d3:rw(247) → d4:have(324) → d5:linarith(149) → d6:exact(0)
- **putnam_1968_a2**: d0:(root)(158) → d1:have(435) → d2:intros(453) → d3:have(545) → d4:exact(290) → d5:exact(0)
- **putnam_1968_b2**: d0:(root)(127) → d1:intro(122) → d2:have(137) → d3:exact?(0)
- **putnam_1968_b1**: d0:(root)(358) → d1:intro(458) → d2:have(1151) → d3:simp(1133) → d4:exact?(691) → d5:exact(0)

## 5. Goedel Tactic Distribution Comparison

Parsed 5000 Goedel proofs, 22293 total tactic invocations.

### Overall Goedel tactic distribution (top 20)

| Tactic      | Count | Goedel % |
| :---------- | ----: | -------: |
| have        |  5070 |    22.7% |
| nlinarith   |  4284 |    19.2% |
| field_simp  |  1127 |     5.1% |
| ring_nf     |  1096 |     4.9% |
| simp        |   988 |     4.4% |
| linarith    |   897 |     4.0% |
| intro       |   854 |     3.8% |
| norm_num    |   797 |     3.6% |
| first       |   783 |     3.5% |
| rw          |   733 |     3.3% |
| apply       |   727 |     3.3% |
| try         |   714 |     3.2% |
| simp_all    |   615 |     2.8% |
| cases'      |   512 |     2.3% |
| ring        |   437 |     2.0% |
| rcases      |   386 |     1.7% |
| exact       |   339 |     1.5% |
| omega       |   334 |     1.5% |
| constructor |   323 |     1.4% |
| rfl         |   184 |     0.8% |


### Goedel closing tactics (last tactic per proof)

| Tactic    | Count |     % |
| :-------- | ----: | ----: |
| nlinarith |  2866 | 57.3% |
| linarith  |   524 | 10.5% |
| try       |   345 |  6.9% |
| omega     |   222 |  4.4% |
| ring      |   176 |  3.5% |
| exact     |   175 |  3.5% |
| norm_num  |   108 |  2.2% |
| simp      |    96 |  1.9% |
| all_goals |    69 |  1.4% |
| rfl       |    65 |  1.3% |
| apply     |    57 |  1.1% |
| simp_all  |    55 |  1.1% |
| decide    |    52 |  1.0% |
| rw        |    43 |  0.9% |
| ring_nf   |    39 |  0.8% |


### Cross-comparison: Goedel vs Putnam

| Tactic      | Goedel % | Putnam Path % | Putnam Dead % |
| :---------- | -------: | ------------: | ------------: |
| have        |    22.7% |         20.5% |         60.9% |
| nlinarith   |    19.2% |          0.0% |          0.1% |
| intro       |     3.8% |         25.0% |          6.7% |
| exact?      |     0.0% |         26.2% |          0.1% |
| field_simp  |     5.1% |          0.0% |          0.1% |
| simp_all    |     2.8% |          2.6% |          8.2% |
| linarith    |     4.0% |          0.2% |          0.1% |
| omega       |     1.5% |          0.7% |          0.0% |
| norm_num    |     3.6% |          1.2% |          2.2% |
| ring        |     2.0% |          0.0% |          0.0% |
| constructor |     1.4% |          2.1% |          1.0% |
| rcases      |     1.7% |          0.2% |          2.5% |
| exact       |     1.5% |          5.7% |          0.3% |
| simp        |     4.4% |          3.3% |          3.7% |
| rw          |     3.3% |          2.1% |          2.1% |


**Key insight:** Goedel have rate (22.7%) is much lower than Putnam dead-branch rate (60.9%). SFT on Goedel would naturally shift the model toward algebraic closers (nlinarith, linarith, field_simp) the base model rarely generates on Putnam.

## 6. Depth-Conditional Tactic Distribution

### Proof-path tactics by depth

- **Depth 1:** intro:73, exact?:15, intros:11, have:9, simp:7, aesop:4
- **Depth 2:** have:36, exact?:33, intro:19, simp_all:5, simp:3, constructor:3
- **Depth 3:** have:26, exact?:24, intro:7, simp_all:4, rw:4, simp:3
- **Depth 4:** exact?:25, have:11, exact:6, intro:3, rw:2, simp:1
- **Depth 5:** exact:9, exact?:9, intro:4, have:2, simpa:2, linarith:1
- **Depth 6:** exact?:5, have:3, exact:1, omega:1, simpa:1
- **Depth 7:** exact:3, constructor:1
- **Depth 8:** exact:1, omega:1
- **Depth 9:** ring_nf:1

### Proof-closing tactics by depth

- **Depth 1:** exact?:15, exact:2, rw:1, intro:1
- **Depth 2:** exact?:30, apply?:3, exact:2, simpa:1
- **Depth 3:** exact?:18, simpa:3, simp_all:1, rw:1, intro:1
- **Depth 4:** exact?:19, exact:1, simp_all:1, rw:1
- **Depth 5:** exact?:9, exact:8, simpa:2, rw:1
- **Depth 6:** exact?:5, exact:1, simpa:1
- **Depth 7:** exact:2
- **Depth 8:** exact:1
- **Depth 9:** ring_nf:1

## 7. Unique First-Tactic Type Distribution

Across 604 theorems with depth-1 nodes (4374 total):
- Median unique first tactics per theorem: **4**
- **have fraction at depth 1:** mean 25.1%, median 22.2%
  - Q25=0.0%, Q75=41.7%
  - 32.1% of theorems have **zero** have tactics at depth 1
  - Proved: 23.7% vs Failed: 25.5%

### Overall depth-1 tactic distribution

| Tactic      | Count |     % |
| :---------- | ----: | ----: |
| have        |  1192 | 27.3% |
| intro       |   985 | 22.5% |
| simp_all    |   562 | 12.8% |
| simp        |   328 |  7.5% |
| rintro      |   268 |  6.1% |
| aesop       |   236 |  5.4% |
| intros      |   228 |  5.2% |
| refine'     |   139 |  3.2% |
| norm_num    |    89 |  2.0% |
| dsimp       |    74 |  1.7% |
| constructor |    57 |  1.3% |
| unfold      |    39 |  0.9% |
| use         |    39 |  0.9% |
| apply       |    35 |  0.8% |
| ext         |    24 |  0.5% |


---

# Supporting Analysis

## B. Log-Prob Ranking Analysis

At 319 branch points with positive children:
- **Rank-1 accuracy:** 36.7% (log-prob correctly ranks proof-path child first)
- Mean positive rank: 2.6, median: 2.0

Note: `llm_log_prob` is whole-proof completion log-prob. All nodes from the same completion share the same value, so ranking within a completion is arbitrary.

| Rank | Count |
| :--- | ----: |
| 1    |   117 |
| 2    |    78 |
| 3    |    47 |
| 4    |    31 |
| 5    |    21 |
| 6    |     7 |
| 7    |     6 |
| 8    |     6 |


### Rank-1 accuracy by depth

| Depth | Rank-1 Rate | Mean Rank |   N |
| :---- | ----------: | --------: | --: |
| 1     |       34.4% |      2.47 | 128 |
| 2     |       36.5% |      2.89 | 104 |
| 3     |       44.9% |      2.35 |  49 |
| 4     |       25.9% |      2.48 |  27 |
| 5     |       50.0% |      3.40 |  10 |
| 6     |      100.0% |      1.00 |   1 |


### Rank-1 accuracy by sibling count

| Siblings | Rank-1 Rate | Mean Rank |   N |
| :------- | ----------: | --------: | --: |
| 11+      |       31.5% |      2.99 |  73 |
| 2-3      |       50.0% |      1.60 |  50 |
| 4-10     |       35.2% |      2.73 | 196 |


### Log-prob by depth (positive vs negative)

| Depth | Pos Mean | Neg Mean |  Delta | Pos N | Neg N |
| :---- | -------: | -------: | -----: | ----: | ----: |
| 1     |    -24.3 |    -42.2 |  +17.9 |   132 |   722 |
| 2     |    -22.9 |    -41.5 |  +18.7 |   113 |  1664 |
| 3     |    -20.7 |    -43.4 |  +22.7 |    77 |  1953 |
| 4     |    -26.1 |    -42.6 |  +16.4 |    53 |  1936 |
| 5     |   -226.3 |    -47.4 | -179.0 |    31 |  1180 |
| 6     |   -501.8 |    -40.1 | -461.7 |    11 |   802 |
| 7     |    -29.9 |    -44.2 |  +14.3 |     4 |   464 |
| 8     |    -36.3 |    -45.8 |   +9.5 |     2 |   213 |
| 9     |    -25.0 |    -36.0 |  +11.0 |     1 |   115 |
| 10    |        — |    -61.7 |      — |     0 |    49 |
| 11    |        — |    -40.4 |      — |     0 |    21 |
| 12    |        — |    -50.1 |      — |     0 |     8 |


### Log-prob distributions

```
Positive  n=424  mean=-50.8  med=-20  p25=-33  p75=-12  min=-6190  max=-2
Negative  n=9,128  mean=-43.1  med=-32  p25=-50  p75=-20  min=-7231  max=-2
```

### Completion deduplication

Nodes sharing (theorem, parent, log-prob) come from the same whole-proof completion.
- Unique completion groups: 9,552
- Total nodes: 9,552
- Mean nodes/group: 1.00, max: 1

## C. State Complexity Analysis

### Root state analysis (proved vs failed)

- Proved root state length: mean=236, median=206
- Failed root state length: mean=248, median=209
- Proved root hypotheses (mean): 0.0
- Failed root hypotheses (mean): 0.0

### Solve rate by topic

| Topic            | Proved | Total |  Rate |
| :--------------- | -----: | ----: | ----: |
| other            |      6 |    15 | 40.0% |
| linear_algebra   |     15 |    43 | 34.9% |
| combinatorics    |     46 |   187 | 24.6% |
| geometry         |     12 |    50 | 24.0% |
| number_theory    |     91 |   411 | 22.1% |
| series_sums      |     37 |   177 | 20.9% |
| real_arithmetic  |     55 |   264 | 20.8% |
| abstract_algebra |     12 |    59 | 20.3% |
| analysis         |     30 |   158 | 19.0% |


### Hypothesis and goal complexity

- Path mean hypothesis count: 3.3
- Dead mean hypothesis count: 7.6
- Path mean chars/goal: 188
- Dead mean chars/goal: 417

### Hypothesis count by depth

| Depth | Path Mean | Dead Mean |
| :---- | --------: | --------: |
| 1     |      3.21 |      2.14 |
| 2     |      3.51 |      5.14 |
| 3     |      4.03 |      6.61 |
| 4     |      3.47 |      8.18 |
| 5     |      2.06 |      8.92 |
| 6     |      2.18 |     10.33 |
| 7     |      3.25 |     11.94 |
| 8     |      2.00 |     13.41 |
| 9     |      0.00 |     15.83 |
| 10    |         — |     18.53 |
| 11    |         — |     15.48 |
| 12    |         — |     13.38 |


## D. Search Efficiency

- **Proved:** 132 theorems, mean 73 nodes (median 42)
- **Failed:** 472 theorems, mean 338 nodes
- Proof efficiency (positive/total in proved): 5.7%
- Mean max depth: proved 6.0, failed 11.2
- Have-tactic fraction on dead branches (proved theorems): 50.1%

### Time-to-proof

QED node position / total nodes per theorem (lower = found proof earlier).

```
QED position ratio  n=132  mean=0.9  med=1  p25=1  p75=1  min=0  max=1
```
- Found in first 10%: 0 theorems
- Found in first 25%: 2 theorems
- Found in first 50%: 6 theorems
- Found in last 25%:  119 theorems

### Fanout at branch points

Parent nodes with 2+ children: 17,330 / 86,942 (19.9%)

| Fanout | Count |
| :----- | ----: |
| 2      |  4599 |
| 3      |  2108 |
| 4      |  1661 |
| 5      |  1474 |
| 6      |  1367 |
| 7      |  1292 |
| 8      |  1098 |
| 9      |   957 |
| 10     |   757 |
| 11     |   546 |
| 12     |   422 |
| 13     |   322 |
| 14     |   218 |
| 15     |   163 |
| 16     |    99 |


## E. Failure Mode Classification

- **deep_exhaust:** 351 theorems
- **have_spiral:** 121 theorems
- **shallow_exhaust:** 0 theorems
- **low_diversity:** 0 theorems

### Max depth reached in failed theorems

| Max Depth | Count |
| :-------- | ----: |
| 5         |     2 |
| 6         |     6 |
| 7         |    25 |
| 8         |    53 |
| 9         |    81 |
| 10        |    79 |
| 11        |    71 |
| 12        |    51 |
| 13        |    27 |
| 14        |    19 |
| 15        |    11 |
| 16        |    14 |
| 17        |     5 |
| 18        |     7 |
| 19        |     7 |


### Near-miss detection

Failed theorems with high q_value or max_depth >= 8: 25

| Theorem        | Max Q | Max Depth | Nodes |
| :------------- | ----: | --------: | ----: |
| putnam_1974_a1 | 0.000 |        45 |   576 |
| putnam_2007_a2 | 0.000 |        42 |   205 |
| putnam_1966_b2 | 0.000 |        38 |   355 |
| putnam_1978_a5 | 0.000 |        30 |   602 |
| putnam_1975_a5 | 0.000 |        28 |   612 |
| putnam_1969_a6 | 0.000 |        26 |   676 |
| putnam_1966_a3 | 0.000 |        25 |   946 |
| putnam_1971_b3 | 0.000 |        23 |   790 |
| putnam_1977_b5 | 0.000 |        23 |   592 |
| putnam_1978_b4 | 0.000 |        23 |   478 |
| putnam_2021_a6 | 0.000 |        21 |   250 |
| putnam_1969_a4 | 0.000 |        20 |   206 |
| putnam_1974_b2 | 0.000 |        20 |   822 |
| putnam_1986_b3 | 0.000 |        20 |   667 |
| putnam_1963_a2 | 0.000 |        19 |   328 |


### Solve rate by year

Years with 0 solves (10): [1964, 1969, 1975, 1976, 1981, 1999, 2002, 2004, 2009, 2011]

| Year | Proved | Total | Rate |
| :--- | -----: | ----: | ---: |
| 1962 |      2 |    11 |  18% |
| 1963 |      1 |     9 |  11% |
| 1964 |      0 |    11 |   0% |
| 1965 |      1 |    11 |   9% |
| 1966 |      1 |    12 |   8% |
| 1967 |      2 |    12 |  17% |
| 1968 |      4 |    10 |  40% |
| 1969 |      0 |    10 |   0% |
| 1970 |      5 |    10 |  50% |
| 1971 |      2 |    10 |  20% |
| 1972 |      1 |     9 |  11% |
| 1973 |      2 |     8 |  25% |
| 1974 |      4 |    10 |  40% |
| 1975 |      0 |     8 |   0% |
| 1976 |      0 |     8 |   0% |
| 1977 |      4 |    10 |  40% |
| 1978 |      1 |    10 |  10% |
| 1979 |      1 |     9 |  11% |
| 1980 |      2 |    10 |  20% |
| 1981 |      0 |     6 |   0% |
| 1982 |      1 |     8 |  12% |
| 1983 |      3 |     9 |  33% |
| 1984 |      1 |     4 |  25% |
| 1985 |      1 |     9 |  11% |
| 1986 |      2 |     8 |  25% |
| 1987 |      3 |     9 |  33% |
| 1988 |      4 |    12 |  33% |
| 1989 |      3 |     6 |  50% |
| 1990 |      2 |    10 |  20% |
| 1991 |      3 |    10 |  30% |
| 1992 |      6 |    11 |  55% |
| 1993 |      1 |    10 |  10% |
| 1994 |      3 |    11 |  27% |
| 1995 |      3 |    10 |  30% |
| 1996 |      5 |    10 |  50% |
| 1997 |      3 |     6 |  50% |
| 1998 |      4 |     9 |  44% |
| 1999 |      0 |    10 |   0% |
| 2000 |      1 |     9 |  11% |
| 2001 |      1 |     7 |  14% |
| 2002 |      0 |     6 |   0% |
| 2003 |      3 |    11 |  27% |
| 2004 |      0 |     9 |   0% |
| 2005 |      2 |    10 |  20% |
| 2006 |      4 |    10 |  40% |
| 2007 |      2 |    10 |  20% |
| 2008 |      2 |     9 |  22% |
| 2009 |      0 |    10 |   0% |
| 2010 |      1 |    12 |   8% |
| 2011 |      0 |    12 |   0% |
| 2012 |      3 |    11 |  27% |
| 2013 |      1 |    10 |  10% |
| 2014 |      3 |    10 |  30% |
| 2015 |      5 |    12 |  42% |
| 2016 |      2 |    10 |  20% |
| 2017 |      2 |     9 |  22% |
| 2018 |      1 |     9 |  11% |
| 2019 |      3 |     8 |  38% |
| 2020 |      4 |     9 |  44% |
| 2021 |      1 |     9 |  11% |
| 2022 |      2 |    12 |  17% |
| 2023 |      3 |     8 |  38% |
| 2024 |      3 |     9 |  33% |
| 2025 |      2 |     7 |  29% |


## F. Training Signal Assessment

From 132 proved theorems:
- **Positive pairs** (proof-path tactics): 424
- **Hard negatives** (siblings of proof-path nodes): 2164
- **Easy negatives** (other dead-branch nodes): 6964

### Positive pair depth distribution

| Depth | Count |
| :---- | ----: |
| 1     |   132 |
| 2     |   113 |
| 3     |    77 |
| 4     |    53 |
| 5     |    31 |
| 6     |    11 |
| 7     |     4 |
| 8     |     2 |
| 9     |     1 |


### DPO preference pairs

- Total DPO pairs (winner-loser at branch points): 2,164
- Winner non-have, loser have: 570 (26.3%)
- Winner have, loser non-have: 223 (10.3%)

### Token budget

- Estimated p95 token count (state + tactic): 198
- Pairs exceeding 2048 tokens: 0 / 424
- Pairs exceeding 4096 tokens: 0 / 424

### exact? audit

- exact? on proof paths: 111 (96 are QED nodes)
- exact? accounts for 26.2% of positive pairs
- Theorems with multiple proofs: 5 (single: 127)

---

## G. Recommendations

### Tactic patterns to amplify in SFT

- **intro** (25% of proof paths, 3.7x enriched): strongest real proof-path signal
- **Goal-closing tactics** (exact, omega, ring, linarith, nlinarith): directly reduce goals rather than adding context
- **Goedel closers** (nlinarith 19.2%, field_simp 5.1%, linarith 4.0%): base model rarely generates these on Putnam
- **Proof-path bigrams**: reinforce common transition patterns as SFT signal

### What to filter from training data

- **have with goal increase** (25% of have on proof paths open new subgoals; 61% of dead branches are have)
- **exact?/decide closers**: defer to kernel search, don't teach tactical reasoning; keep for search (98.7% win rate) but consider filtering from SFT
- **States > 2048 tokens**: 0 of 424 pairs

### Critic (EBM) vs Generation (SFT) assessment

- **Log-prob rank-1 accuracy: 36.7%** — below 40%, strong case for Critic
- **State size divergence**: proof paths stay compact, dead branches balloon 4.5x by depth 5 — trivially learnable signal for EBM
- **Hypothesis accumulation**: dead branches accumulate 2.5x more hypotheses — another learnable signal
- **Counter-argument**: dominant failure (have-chains) is detectable by simple heuristics, reducing Critic's unique contribution

### SFT strategy

1. **SFT on Goedel traced pairs** — Goedel have rate (22.7%) is close to proof-path rate (20.5%); naturally shifts distribution without explicit penalty; introduces missing closers
2. **Mix 5-10% Putnam positive pairs** — Goedel has 0% exact? and low intro; Putnam pairs anchor these high-value tactics
3. **DPO on branch-point contrasts** — 2,164 preference pairs available, 570 directly train have-avoidance

### Estimated training data from these trajectories

- SFT positive pairs: 424
- Hard negatives (contrastive): 2164
- DPO preference pairs: 2,164
- Combine with Goedel ~30K pairs for volume; use Putnam trajectories for preference/contrastive signal
