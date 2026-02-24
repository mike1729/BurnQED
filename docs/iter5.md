# Iteration 5 — Failed LLM Fine-tuning

**Date:** 2026-02-24
**Verdict:** LLM fine-tuning unsuccessful. Reverted to iter_4 model for EBM training.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `models/llm/iter_4` (merged iter_3 r32) |
| LoRA mode | fresh (new adapter on merged base) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA MLP | yes |
| Learning rate | 6.25e-6 |
| Training steps | 2000 |
| Epochs completed | 1.31 |
| Training examples | 133,186 |
| Trainable params | 74.96M / 3.95B total |
| Runtime | 2h 41m |
| Final train loss | 0.291 |

## Evaluation Results

### Train-eval (100 Mathlib theorems, budget 300)

| Checkpoint | Proved | Rate |
|-----------|--------|------|
| Pre-train (iter_4) | 34/100 | **34.0%** |
| Post-train (iter_5) | 30/100 | **30.0%** |
| **Delta** | -4 | **-4.0%** |

### Separation Probe (200 pos / 200 neg)

| Metric | iter_4 | iter_5 | Delta |
|--------|--------|--------|-------|
| centroid_l2 | **6.404** | 1.328 | -5.076 |
| linear_probe_acc | **0.828** | 0.683 | -0.145 |
| norm_gap | **4.294** | 0.686 | -3.608 |
| pos_norm | 10.362 | 14.523 | +4.161 |
| neg_norm | 14.656 | 15.209 | +0.553 |
| delta_cosine | -0.004 | 0.001 | +0.005 |

### miniF2F v2s (partial — interrupted at 145/244, budget 600, no EBM)

Partial results before interruption: 38 proved, 6 fail, 87 timeout, 14 errors at 145/244.

## Analysis

The fresh LoRA training **destroyed the embedding separation** built up in the base model
through prior iterations. Key findings:

1. **Centroid L2 collapsed**: 6.40 → 1.33 (80% reduction). Positive and negative proof
   state embeddings are no longer well-separated in the representation space.

2. **Linear probe accuracy dropped**: 83% → 68%. The model's ability to distinguish
   provable from unprovable states in embedding space significantly degraded.

3. **Norm gap collapsed**: 4.29 → 0.69. The norm-based signal that previously helped
   distinguish proof states was nearly eliminated.

4. **Proof rate regressed**: 34% → 30% on train-eval (100 Mathlib theorems).

The fresh LoRA likely disrupted the fine-grained representation structure in the frozen
base while optimizing for next-token prediction on tactic sequences. The tactic generation
loss (0.291) was reasonable, but the representation quality (which the EBM depends on)
was harmed.

## EBM Training

The EBM embedding encoding (403,446 states) and initial EBM training (50K steps) completed
before the fine-tuning failure was confirmed:

- **Encoding:** 403,446 states encoded (~4h), 1,732 OOM errors (0.4%)
- **EBM training:** 50K steps, final train loss 1.176, val loss 1.218, gap 2.56, rank 0.64
- **However:** These embeddings were computed from the iter_5 (damaged) model, so the EBM
  checkpoint at `checkpoints/ebm/iter_5` is not useful.

## Resolution

Reverted to iter_4 model:
- `models/llm/iter_5` → symlink to `models/llm/iter_4`
- Retrain EBM using iter_4 model embeddings with improved training settings

## EBM Retraining (iter_4 model, improved settings)

### Iter 4 EBM analysis

Iter 4 EBM plateaued at rank=0.66 train / 0.60 val. Root cause: with K=7 and
hard_ratio=0.3, only ~2 of 7 negatives per sample are hard siblings. The 5
easy/medium negatives are trivially separated, but InfoNCE still computes
gradient for them, diluting the signal needed to crack the hard siblings.

```
Iter 4 EBM metrics (50K steps, info_nce, K=7, hard_ratio=0.3):
Step  | Train Loss | Val Loss | Train Gap | Val Gap | Train Rank | Val Rank
    0 |   4.42     |   3.72   |  -0.14    |  -0.07  |   0.11     |  0.07
  2K  |   1.66     |   1.56   |   1.16    |   1.58  |   0.49     |  0.55
 10K  |   1.28     |   1.34   |   2.15    |   2.07  |   0.61     |  0.58
 30K  |   1.12     |   1.25   |   2.89    |   2.80  |   0.65     |  0.60
 48K  |   1.07     |   1.32   |   3.11    |   2.92  |   0.66     |  0.59
```

No overfitting (val loss stable 1.19-1.36), but rank saturated by step 8K.

### Changes for iter 5 EBM

| Setting | Iter 4 | Iter 5 | Rationale |
|---------|--------|--------|-----------|
| Loss | info_nce | **margin_ranking** | Zeroes out easy neg gradient (loss=0 once margin satisfied), concentrating 100% of backprop on hard siblings |
| hard_ratio | 0.3 | **0.6** | 4 of 7 negatives are hard siblings (doubled density of failing signal) |
| medium_ratio | 0.4 | **0.3** | Less wasted capacity on medium negatives |
| dropout | 0.1 | **0.15** | Hard negs are only ~6% of records — with 60% sampling, same states repeat often; higher dropout prevents memorization |
| Steps | 50K | **75K** | 1.9x more data (1.26M rows vs 650K), need more steps for same epoch coverage |
| margin | 1.0 | 1.0 | Same |
| K | 7 | 7 | Same |
| LR | 3e-5 | 3e-5 | Same |
| Batch size | 256 | 256 | Same |

Key insight: margin ranking loss acts as an automatic hard-negative miner — once
an easy negative is pushed `margin` units away, its loss and gradient become
exactly zero. All gradient concentrates on the hard cases the model is actually
failing on.

### Code changes

These parameters were previously hardcoded in the pipeline or left at defaults
baked into the sampler/model constructors, meaning every experiment required
editing Rust source and recompiling. Exposing them as CLI args (and shell env
vars) enables rapid hyperparameter sweeps — critical because the iter 4 plateau
showed that the default ratio/dropout values are suboptimal, and we need to
iterate on these settings without code changes.

Added `--hard-ratio`, `--medium-ratio`, `--dropout` as CLI args to `train-ebm`,
wired through the pipeline to `ContrastiveSampler::with_ratios()` and
`EnergyHeadConfig::with_dropout()`. Both methods already existed but were
previously hardcoded.

**Files modified:**
- `crates/prover-core/src/main.rs` — three new `#[arg]` fields on `TrainEbm`
  variant, wired in the match arm
- `crates/prover-core/src/pipeline.rs` — three new fields on `TrainEbmArgs`;
  `with_dropout(args.dropout)` on `EnergyHeadConfig`; `.with_ratios()` chained
  on both train and val `ContrastiveSampler`
- `scripts/run_ebm_train.sh` — `HARD_RATIO`, `MEDIUM_RATIO`, `DROPOUT` env vars
  with defaults (0.3, 0.4, 0.1), passed as `--hard-ratio`, `--medium-ratio`,
  `--dropout` to the `train-ebm` invocation

### Encoding

- 403,446 unique states from 9 trajectory files (iters 0-4)
- 217,539 warm-cached from iter_4 EBM (same model), 185,907 new
- Uses nf4 encode server (~7GB VRAM) instead of direct bf16 loading

### Training command

```bash
EBM_STEPS=75000 LOSS_TYPE=margin_ranking \
HARD_RATIO=0.6 MEDIUM_RATIO=0.3 DROPOUT=0.15 \
START_STEP=2 bash scripts/run_ebm_train.sh 5
```

### Bug fixes applied before training

1. **Temperature removed from margin loss** — `forward()` returns raw energy;
   `temperature_scale()` only called in InfoNCE path. Temperature with margin
   lets optimizer collapse τ→0 to trivially zero the hinge loss.
2. **NaN guard** — skip optimizer step when loss is NaN/Inf to prevent permanent
   weight corruption.
3. **Flat-buffer EmbeddingCache** — `HashMap<String, u32>` index + contiguous
   `Vec<f32>` buffer eliminates ~3.5GB overhead at 403K entries.

### EBM training results

```
Step  | Train Loss | Val Loss | Train Gap | Val Gap | Train Rank | Val Rank | Pair  | Active
    0 |   1.03     |   1.02   |  -0.00    |  -0.00  |   0.14     |  0.07    | 0.50  | 1.00
  2K  |   0.89     |   0.90   |   0.79    |   1.12  |   0.48     |  0.54    | 0.80  | 0.61
  4K  |   0.90     |   0.89   |   1.18    |   1.22  |   0.60     |  0.57    | 0.87  | 0.43
  8K  |   0.90     |   0.90   |   1.32    |   1.31  |   0.63     |  0.57    | 0.88  | 0.39
 10K  |   0.90     |   0.90   |   1.37    |   1.27  |   0.63     |  0.57    | 0.88  | 0.38
 20K  |   0.90     |   0.91   |   1.54    |   1.44  |   0.65     |  0.59    | 0.89  | 0.35
 30K  |   0.90     |   0.91   |   1.62    |   1.51  |   0.66     |  0.60    | 0.89  | 0.34
 40K  |   0.90     |   1.00   |   1.69    |   1.67  |   0.67     |  0.59    | 0.89  | 0.32
 50K  |   0.90     |   0.97   |   1.74    |   1.68  |   0.67     |  0.59    | 0.90  | 0.32
 60K  |   0.90     |   0.99   |   1.77    |   1.69  |   0.68     |  0.59    | 0.90  | 0.31
 70K  |   0.90     |   0.99   |   1.79    |   1.70  |   0.68     |  0.59    | 0.90  | 0.31
 75K  |   0.90     |   0.99   |   1.79    |   1.71  |   0.68     |  0.59    | 0.90  | 0.31
```

**Final unbiased validation** (124,672 samples, hard_ratio=0.1, medium_ratio=0.3):
- Rank accuracy: **0.626**
- Pairwise accuracy: **0.896**
- Energy gap: **2.08**
- Loss: 1.024
- Avg pos energy: 0.20, avg neg energy: 2.28, energy std: 1.42
- 0 NaN/Inf steps skipped, runtime: 39 min

**Comparison with iter 4 EBM** (info_nce, K=7, hard_ratio=0.3, 50K steps):

| Metric | Iter 4 (50K) | Iter 5 (75K) | Delta |
|---|---|---|---|
| Train rank | 0.66 | **0.68** | +0.02 |
| Val rank | 0.59 | **0.63** (final: 0.63) | +0.04 |
| Train gap | 3.11 | 1.79 | -1.32 |
| Val gap | 2.92 | 2.08 | -0.84 |
| Train loss | 1.07 | 0.90 | -0.17 |
| Val loss | 1.32 | 1.02 | -0.30 |
| Pairwise | — | **0.90** | new metric |

Note: gaps are not directly comparable between InfoNCE (temperature-scaled) and
margin ranking (raw energy). Rank accuracy and pairwise accuracy are comparable.
Train rank improved 0.66→0.68, val rank 0.59→0.63. Modest improvement despite
doubling hard_ratio and switching loss function. The rank saturation around 0.68
suggests the frozen encoder's embedding quality is the binding constraint, not the
EBM architecture or loss function — consistent with the sibling collision analysis
showing 14% of pairs in the borderline zone.

## Data Analysis

### Math branch distribution (proved theorems)

9,342 proved theorems out of 61,233 in the theorem index (15.3% coverage).
Top-level Mathlib modules:

| Branch | Count | % |
|---|---|---|
| CategoryTheory | 582 | 6.2% |
| MeasureTheory | 442 | 4.7% |
| Polynomial | 359 | 3.8% |
| List | 323 | 3.5% |
| Nat | 283 | 3.0% |
| Finset | 264 | 2.8% |
| Set | 257 | 2.8% |
| Real | 179 | 1.9% |
| Matrix | 176 | 1.9% |
| SimpleGraph | 159 | 1.7% |
| Equiv | 116 | 1.2% |
| LinearMap | 102 | 1.1% |
| Multiset | 98 | 1.0% |
| Filter | 96 | 1.0% |

1,803 unique top-level branches (very long tail).
Top sub-branches: CategoryTheory.Limits (153), MeasureTheory.Measure (87),
Equiv.Perm (82), FirstOrder.Language (65), CategoryTheory.ShortComplex (46).

### Proved theorems per trajectory file

| File | Proved |
|---|---|
| iter_0.parquet | 120 |
| iter_0_noisy.parquet | 175 |
| iter_1.parquet | 1,281 |
| iter_1_harvest.parquet | 614 |
| iter_1_negatives.parquet | 4,352 |
| iter_2.parquet | 1,749 |
| iter_2_broken_ebm.parquet | 566 |
| iter_2_search.parquet | 44 |
| iter_4.parquet | 2,065 |

### Depth distribution

**Remaining depth** (50,829 positive states with known depth):

| Remaining | Count | % | Cumulative |
|---|---|---|---|
| 0 | 6,614 | 13.0% | 13.0% |
| 1 | 12,167 | 23.9% | 36.9% |
| 2 | 7,035 | 13.8% | 50.8% |
| 3 | 7,015 | 13.8% | 64.6% |
| 4 | 4,598 | 9.0% | 73.6% |
| 5 | 3,167 | 6.2% | 79.9% |
| 6–10 | 7,075 | 13.9% | 93.8% |
| 11–20 | 2,690 | 5.3% | 99.1% |
| 21+ | 468 | 0.9% | 100% |

74% within 5 steps of completion. Peak at remaining=1 (24%). 96% of all records
have unknown depth (-1, negative/failed states).

**Depth from root** (123,724 positive states):

| Depth | Count | % |
|---|---|---|
| 0 | 12,876 | 10.4% |
| 1 | 12,572 | 10.2% |
| 2 | 21,649 | 17.5% |
| 3 | 20,036 | 16.2% |
| 4 | 13,901 | 11.2% |
| 5 | 9,989 | 8.1% |
| 6–10 | 23,157 | 18.7% |
| 11–20 | 8,272 | 6.7% |
| 21+ | 1,272 | 1.0% |

Most proof-tree activity at depths 2–5 (53%). Max observed: 54.

## Sibling Collision Analysis

Measured embedding separation between sibling proof states (positive and negative
states sharing the same parent_state_id) to assess whether the 7B encoder provides
sufficient signal for the EBM to distinguish them.

**Dataset:** 28,644 sibling (pos, neg) pairs from trajectory data, sampled 20,000.

### Test 1: Sibling L2 Distance

| Metric | Sibling pairs | Random pos-neg |
|---|---|---|
| Mean L2 | **3.14** | 4.65 |
| Median L2 | 2.83 | 4.11 |
| Std | 1.98 | 1.89 |
| Cosine sim | 0.981 | — |

L2 threshold analysis:
- < 0.5 (collision): **3.7%**
- < 1.0 (borderline): **13.9%**
- < 2.0 (separable): 31.2%
- ≥ 2.0 (healthy): **68.8%**

Siblings are 1.52 L2 units closer than random pairs on average. ~14% of sibling
pairs are in the borderline zone (L2 < 1.0), meaning the EBM must work with very
small embedding differences for roughly 1 in 7 hard cases.

### Test 3: SVD Rank Collapse Probe

SVD on 10,000 sibling difference vectors (pos − neg) to check if LoRA r=32
creates a rank bottleneck in the embedding space.

| Dims | Cumulative variance |
|---|---|
| 1 | 64.5% |
| 2 | 75.6% |
| 4 | 81.0% |
| 8 | 86.2% |
| 16 | 90.3% |
| **32** | **93.5%** |
| 64 | 96.2% |
| 100 | ~98% |

S[32]/S[33] = **1.01** — no cliff at rank 32. The spectrum decays smoothly,
confirming **LoRA rank is NOT the bottleneck**. Difference vectors span a smooth
~64–100 dimensional subspace.

Key finding: **64.5% of variance in a single dimension.** There is one dominant
direction separating positive from negative siblings. The EBM primarily needs to
learn this direction plus a handful of corrections.

### Conclusions

1. **No LoRA rank bottleneck.** The embedding space is not rank-deficient at 32.
   Increasing LoRA rank would not meaningfully improve embedding separation.
2. **14% borderline pairs are the hard cases.** These are siblings with L2 < 1.0
   where the encoder provides minimal signal. The EBM's hard-negative mining
   (hard_ratio=0.6) specifically targets these.
3. **Low-dimensional structure is exploitable.** With 93.5% of variance in 32 dims,
   the EBM's 4096→2048→1024→512→1 architecture has more than enough capacity.
   The challenge is not representation power but gradient signal on the 14% hard cases.
