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
