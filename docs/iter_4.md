# Iteration 4 Training Report

## LLM Fine-tuning (LoRA)

**Base Model**: DeepSeek-Prover-V2-7B
**Output**: `models/llm/iter_3_r32` (symlinked as `models/llm/iter_4`)

### LoRA Config
| Param | Value |
|---|---|
| Rank (r) | 32 |
| Alpha | 64 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Dropout | 0.05 |
| Bias | None |
| Trainable params | 74.96M / 3.95B (1.9%) |

### Training Hyperparams
| Param | Value |
|---|---|
| Learning rate | 3e-5 (cosine decay) |
| Total steps | 12,000 |
| Epochs | ~5.1 (resumed from checkpoint-6000 at epoch 2.56) |
| Training examples | 131,623 |
| Throughput | 8.7 samples/s, 0.435 steps/s (~4.16 s/step) |
| Runtime | ~7.67 hours (resumed portion) |
| Final train loss | 0.1506 (smoothed over full run) |

### Training Data Sources
- Base Mathlib tactic pairs (subsampled to 50K)
- iter_0.parquet: 84 usable examples (324 positive / 3,830 total, 240 empty filtered)
- iter_0_noisy.parquet: 220 usable (575 positive / 4,619 total)
- iter_1.parquet: 1,001 usable (3,570 positive / 140,910 total)
- iter_1_harvest.parquet: 190 usable (1,460 positive / 37,861 total)
- iter_1_negatives.parquet: 88,366 usable (106,328 positive / 127,182 total, 17,962 deduped)
- iter_2.parquet: 667 usable (4,177 positive / 250,275 total)
- iter_2_broken_ebm.parquet: 150 usable (1,310 positive / 80,314 total)
- iter_2_search.parquet: 14 usable (106 positive / 5,033 total)
- **Total trajectory**: 90,692 unique examples (filtered: 9,100 empty, 10 sorry, 18,050 dup)
- **Trajectory split**: 81,623 train / 9,069 val
- **Combined training set**: 131,623 (50K base + 81,623 trajectory)

### Training Progress

**Train loss** (sampled every ~600 steps, logged every 10 steps):

| Step | Epoch | Train Loss | Grad Norm | Learning Rate |
|------|-------|------------|-----------|---------------|
| 6,010 | 2.56 | 0.3085 | 0.780 | 1.52e-05 |
| 6,600 | 2.81 | 0.3291 | 0.769 | 1.28e-05 |
| 7,200 | 3.07 | 0.3214 | 0.849 | 1.05e-05 |
| 7,800 | 3.32 | 0.3030 | 0.809 | 8.28e-06 |
| 8,400 | 3.58 | 0.2979 | 0.866 | 6.22e-06 |
| 9,000 | 3.84 | 0.2930 | 0.795 | 4.38e-06 |
| 9,600 | 4.10 | 0.3050 | 0.910 | 2.84e-06 |
| 10,200 | 4.36 | 0.2949 | 0.864 | 1.59e-06 |
| 10,800 | 4.61 | 0.2948 | 0.873 | 6.99e-07 |
| 11,400 | 4.87 | 0.2932 | 0.825 | 1.58e-07 |
| 11,940 | 5.10 | 0.2769 | 0.858 | 6.32e-11 |

**Average train loss by epoch**: Epoch 2: 0.3201, Epoch 3: 0.3067, Epoch 4: 0.2889, Epoch 5: 0.2853

**Eval loss** (500 examples, evaluated every ~100 steps):

| Epoch | Avg Eval Loss | Min Eval Loss |
|-------|---------------|---------------|
| 2 | 0.4914 | 0.4868 |
| 3 | 0.4783 | 0.4707 |
| 4 | 0.4722 | 0.4711 |
| 5 | 0.4715 | 0.4714 |

- **Best eval loss**: 0.4707 at epoch 3.96
- **Final full eval loss**: 0.4599 (full val set, 12,970 examples)
- **Final trajectory eval loss**: 0.2501 (trajectory val set only)

### Separation Probe (embedding quality monitor)

| Step | Epoch | Centroid L2 | Norm Gap | Pos Norm | Neg Norm |
|------|-------|-------------|----------|----------|----------|
| 7,000 | 2.98 | 5.904 | — | — | — |
| 9,000 | 3.83 | 6.487 | 4.346 | 10.500 | 14.846 |
| 10,000 | 4.25 | 6.817 | 4.455 | 10.092 | 14.547 |
| 11,000 | 4.68 | 6.844 | 4.476 | 10.082 | 14.558 |
| 12,000 | 5.10 | 6.851 | 4.477 | 10.067 | 14.544 |

Centroid L2 increased monotonically (5.9 → 6.9), indicating better separation between positive and negative state embeddings throughout training.

---

## Training Data (for EBM)

**Trajectories** (8 files, ~175 MB total):

| File | Size |
|---|---|
| iter_0.parquet | 665 KB |
| iter_0_noisy.parquet | 927 KB |
| iter_1.parquet | 35.3 MB |
| iter_1_harvest.parquet | 6.1 MB |
| iter_1_negatives.parquet | 36.7 MB |
| iter_2.parquet | 75 MB |
| iter_2_broken_ebm.parquet | 21.1 MB |
| iter_2_search.parquet | 653 KB |

**Contrastive Sampler Stats** (from 650,024 total records, 583,943 train / 66,081 val):

|  | Train | Val |
|---|---|---|
| Eligible positives (proof-path) | 6,539 | 738 |
| Sibling negatives | 31,049 | 3,182 |
| Non-sibling negatives | 447,480 | 50,397 |
| Total negatives | 478,529 | 53,579 |

**Embeddings**: 217,539 unique states (dim=4096), 24 uncached (~0.01%)

---

## EBM Training (fp16 corrected)

**Checkpoint**: `checkpoints/ebm/iter_4_fp16/`
**Embeddings**: fp16-derived (cosine sim 0.9988 with SGLang), stored as f64 in parquet, cast to f32 on load.

### EBM Hyperparams
| Param | Value |
|---|---|
| Loss | InfoNCE |
| Steps | 20,000 |
| Batch size | 256 |
| Learning rate | 3e-5 (cosine decay) |
| K negatives | 7 |
| Runtime | ~10 min (CUDA) |

### EBM Training Progress

| Step | Train Loss | Gap | Rank | Val Loss | Val Gap | Val Rank | LR |
|------|-----------|------|------|----------|---------|----------|----|
| 0 | 2.3953 | 0.25 | 0.18 | 2.1482 | 0.18 | 0.21 | 3.00e-8 |
| 2,000 | 1.4956 | 1.58 | 0.54 | 1.3104 | 2.05 | 0.58 | 2.98e-5 |
| 4,000 | 1.2798 | 2.11 | 0.62 | 1.3010 | 2.24 | 0.59 | 2.82e-5 |
| 6,000 | 1.2289 | 2.28 | 0.63 | 1.2296 | 2.20 | 0.62 | 2.52e-5 |
| 8,000 | 1.1938 | 2.40 | 0.64 | 1.1674 | 2.48 | 0.61 | 2.10e-5 |
| 10,000 | 1.1648 | 2.53 | 0.64 | 1.2468 | 2.45 | 0.60 | 1.62e-5 |
| 12,000 | 1.1402 | 2.63 | 0.65 | 1.3285 | 2.75 | 0.58 | 1.13e-5 |
| 14,000 | 1.1196 | 2.72 | 0.65 | 1.2959 | 2.67 | 0.60 | 6.80e-6 |
| 16,000 | 1.1057 | 2.77 | 0.65 | 1.2232 | 2.76 | 0.60 | 3.16e-6 |
| 18,000 | 1.0994 | 2.80 | 0.66 | 1.2366 | 2.68 | 0.60 | 8.13e-7 |

- **Best val loss**: 1.1674 at step 8,000
- Energy gap (pos−neg) widened from 0.25 → 2.80 during training
- Slight val loss increase after step 10K (overfitting), but gap/rank stable
- Positive energies converged to ~0.06, negative energies to ~2.86

---

## Search Config

```toml
[search]
max_nodes = 300
max_depth = 50
num_candidates = 8
alpha = 0.5
beta = 0.5
llm_temperature = 40.4       # ~90% EBM / 10% LLM
ebm_temperature = 1.0
timeout_per_theorem = 120
batch_expansion_size = 8
batch_encode_size = 8
harvest_siblings = true
```

### Probe Tactics (17)
simp, ring, omega, norm_num, decide, trivial, rfl, tauto, linarith, push_neg,
contradiction, exfalso, constructor, left, right, ext, simp_all

---

## Evaluation

### Eval dataset: `data/eval_clean_100.json`
100 Mathlib theorems sampled from `theorem_index.json` (seed 42), **disjoint from all training data**:
- Excluded 25,025 theorems appearing in any trajectory parquet (iter_0 through iter_2)
- Excluded 198 train_eval theorems
- Clean pool: 36,547 theorems
- Median hypothesis count: 6 (similar to train_eval distribution)

### Preliminary results (contaminated sample, `data/debug_sample_100.json`)

28/100 theorems overlapped with training data. Results on 95 theorems that ran:

| Config | Proved | Rate |
|--------|--------|------|
| LLM+EBM 50/50 (`llm_t=4.52`) | 34/95 | 35.8% |
| LLM+EBM 90% EBM (`llm_t=40.4`) | 40/95 | 42.1% |

90% EBM found 11 theorems the 50/50 missed, lost only 5. Net +6 theorems.

### Clean eval results (`data/eval_clean_100.json`)

83/100 theorems ran (17 failed elaboration — no compiled oleans).

| Config | Proved | Rate | Trajectory |
|--------|--------|------|------------|
| LLM-only (no EBM) | 39/83 | 47.0% | `trajectories/eval_clean_100_llm_only.parquet` |
| LLM+EBM 90% (`llm_t=40.4`) | 49/83 | **59.0%** | `trajectories/eval_clean_100_ebm90.parquet` |

**+10 theorems (+12pp), zero regressions.** Every LLM-only proof was also found by EBM.
Two EBM-exclusive proofs needed depth-8 search (270 and 181 nodes).
Median proof depth: 1 (both configs). Max: 6 (LLM-only), 8 (EBM).

### Historical Mathlib proof rates

| Run | Dataset | Proved | Rate |
|-----|---------|--------|------|
| iter_1, no EBM | theorem_index 500 | 18/500 | 3.6% |
| iter_1, with EBM | theorem_index 500 | 30/500 | 6.0% |
| iter_3, pre-train | train_eval 100 | 28/100 | 28.0% |
| iter_4, post-train | train_eval 198 | 65/198 | 32.8% |

---

## Known Issues & Fixes

### Embedding dtype mismatch
EBM was trained on bf16 embeddings (`encode_embeddings.py`), but SGLang sgl.Engine loads model in fp16. Mean cosine similarity between training and inference embeddings: **0.945** (should be ~1.0).

**Fix**: Re-encoding all 217K states with `encode_server.py --dtype float16`. Verified cosine similarity with SGLang: **0.9988**.

### Score scale mismatch
`alpha * llm_log_prob + beta * ebm_score` mixes signals on different scales. LLM log-probs range ~[-1, -10], EBM energies can be arbitrary.

**Fix**: Added `llm_temperature` / `ebm_temperature` config params. Score is now `alpha * (llm_log_prob / llm_temp) + beta * (ebm_score / ebm_temp)`.

### SGLang zero embeddings (#8066)
sgl.Engine intermittently returns zero hidden states (~10-18% failure rate), corrupting EBM scoring during eval.

**Fix**: Use standalone `encode_server.py` (HuggingFace transformers) instead of sgl.Engine for encoding. No zero-embedding bug.

---

## File Locations

| Artifact | Path |
|---|---|
| LLM model | `models/llm/iter_3_r32/` (symlinked as `iter_4`) |
| Embeddings (bf16, original) | `checkpoints/ebm/iter_4/embeddings.parquet` |
| Embeddings (fp16, corrected) | `checkpoints/ebm/iter_4/embeddings_fp16.parquet` |
| Trajectories | `trajectories/iter_{0,1,2}*.parquet` |
| Training logs | `logs/iter_4_*.log` |
| Search config | `configs/search.toml` |
