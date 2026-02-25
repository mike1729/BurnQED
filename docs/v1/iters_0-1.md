# Expert Iteration Experiment Log — Iterations 0-1

**Date:** 2026-02-14
**Hardware:** 1× A100 40GB, Ubuntu
**Base model:** DeepSeek-Prover-V2-7B (SGLang, fp16)
**Search config:** max_nodes=100, max_depth=50, num_candidates=4 (search) / 8 (data collection) / 16 (eval), timeout=120s

## Baseline (Raw Model)

| Benchmark | Result | Details |
|-----------|--------|---------|
| B1: Pipeline validation (test_theorems) | 5/16 (31.2%) | 10-node budget, trivial theorems |
| B2: miniF2F (budget 600) | 15/244 (6.1%) | 16 candidates, avg 1.2 nodes, 12.6s/theorem |
| B3: theorem_index search | 77/2000 (4.3%) | 8 candidates, 201 start_proof errors |
| B3b: Baseline EBM training | Completed | 2000 steps on baseline trajectories |

## Iteration 0 — LoRA Fine-tuning on Mathlib Tactic Pairs

### Training (Step 1)

- **Method:** QLoRA (4-bit NF4, LoRA rank 16, alpha 32)
- **Data:** 246,526 Mathlib tactic pairs + 12,970 validation
- **Hyperparameters:** max_steps=1500, LR=2e-4 (cosine + 100 warmup), batch=8, grad_accum=4, max_seq_len=1024
- **Trainable params:** 37.5M / 3.9B (0.96%)
- **Duration:** ~2h48m (training) + ~11m (export)
- **Attention:** SDPA (flash-attn compilation OOM on instance)

**Loss progression:**

| Checkpoint | Train Loss | Eval Loss | Gap |
|-----------|------------|-----------|-----|
| Step 500 | ~0.50 | 0.512 | 0.01 |
| Step 1000 | ~0.43 | 0.439 | 0.01 |
| Step 1500 (final) | ~0.40 | 0.419 | 0.02 |

No overfitting — train/eval gap stayed minimal throughout. Loss decreased steadily.

### Search (Steps 3-5)

**Step 3 — Proof search (theorem_index, 2000 theorems, 8 candidates):**

| Metric | Baseline | Iter 0 | Change |
|--------|----------|--------|--------|
| Proved | 77/2000 (4.3%) | 120/2000 (6.7%) | +56% relative |
| Avg nodes/theorem | 1.3 | 1.7 | +31% |
| Avg time/theorem | 1.2s | 1.5s | +25% |
| Total wall time | — | 2738s (~46 min) | — |
| Errors (start_proof) | 201 | 201 | Same |

**Step 3b — Noise injection search (T=1.2, same 2000 theorems):**

| Metric | Normal (T=0.8) | Noisy (T=1.2) |
|--------|---------------|---------------|
| Proved | 120/2000 (6.7%) | 175/2000 (9.7%) |
| Avg nodes/theorem | 1.7 | 2.1 |
| Total wall time | 2738s | 3551s |
| Trajectory records | 3,830 | 4,619 |

Higher temperature discovers more diverse proofs — valuable for training data diversity.

**Step 4 — miniF2F evaluation (budget 600, 16 candidates):**

| Benchmark | Baseline | Iter 0 | Change |
|-----------|----------|--------|--------|
| miniF2F (budget 600) | 15/244 (6.1%) | **76/244 (31.1%)** | **+5.1× improvement** |

This is the headline result: a single LoRA fine-tuning pass on Mathlib tactic pairs yields a 5× improvement on miniF2F.

### Trajectory Summary

- Total records: 3,830 (normal) + 4,619 (noisy) = 8,449
- Positive records: 324 (normal) + 575 (noisy) = 899
- After filtering (non-empty state+tactic): 309 usable trajectory examples

## Iteration 1 — Expert Iteration with Trajectory Data

### Training (Step 1)

Based on iter 0 observations, several training adjustments were made:

1. **Reduced max_steps: 1500 → 800.** Iter 0 loss plateaued around step 1200.
2. **Base data subsampling (10K).** Model already learned from full 246K base in iter 0.
3. **Trajectory upsampling (×10).** 309 unique trajectory examples × 10 = ~3,090 copies. ~23% of training mix.
4. **LR halved: 2e-4 → 1e-4.** Prevents overwriting learned representations.

**Training configuration:**
- Base: `checkpoints/llm/iter_0` (warm LoRA adapter)
- Data: 10K base subsample + ~3K trajectory (×10 upsample) = ~13K total
- 800 steps × 32 effective batch = ~25K samples = ~2 passes through dataset

### Search (Steps 2-5)

- EBM trained on iter 0 trajectories
- Proof search with EBM guidance on 2000 theorems
- Generate-negatives pipeline for contrastive training data (see `docs/ebm_overhaul.md`)
- 127K records generated including probe tactics + zombie walk

### Trajectory Data Collected

| File | Records | Notes |
|------|---------|-------|
| `iter_0.parquet` | 3,830 | Normal search (T=0.8) |
| `iter_0_noisy.parquet` | 4,619 | Noise injection (T=1.2) |
| `iter_1.parquet` | ~100K+ | 2000-theorem search |
| `iter_1_harvest.parquet` | ~15K | Harvested proof paths |
| `iter_1_negatives.parquet` | 127K | Contrastive negatives (probes + LLM candidates) |
| **Total** | **314,402** | 112,257 positive, 202,145 negative |

## Iteration 2 — Scaling Up (In Progress)

### LLM Training (Step 1)

- **Base:** Merged iter 1 model (`models/llm/iter_1`)
- **Checkpoint selection:** 3 checkpoints evaluated (1000, 1500, 2000 steps). `full_eval_loss` plateaued at step 1500 (0.4148), while `traj_eval_loss` kept decreasing — sign of overfitting to trajectory distribution. **Selected: checkpoint-1500.**
- **Export:** Merged safetensors to `models/llm/iter_2` (6 shards, 25.75 GB)

### Theorem Selection

15,000 theorems selected from 61,542 total, excluding:
- 21,776 depth≤1 theorems (from tactic_pairs metadata)
- 3,164 probe-easy theorems (solved by built-in tactics only)
- 3,600 already-searched theorems (from iter 0-1 runs)
- Remaining 36,488 eligible → random 15K sample (seed=42)
- Stored in `data/iter2_search_theorems.json`

### EBM Training (Step 2)

- **Data:** 314,402 records across 5 trajectory files (76MB total)
- **States to encode:** 112,254 unique (after dedup)
- **Network:** 11M params (4096→2048→1024→512→1)
- **Training:** 1500 steps × 128 batch = 192K samples (~1.5 effective epochs)
- **Resume:** Fresh start (EBM_RESUME=none — new model means incompatible embeddings)
- **Encoding:** Checkpoint saves every 20K states for crash resilience

### SGLang `--is-embedding` Discovery

The `/encode` endpoint requires `--is-embedding` flag. Without it, SGLang returns 400
and encoding falls back to slow `/generate` + `return_hidden_states` (~1.2 states/sec).
With `--is-embedding`, encoding runs at ~86 states/sec (batch_size=64, concurrency=2).

**Stability issues:** SGLang with `--is-embedding` on a generative model (DeepSeek-Prover-V2-7B)
can freeze after processing ~8-28K states. Health check returns 503, GPU shows 0% utilization.
Mitigated by periodic checkpoint saves (every 20K states) + auto-resume on re-run.

**Batch response format:** SGLang native embedding mode returns `[{"embedding": [...]}, ...]`
(top-level array), not `{"embeddings": [...]}` (object wrapper). Code updated to handle both.

## Key Findings So Far

1. **Fine-tuning works dramatically.** 1500 steps of QLoRA on Mathlib tactic pairs yields 5× improvement on miniF2F (6.1% → 31.1%).

2. **No overfitting risk with LoRA.** 37M trainable params on 246K examples — train/eval gap stayed at ~0.02 throughout.

3. **Noise injection finds more proofs.** T=1.2 proved 175 vs 120 at T=0.8 — 46% more theorems. Worth the extra search time for training data diversity.

4. **201 start_proof errors are persistent.** These are Pantograph/Lean environment failures for specific theorems, not model failures. Same count in baseline and iter 0.

5. **Most proofs are 1-step.** Avg nodes ~1.7 — the model mostly succeeds with `simp`, `omega`, or direct lemma application. Multi-step proofs are rare but do occur (up to 10 nodes).

## Candidate Diversity & EBM Effectiveness

**Problem:** At T=0.6 with 8 candidates, dedup leaves only 2-3 unique tactics per expansion. The EBM scorer can't add much value when there's almost nothing to rank.

**Evidence from iter 0:**
- Regular search (T=0.6, 8 candidates): 120/2000 proved, avg 1.7 nodes
- Noisy search (T=1.2, 8 candidates): 175/2000 proved, avg 2.1 nodes — **46% more proofs**
- Higher temperature produces more diverse candidates, enabling more productive search

**Mitigations applied:**
- When `--ebm-path` is set, defaults auto-override to T=1.0 and 8 candidates (CLI flags still take priority)
- `candidates_per_expansion` stats (min/avg/max) now printed in search summary for monitoring
- Noise injection search at T=1.2 continues for training data diversity

**Alternative models to consider if diversity remains insufficient:**
- **DeepSeek-R1-Distill-Qwen-7B:** Strong reasoning capabilities, often outperforms specialized provers on complex logic. Not theorem-proving specific but may produce more diverse tactics.
- **Goedel-Prover-V2 (8B):** High efficiency, more diverse search behaviors than DeepSeek-Prover-V2-7B. Recently released, designed for formal math proving.

Both are SGLang-compatible — switching requires only changing `LLM_BASE` env var and possibly adjusting the prompt format in `SglangClient::format_prompt()`.

**Decision point:** After iter 1 search with EBM, check `Candidates/expand` stats. If avg unique candidates < 4, strong signal to try an alternative backbone.

## Infrastructure Notes

- **PYTHONUNBUFFERED=1** required in training scripts — otherwise `accelerate launch` buffers stdout and loss logs are invisible when piped through `tee`.
- **Pantograph ready-line timeout** increased from 30s → 120s — Mathlib import can take 60-90s on cold start.
- **SGLang must be stopped** before training (both compete for GPU memory).
- **Lean workers: 6** with concurrency 6 — enough to overlap Lean verification with GPU generation on single A100.
