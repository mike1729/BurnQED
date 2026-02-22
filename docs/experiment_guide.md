# Experiment Execution Guide

## Quick Start

```bash
# 1. Data preparation (CPU, ~5 min download or hours with --trace)
./scripts/prepare_data.sh              # Downloads pre-traced data (default)
./scripts/prepare_data.sh --trace      # Local LeanDojo trace (optional)

# 2. Cloud bootstrap (on GPU instance)
bash scripts/setup_runpod.sh   # RunPod RTX 4090 (recommended)
bash scripts/setup_lambda.sh   # Lambda Labs A100

# 3. Run full experiment
NUM_WORKERS=64 ./scripts/run_all_iterations.sh
```

## Scripts

| Script | Purpose | GPU? |
|--------|---------|------|
| `scripts/setup_runpod.sh` | Setup for RunPod RTX 4090 (auto-detects Network Volume, VRAM) | No |
| `scripts/setup_lambda.sh` | Setup for Lambda Labs A100 | No |
| `scripts/migrate_to_runpod.sh` | Transfer checkpoints/trajectories from Lambda to RunPod | No |
| `scripts/prepare_data.sh` | Trace Mathlib + format tactic pairs + validate outputs | No |
| `scripts/run_baseline.sh` | Phase B: raw model baseline on test_theorems + miniF2F + theorem_index + baseline EBM | Yes |
| `scripts/run_iteration.sh N` | One expert iteration: fine-tune → export → EBM → search → eval + ablation | Yes |
| `scripts/run_all_iterations.sh` | Full experiment: baseline + iters 0-4 + final analysis | Yes |
| `scripts/resume_search.sh N` | Resume interrupted search from partial Parquet file | Yes |
| `scripts/smoke_test.sh` | Smoke test: 6 theorems, 4-node budget, 4 candidates, EBM train+search | Yes |
| `scripts/generate_ebm_training_data.sh` | Generate contrastive EBM training data from tactic pairs | Yes |
| `scripts/great_reset.sh` | Reset to base model: baseline EBM → LoRA → re-encode + retrain EBM → search → eval | Yes |

## Throughput Tuning

**Server-side batch generation:** With `batch_generate_size=32`, the search engine pops 32 frontier nodes and generates candidates for all of them in a **single HTTP request**. `SglangClient::generate_candidates_batch()` replicates each state's prompt `n` times into a flat batch (e.g. 32 states × 8 candidates = 256 prompts), and SGLang's RadixAttention caches shared prefixes. This replaces the previous approach of N×N sequential HTTP requests. EBM encode batching is controlled separately by `batch_encode_size` (default 8) to avoid OOM on quantized encode servers.

**Key finding:** Batched decode scales ~linearly in N on this model (not constant as hoped). 32 candidates takes ~19s vs ~2.5s for 4 candidates (~7.5× slower). After dedup at T=0.6, 32 candidates yield only 2-4 unique tactics — same as 4-8 candidates. High candidate counts waste GPU time.

**Recommended defaults** (set in `configs/search.toml` and scripts):

| Parameter | Value | Reason |
|-----------|-------|--------|
| `num_candidates` | 4 | 2-3 unique after dedup at T=0.6; ~2.5s GPU time. |
| `num_workers` | 6 | Enough to overlap Lean verification with GPU generation |
| `concurrency` | 6 | Match workers — each needs one active search |
| `max_nodes` | 100 | At 4% prove rate, proofs found within 2 nodes. 100 ≈ 3 expansions with backtrack. |

The SGLang server handles request batching and scheduling internally. Multiple concurrent search workers can issue generation/encode requests in parallel.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NUM_WORKERS` | 8 | Number of Lean worker processes |
| `CONCURRENCY` | 8 | Number of theorems searched in parallel |
| `MAX_ITER` | 4 | Maximum iteration number (0-indexed) |
| `MAX_THEOREMS` | 2000 | Maximum theorems per search run |
| `EVAL_MAX_THEOREMS` | 500 | Maximum theorems per eval run |
| `LLM_BASE` | `deepseek-ai/DeepSeek-Prover-V2-7B` | Base model for fine-tuning |
| `SKIP_BASELINE` | 0 | Set to 1 to skip Phase B baseline |
| `MODEL_PATH` | (none) | Local model dir for tokenizer in data prep |
| `MATHLIB_COMMIT` | `v4.26.0` | Mathlib4 tag to trace (matches Pantograph lean-toolchain) |
| `THEOREM_INDEX` | `data/theorem_index.json` | Path to theorem index for search |
| `EBM_STEPS` | 1500 | Number of EBM training steps |
| `EBM_RESUME` | `auto` | EBM checkpoint resume: `auto` (from prev iter), `none` (fresh) |
| `ENCODE_BATCH_SIZE` | 64 | Texts per HTTP request during embedding precomputation |
| `ENCODE_CONCURRENCY` | 2 | Concurrent encode HTTP requests |
| `SGLANG_URL` | `http://localhost:30000` | SGLang inference server URL |

## Experiment Outputs

```
baselines/                          # Phase B: raw model baseline
├── raw_test_theorems.parquet       # Pipeline validation (16 theorems)
└── raw_minif2f.json                # miniF2F zero-shot evaluation

checkpoints/ebm/baseline/           # Baseline EBM (trained on raw model trajectories)
├── final.mpk                       # burn-rs model weights
├── energy_head_config.json         # EnergyHeadConfig for loading
└── embeddings.parquet              # Precomputed embedding cache

eval_results/                       # Phase C-E: per-iteration evaluations
├── iter_0.json                     # Fine-tuned, no EBM
├── iter_1.json                     # Fine-tuned + EBM
├── iter_1_no_ebm.json              # EBM ablation
├── iter_2.json ... iter_4.json
└── iter_4_no_ebm.json              # Final ablation

trajectories/                       # Training data for next iteration
├── baseline_raw.parquet            # Raw model trajectories
├── iter_0.parquet                  # Iter 0 trajectories
├── iter_0_noisy.parquet            # Iter 0 noise injection (T=1.2)
└── iter_1.parquet ... iter_4.parquet

checkpoints/
├── llm/iter_0 ... iter_4           # LoRA adapters
├── ebm/baseline                    # Baseline EBM (raw model encoder)
└── ebm/iter_1 ... iter_4           # Fine-tuned EBM weights + config + embeddings cache

models/llm/iter_0 ... iter_4        # Merged safetensors for SGLang
logs/iter_0.log ... iter_4.log      # Per-iteration logs
```

## EBM Embedding Precomputation

EBM training requires precomputing embeddings for all unique proof states (~100K+).
This is the most time-consuming step.

**Crash-resilient encoding:** Progress is checkpointed every 20K states to
`${EBM_DIR}/embeddings.parquet`. On re-run, the script auto-loads the partial cache
and encodes only the remaining states.

**SGLang `--is-embedding` flag:** Enables the fast `/encode` endpoint (~7× faster
than legacy path). However, may disable `/generate` — restart SGLang without it
before proof search. See `docs/sglang.md` for details.

**Tuning encode throughput:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `ENCODE_BATCH_SIZE` | 64 | Texts per HTTP request. Reduce to 16 on OOM. |
| `ENCODE_CONCURRENCY` | 2 | Concurrent requests. Keep low (1-2) on 24GB GPUs. |

## Go/No-Go Checkpoints

1. **After B2** (raw baseline): If <5% on miniF2F → investigate model loading or search config
2. **After C3** (iter 0): If no improvement over baseline → check training data + loss curves
3. **After D4 vs D5** (EBM ablation): Key result — if EBM shows no improvement → investigate embeddings
4. **After each iteration**: If solve rate plateaus or decreases → stop early
