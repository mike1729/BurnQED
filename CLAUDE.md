# burn-qed — Project Summary for Claude

## What This Is

Lean 4 theorem prover combining an LLM policy (tactic generation) with an Energy-Based Model value function (proof state scoring) to guide best-first proof search. Trained via expert iteration. Primary evaluation benchmark: miniF2Fv2 (244 competition-level math problems formalized in Lean 4).

## Architecture

```
DeepSeek-Prover-V2-7B (SGLang, port 30000)
├── Policy: autoregressive tactic generation (LM head)
└── Embeddings: mean-pool hidden states → encode server (port 30001, nf4, ~6.9GB VRAM)
                    │
                    ▼
        GoalConditionedEnergyHead (PyTorch, ~25M params)
        [z_state; z_goal; z_state⊙z_goal] → 12288 → 2048 → 1024 → 512 → 1
        Spectral norm all layers, SiLU, dropout 0.15, learnable temperature
        Output: scalar energy (lower = more provable)

Lean 4 REPL Pool (Rust/tokio, Pantograph JSON protocol)
└── Verifies tactics, returns new goals
```

Single shared 7B backbone for both policy and value. Rust core (`prover-core`) handles search, trajectory I/O, and orchestration. Python handles LLM fine-tuning (HuggingFace PEFT/LoRA) and EBM training. The EBM originally lived in burn-rs but has been **deprecated in favor of PyTorch** as of v2.

## Current Status: v2 Pivot (Starting ~Feb 25, 2026)

### What Happened in v1 (Iterations 0–5)

**Training data:** Mathlib4 (122K abstract theorems) — category theory, topology, measure theory.

**Best results (iter_4):**
- miniF2F LLM+EBM: **41.7%** (partial, 139/244 evaluated)
- Mathlib clean eval: **59.0%** (LLM+EBM) vs 47.0% (LLM-only) — EBM added +12pp, zero regressions
- Embedding quality: centroid_l2 = 6.40, linear_probe = 0.83

**iter_5 FAILURE — Embedding Collapse:**
- Fresh LoRA on iter_4 base destroyed embedding separation
- centroid_l2: 6.40 → 1.33 (80% collapse)
- linear_probe: 0.83 → 0.68
- Proof rate regressed: 34% → 30%
- Root cause: SFT-only training optimizes next-token prediction, which disrupts the representation structure the EBM depends on

### Why v2: Two Problems to Solve

1. **Distribution mismatch:** Mathlib4 (abstract math) ≠ miniF2F (competition algebra/number theory). Limited LoRA capacity wasted on out-of-distribution patterns.
2. **Embedding collapse:** SFT-only training destroys embeddings. Need joint LLM+EBM training where contrastive loss actively protects representation quality.

### v2 Strategy

**Data pivot:** Drop Mathlib4. Retrain from scratch on competition-focused datasets:
- Lean Workbook: 57K + 83K competition problems
- Goedel Workbook proofs: 29.7K proved (DeepSeek-Prover-V1.5 generated)
- NuminaMath-LEAN: 100K IMO/USAMO/AMC/AIME formalized

**Architecture pivot:** Joint LoRA + GoalConditioned EBM training. Single forward pass, two losses (SFT cross-entropy + InfoNCE contrastive), shared gradients through LoRA. Based on CURL (Laskin 2020) and VLM joint training findings.

**Scientific methodology:** 3-config comparison isolates variables:
- Config A: iter_0 LoRA + decoupled EBM (frozen embeddings) — baseline
- Config B: iter_1 LoRA + decoupled EBM on iter_1 embeddings — measures embedding improvement from joint training
- Config C: iter_1 LoRA + jointly-trained EBM — full system
- A→B: did joint training protect/improve embeddings?
- B→C: did live EBM gradients help the EBM head?
- A→C: total system improvement

### v2 Execution Plan (11 days, ~$49 GPU)

| Phase | Days | What |
|-------|------|------|
| 0: Data Pipeline | 1–3 | PyTorch EBM port, Lean audit, parallel LeanDojo tracing, sorry filter |
| 1: SFT Baseline | 3–4 | iter_0 LoRA r=32 on competition data, deep trajectory generation (800 nodes) |
| 2: Baselines | 5–6 | Embedding metrics, decoupled GoalCond EBM training, miniF2F baseline |
| 3: Joint Training Infra | 7–8 | JointDataset, JointProver, training loop with monitoring |
| 4: Joint Training + Eval | 9–11 | Train iter_1 LoRA r=64, 3-config evaluation, attribution analysis |

## Key Files

```
burn-qed/
├── docs/
│   ├── v2_execution_plan.md     # THE PLAN — 1145 lines, 15 gotchas, all red-team fixes
│   ├── burn-qed_plan.md         # Original v1 architecture plan
│   ├── ebm_overhaul.md          # EBM architecture upgrade (v1)
│   └── experiment_guide.md      # Scripts, env vars, tuning
├── crates/                      # Rust core (search, lean-repl, policy, trajectory, prover-core)
│   └── ebm/                     # burn-rs EBM (v1, behind --features burn-ebm)
├── python/
│   ├── encode_server.py         # Embedding extraction server (nf4)
│   ├── training/                # LLM fine-tuning scripts (SFT)
│   └── joint/                   # v2 joint training stubs
│       ├── ebm_head.py          # GoalConditionedEnergyHead
│       ├── dataset.py           # JointDataset (SFT + contrastive streams)
│       ├── losses.py            # InfoNCE (no temperature — EBM head handles it)
│       ├── model.py             # JointProver
│       ├── monitoring.py        # separation_probe, ebm_metrics
│       └── train.py             # Main training loop
├── data/traced/                 # sft_train_seed.jsonl, sft_train_full.jsonl
├── iterations/
│   ├── iter_0/                  # SFT-only baseline + decoupled EBM + trajectories
│   └── iter_1/                  # Joint-trained model + jointly-trained EBM
├── archive/v1/                  # Archived v1 artifacts and burn-rs scripts
└── scripts/                     # Server startup, eval orchestration
```

> **Note:** `docs/v2_execution_plan.md` still references `v2/` paths — it's the reference plan. Paths in this file are the source of truth for actual layout.

## 15 Gotchas (Hard-Won Lessons)

1. **Temperature Double-Dip:** InfoNCE has NO temperature param. EBM head has learnable temperature.
2. **25M Param Init Explosion:** First EBM layer: `weight.data *= 0.1`
3. **Monitor Temperature:** Log every 50 steps. Healthy [0.5, 3.0]. Floor/ceiling = ABORT.
4. **Tokenizer Padding:** `padding_side="right"`. Verify last-token indexing grabs content, not `<eos>`.
5. **Lean Version:** Check FIRST. Mismatch between LeanDojo, Workbook, NuminaMath = lost day.
6. **LeanDojo Tracing Time:** 10–15h wall clock for 30K theorems, not 4–6h. Seed 20% first.
7. **Confounding Variable:** iter_0 and iter_1 use identical EBM architecture. burn-rs DEPRECATED.
8. **Search Depth:** 2K theorems × 800 nodes × 300s (not 5K × 300 × 120s). Depth > breadth.
9. **Goal Embeddings:** Task 2.1 extracts BOTH z_state AND z_goal. Day 6 EBM needs both.
10. **Zombie Lean Processes:** `cleanup_lean_processes()` every 500 theorems + `signal.alarm(120)`.
11. **Validation Split:** By THEOREM NAME, not tactic pairs. Prevents variable-name leakage.
12. **Sorry Filter:** Reject entire theorem if any tactic contains `sorry`/`admit`/`cheat`.
13. **Loss Masking:** `DataCollatorForCompletionOnlyLM` — only train on tactic tokens after `[PROOFSTEP]`. Without this, 90% of LoRA capacity wasted echoing proof states.
14. **Special Token Fragmentation:** Check if `[GOAL]`/`[PROOFSTEP]` are in vocab. If not, `add_special_tokens()` + `resize_token_embeddings()` + `modules_to_save=["embed_tokens", "lm_head"]`.
15. **CPU Worker OOM:** Cap `NUM_WORKERS = min(16, int(cpu_count() * 0.75))`. 30 Lean REPLs will OOM 200GB RAM.

## Success Metrics

| Metric | iter_0 target | iter_1 target | Red line |
|--------|---------------|---------------|----------|
| miniF2F (LLM-only) | ≥ 35% | ≥ 35% | < 25% |
| miniF2F (LLM+EBM) | ≥ 40% | ≥ 45% | < 35% |
| centroid_l2 | ≥ 5.0 | ≥ iter_0 | < 3.0 |
| linear_probe_acc | ≥ 0.75 | ≥ iter_0 | < 0.65 |
| EBM rank-1 (depth 4+) | ≥ 0.30 | ≥ 0.40 | < 0.25 |

## Settled Architecture Decisions (Do NOT Revisit)

1. Shared 7B backbone (not separate encoder)
2. SGLang inference server (not in-process)
3. SpectralNorm Option C (random reinit, 5 power iterations)
4. Pantograph protocol for Lean REPL (JSON lines, `\n` terminated, 30s timeout)
5. Worker recycling: 1000 requests OR 30 min TTL
6. Batch EBM scoring with deferred collect-then-score pattern
7. **burn-rs EBM pipeline DEPRECATED** — replaced by PyTorch GoalConditionedEnergyHead in v2. Gated behind `--features burn-ebm`; default builds have zero burn dependencies

## Related Literature

- **PACT** (Han 2021): Joint training with auxiliary tasks on proof terms → 32%→48% Lean
- **HTPS** (Lample 2022): AlphaZero policy+value on shared transformer → 82.6% Metamath, 42% miniF2F
- **AlphaProof** (Hubert 2025): Policy/value + AlphaZero search + millions of auto-formalized problems
- **CURL** (Laskin & Srinivas 2020): Contrastive auxiliary loss prevents representation collapse under primary task — directly parallels our iter_5 problem
- **VLM joint training** (Amazon 2024): Contrastive + SFT on shared backbone fixes misaligned embeddings