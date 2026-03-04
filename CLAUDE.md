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
- Goedel Workbook proofs: 29.7K proved (DeepSeek-Prover-V1.5 generated), migrating to Lean 4.27 (Phase M)
- LEAN-GitHub: 218K pre-traced tactic pairs from 28.6K theorems (`internlm/Lean-Github`) — human-written proof diversity, no tracing needed
- NuminaMath-LEAN: 104K IMO/USAMO/AMC/AIME formalized (Lean v4.15.0) — deferred to Phase 2

**Architecture pivot:** Joint LoRA + GoalConditioned EBM training. Single forward pass, two losses (SFT cross-entropy + InfoNCE contrastive), shared gradients through LoRA. Based on CURL (Laskin 2020) and VLM joint training findings.

**Scientific methodology:** 3-config comparison isolates variables:
- Config A: iter_0 LoRA + decoupled EBM (frozen embeddings) — baseline
- Config B: iter_1 LoRA + decoupled EBM on iter_1 embeddings — measures embedding improvement from joint training
- Config C: iter_1 LoRA + jointly-trained EBM — full system
- A→B: did joint training protect/improve embeddings?
- B→C: did live EBM gradients help the EBM head?
- A→C: total system improvement

### v2 Execution Plan (~10-12 days, ~$40-55 GPU)

| Phase | Days | What |
|-------|------|------|
| M: Migration | M.0–M.11 | Goedel → Lean 4.27 migration, data exploration, miniF2F porting, LEAN-GitHub, release |
| 0: Data Pipeline | 1–3 | PyTorch EBM port, Lean audit, parallel LeanDojo tracing, sorry filter |
| 1: SFT Baseline | 3–4 | iter_0 LoRA r=32 on competition data, deep trajectory generation (800 nodes) |
| 2: Baselines | 5–6 | Embedding metrics, decoupled GoalCond EBM training, miniF2F baseline (v1 + v2s + v2c) |
| 3: Joint Training Infra | 7–8 | JointDataset, JointProver, training loop with monitoring |
| 4: Joint Training + Eval | 9–11 | Train iter_1 LoRA r=64, 3-config evaluation, attribution analysis |

## Key Files

```
burn-qed/
├── docs/                            # Documentation
│   ├── v2_execution_plan.md         # THE PLAN — 1145 lines, 15 gotchas, all red-team fixes
│   ├── burn-qed_plan.md             # Original v1 architecture plan
│   ├── ebm_overhaul.md              # EBM architecture upgrade (v1)
│   └── experiment_guide.md          # Scripts, env vars, tuning
├── crates/                          # Rust core (search, lean-repl, policy, trajectory, prover-core)
│   └── ebm/                         # burn-rs EBM (v1, behind --features burn-ebm)
├── python/
│   ├── encode_server.py             # Embedding extraction server (nf4)
│   ├── training/                    # LLM fine-tuning scripts (SFT)
│   └── joint/                       # v2 joint training stubs
│       ├── ebm_head.py              # GoalConditionedEnergyHead
│       ├── dataset.py               # JointDataset (SFT + contrastive streams)
│       ├── losses.py                # InfoNCE (no temperature — EBM head handles it)
│       ├── model.py                 # JointProver
│       ├── monitoring.py            # separation_probe, ebm_metrics
│       └── train.py                 # Main training loop
├── scripts/                         # Pipeline orchestration (paths from _lib.sh)
├── configs/                         # TOML configs (search, models)
├── vendor/                          # Git submodules (Pantograph)
│
└── data/                            # ALL generated/downloaded artifacts (gitignored)
    ├── lean/                        # Raw HF dataset downloads
    │   ├── workbook/                # internlm/Lean-Workbook
    │   ├── goedel_proofs/           # Goedel-LM/Lean-workbook-proofs
    │   ├── lean_github/             # internlm/Lean-Github
    │   └── numinamath/              # AI-MO/NuminaMath-LEAN
    ├── benchmarks/                  # Evaluation benchmark JSONs (tracked in git)
    ├── sft/                         # Formatted SFT training data (train.jsonl, val.jsonl)
    ├── models/                      # Model weights (base/ + merged/)
    ├── checkpoints/                 # Training checkpoints (lora/ + ebm/)
    ├── trajectories/                # Search result parquets
    ├── embeddings/                  # Extracted embeddings per iteration
    ├── evals/                       # Evaluation results & reports
    ├── logs/                        # Training and search logs
    └── archive/v1/                  # Archived v1 artifacts
```

## 17 Gotchas (Hard-Won Lessons)

1. **Temperature Double-Dip:** InfoNCE has NO temperature param. EBM head has learnable temperature.
2. **25M Param Init Explosion:** First EBM layer: `weight.data *= 0.1`
3. **Monitor Temperature:** Log every 50 steps. Healthy [0.5, 3.0]. Floor/ceiling = ABORT.
4. **Tokenizer Padding:** `padding_side="right"`. Verify last-token indexing grabs content, not `<eos>`.
5. **Lean Version:** Check FIRST. Mismatch between LeanDojo, Workbook, NuminaMath = lost day. Phase M migrates Goedel to 4.27; LEAN-GitHub pre-traced strings are version-agnostic for SFT.
6. **LeanDojo Tracing Time:** 10–15h wall clock for 30K theorems, not 4–6h. Seed 20% first.
7. **Confounding Variable:** iter_0 and iter_1 use identical EBM architecture. burn-rs DEPRECATED.
8. **Search Depth:** 2K theorems × 800 nodes × 300s (not 5K × 300 × 120s). Depth > breadth.
9. **Goal Embeddings:** Task 2.1 extracts BOTH z_state AND z_goal. Day 6 EBM needs both.
10. **Zombie Lean Processes:** `cleanup_lean_processes()` every 500 theorems + `signal.alarm(120)`.
11. **Validation Split:** By THEOREM NAME, not tactic pairs. Prevents variable-name leakage.
12. **Sorry Filter:** Reject entire theorem if any tactic contains `sorry`/`admit`/`cheat`.
13. **Loss Masking:** `DataCollatorForCompletionOnlyLM` with `response_template="` ``` `\n"` (closing code fence). Only train on tactic tokens after the closing fence. Without this, 90% of LoRA capacity wasted echoing proof states. See `docs/data_format_spec.md` for full format details.
14. **Prompt Format:** DeepSeek-native format with tactic state as Lean comment inside code fence. No special tokens (`[GOAL]`/`[PROOFSTEP]` are NOT used). See `docs/data_format_spec.md`.
15. **CPU Worker OOM:** Cap `NUM_WORKERS = min(16, int(cpu_count() * 0.75))`. 30 Lean REPLs will OOM 200GB RAM.
16. **Phase M Survival Rate:** If Goedel 4.27 migration < 90% survival, SFT dataset may be too small. Below 80%, consider older toolchain.
17. **Token Geometry Truncation:** Don't assume `--max-length 2048`. Phase M Task M.7 computes actual distribution. If p95 ≤ 1024, using 2048 wastes ~50% VRAM on padding.

## Success Metrics

| Metric | iter_0 target | iter_1 target | Red line |
|--------|---------------|---------------|----------|
| miniF2F (LLM-only) | ≥ 35% | ≥ 35% | < 25% |
| miniF2F (LLM+EBM) | ≥ 40% | ≥ 45% | < 35% |
| centroid_l2 | ≥ 5.0 | ≥ iter_0 | < 3.0 |
| linear_probe_acc | ≥ 0.75 | ≥ iter_0 | < 0.65 |
| EBM rank-1 (depth 4+) | ≥ 0.30 | ≥ 0.40 | < 0.25 |

## v2 Task Tracker

Reference: `docs/v2_execution_plan.md` for full details, code snippets, and gotchas.

### Phase M: Goedel Migration + Data Exploration (Days M.0–M.11)

- [x] **M.1** Bulk compile attempt — 28,016/29,759 proofs compile on Lean 4.27 (94.1%)
- [x] **M.2** Automated fixes (renames, instance patches, `set_option autoImplicit true`) — applied
- [x] **M.3** Manual triage — 28,016 passing, remaining dropped. Merged into 16 chunks for tracing
- [ ] **M.4** Port miniF2F-v2s/v2c statements to 4.27, verify all 488 compile — eval benchmark ready
- [x] **M.5** Pantograph extraction (replaced LeanDojo ExtractData) — **60,341 pairs from 24,879 theorems** → `data/traced/pantograph_pairs/goedel_427_pairs.jsonl`. Avg 2.4 tactics/thm, 46% single-tactic (machine-generated proofs are shallow). Script: `python/data/goedel_migration/extract_pairs_pantograph.py`
- [x] **M.6** Integrity sweep — 28,016/28,016 passing proofs clean (0 sorry/admit/cheat/sorryAx in tactic blocks, 64 trivial single-tactic proofs). Report: `data/traced/integrity_report.json`
- [x] **M.7** Token geometry — p95=506 tokens, p99=793, **recommended max_length=1024** (0.41% truncated, saves ~50% VRAM vs 2048). Goedel p95=296, LEAN-GitHub p95=545. Tactic p95=78 tokens. Mean full sequence: 202 tokens.
- [x] **M.8** Proof structure & tactic vocabulary — Combined: 849 unique tactic heads, top-5 cover 50% (good diversity). Contrastive pool: 19,213 theorems at depth>=3 (222K pairs), 12,315 at depth>=5. Goedel dominated by `nlinarith` (53% of single-tactic proofs); LEAN-GitHub much more diverse (entropy 4.87 vs 3.99 bits). Combined effective vocab: 32 tactics. Search pool for Task 1.3: ample (2K from 19K available).
- [x] **M.9** Download + filter LEAN-GitHub — **196,853 pairs from 19,449 theorems** → `data/traced/lean_github_pairs.jsonl`. Avg 10.1 tactics/thm, median 4, 64.7% depth≥3. Filters: sorry (1,364 thms removed), state>4096 chars (630 rows), trivial subsampling 10% (10,909 rows). Lean 4.27 compat: 99.8% safe (<0.24% confirmed-renamed APIs). Script: `python/data/process_lean_github.py`
- [x] **M.10** Merge into unified SFT dataset — Goedel (60K) + LEAN-GitHub (197K) = **257K pairs, 44K theorems**. Train: 245,025 pairs (41,563 thms), Val: 12,169 pairs (2,094 thms). Split by theorem name (Gotcha 11). Contrastive pool (depth≥3): 19,213 theorems. Script: `python/data/merge_sft_dataset.py`. Output: `data/sft/{train,val}.jsonl`, `data/sft/contrastive_pool.json`
- [ ] **M.11** Release Goedel-4.27 on HuggingFace, write migration notes — community contribution

### Phase 0: Environment Setup + Data Pipeline (Days 1–3)

- [x] **0.pre** Archive v1 infrastructure, feature-gate burn-rs, create Python stubs
- [x] **0.1** Lean version audit (2h) — our Pantograph is Lean v4.27.0 / Mathlib v4.27.0. Datasets are v4.8–4.9 (18 minor versions behind). Main risk: Mathlib lemma renames. Check LeanDojo vs Workbook vs NuminaMath versions, quantify rename impact
- [x] **0.3** Download datasets — Goedel (migrated + traced in Phase M), LEAN-GitHub (downloaded + filtered in M.9). NuminaMath deferred to Phase 2
- [x] **0.3e** ~~Pantograph validation — Goedel~~ — superseded by M.5 (full Pantograph extraction, 24,879 theorems pass)
- [x] **0.3h** ~~Download + filter LEAN-GitHub~~ — done as M.9: 196,853 pairs, 19,449 theorems
- [x] **0.4** ~~Trace Goedel Workbook proofs~~ — done as M.5: 60,341 pairs via Pantograph
- [ ] **0.5** Trace NuminaMath-LEAN proved subset — **deferred to Phase 2** (Goedel + LEAN-GitHub = 257K pairs, sufficient)
- [x] **0.6** ~~Merge and format unified SFT dataset~~ — done as M.10: 245K train / 12K val, DeepSeek-native format, contrastive pool built

### Phase 1: Iter 0 SFT + EBM Baseline (Days 3–4)

- [ ] **1.0** Port GoalConditionedEnergyHead to PyTorch (3h) — implement `python/joint/ebm_head.py` with unit tests: output shape, spectral norms ~1.0, first-layer weight ~0.1x, temperature divides forward(), EmbeddingExtractor grabs correct token
- [ ] **1.1** Train iter_0 LoRA r=32 (5–7h GPU) — seed data first, retrain on full when ready. Completion-only loss masking (Gotcha 13). Check special tokens (Gotcha 14)
- [ ] **1.2** Merge LoRA and deploy to SGLang (0.5h)
- [ ] **1.3** Search on contrastive pool — 2K theorems × 800 nodes × 300s (6–8h GPU) — depth over breadth (Gotcha 8), T=1.3, 16 candidates
- [ ] **1.4** Quick miniF2F sanity check (1.5h GPU) — LLM-only baseline, compare to old iter_4

### Phase 2: Embedding + EBM Baseline (Days 5–6)

- [ ] **2.1** Extract embeddings from iter_0 — STATES AND GOALS (2h GPU) — save to `data/embeddings/iter_0/` (Gotcha 9: extract both z_state and z_goal)
- [ ] **2.2** Compute embedding baseline metrics (2h, no GPU) — centroid_l2, linear_probe, norm_gap, sibling_l2, variance spectrum, depth-stratified clustering, dual-label rate
- [ ] **2.3** Generate embedding visualizations (1h) — t-SNE/UMAP, depth coloring, sibling histograms, eigenvalue spectrum
- [ ] **2.4** Train decoupled goal-conditioned EBM on frozen iter_0 embeddings (3h GPU) — SAME architecture as iter_1. Save to `data/checkpoints/ebm/iter_0/`
- [ ] **2.5** Compute EBM baseline metrics (1h) — rank-1 accuracy (overall + by depth), energy gap, sibling discrimination, active ratio
- [ ] **2.6** EBM-augmented miniF2F search (2–3h GPU) — needs Python encode server wrapping GoalConditionedEnergyHead
- [ ] **2.7** Compile baseline report (1h) — `data/evals/iter_0/baseline_report.md`

### Phase 3: Joint Training Loop (Days 7–8)

- [ ] **3.1** JointDataset for competition data (3h) — implement `python/joint/dataset.py`: SFT + contrastive interleaved streams
- [ ] **3.2** JointProver model (2.5h) — implement `python/joint/model.py`: backbone + LoRA r=64 + GoalConditionedEnergyHead
- [ ] **3.3** InfoNCE loss — no temperature (0.5h) — implement `python/joint/losses.py` (Gotcha 1)
- [ ] **3.4** Training loop with monitoring (2h) — implement `python/joint/train.py`: joint SFT+InfoNCE backward, separation probe every 500 steps, temperature logging every 50 steps (Gotcha 3)
- [ ] **3.5** Smoke test — 1 batch forward+backward (0.5h GPU) — verify finite losses, gradients flow to both LoRA and EBM head, peak VRAM <36GB

### Phase 4: Joint Training + Evaluation (Days 9–11)

- [ ] **4.1** Train iter_1 joint (6–8h GPU) — LoRA r=64, lr_lora=2e-5, lr_ebm=3e-5, λ_ebm=0.1, 6K steps. Monitor abort conditions: centroid_l2 <3.0, SFT loss >0.5 after 2K steps, temperature floor/ceiling
- [ ] **4.2** Export and deploy (0.5h) — merge LoRA to `data/models/merged/iter_1/`, save EBM head to `data/checkpoints/ebm/iter_1/`
- [ ] **4.3** Repeat ALL baseline measurements on iter_1 (4h GPU) — 3-config comparison: A (iter_0 decoupled), B (iter_1 LoRA + decoupled EBM), C (iter_1 full joint)
- [ ] **4.4** Comparative analysis (2h) — `data/evals/iter_1/analysis/comparison_report.md`: did competition data help? did joint training protect embeddings? did goal conditioning help? end-to-end miniF2F delta?
- [ ] **4.5** Decision point — iter_1 > iter_0 + centroid held → plan iter_2; iter_0 >> old but iter_1 ≈ iter_0 → data was bottleneck; centroid collapsed → debug joint training

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