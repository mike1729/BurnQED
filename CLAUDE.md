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
- Goedel Workbook proofs: 29.7K proved (DeepSeek-Prover-V1.5 generated), migrating to Lean 4.26 (Phase M)
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

### v2 Execution Plan (~15 days, ~$49 GPU)

| Phase | Days | What |
|-------|------|------|
| M: Migration | M.0–M.6 | Goedel → Lean 4.26 migration, miniF2F-v2s/v2c porting, LEAN-GitHub integration |
| 0: Data Pipeline | 1–3 | PyTorch EBM port, Lean audit, parallel LeanDojo tracing, sorry filter |
| 1: SFT Baseline | 3–4 | iter_0 LoRA r=32 on competition data, deep trajectory generation (800 nodes) |
| 2: Baselines | 5–6 | Embedding metrics, decoupled GoalCond EBM training, miniF2F baseline (v1 + v2s + v2c) |
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
├── data/                        # HF datasets, miniF2F JSONs, traced tactic pairs
├── iterations/
│   ├── iter_0/                  # SFT-only baseline + decoupled EBM + trajectories
│   └── iter_1/                  # Joint-trained model + jointly-trained EBM
├── archive/v1/                  # Archived v1 artifacts and burn-rs scripts
└── scripts/                     # Server startup, eval orchestration
```

## 15 Gotchas (Hard-Won Lessons)

1. **Temperature Double-Dip:** InfoNCE has NO temperature param. EBM head has learnable temperature.
2. **25M Param Init Explosion:** First EBM layer: `weight.data *= 0.1`
3. **Monitor Temperature:** Log every 50 steps. Healthy [0.5, 3.0]. Floor/ceiling = ABORT.
4. **Tokenizer Padding:** `padding_side="right"`. Verify last-token indexing grabs content, not `<eos>`.
5. **Lean Version:** Check FIRST. Mismatch between LeanDojo, Workbook, NuminaMath = lost day. Phase M migrates Goedel to 4.26; LEAN-GitHub pre-traced strings are version-agnostic for SFT.
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

### Phase M: Goedel Migration + Data Integration (Days M.0–M.6)

- [ ] **M.0** Clone Goedel, update toolchain to 4.26, attempt `lake build` — log errors, survival count
- [ ] **M.1** Automated fixes (renames, instance patches), rebuild — improved survival count
- [ ] **M.2** Manual triage of remaining failures, drop or fix — target ≥95% compilation (≥28,270 of 29,759)
- [ ] **M.3** Port miniF2F-v2s/v2c statements to 4.26, verify all 488 compile — eval benchmark ready
- [ ] **M.4** LeanDojo trace on compiled Goedel 4.26 proofs — tactic pairs parquet
- [ ] **M.5** Download + filter LEAN-GitHub (`internlm/Lean-Github`, 218K tactics), merge with Goedel pairs — unified SFT dataset
- [ ] **M.6** Release Goedel-4.26 on HuggingFace, write migration notes — community contribution

### Phase 0: Environment Setup + Data Pipeline (Days 1–3)

- [x] **0.pre** Archive v1 infrastructure, feature-gate burn-rs, create Python stubs
- [x] **0.1** Lean version audit (2h) — our Pantograph is Lean v4.26.0 / Mathlib v4.26.0. Datasets are v4.8–4.9 (18 minor versions behind). Main risk: Mathlib lemma renames. Check LeanDojo vs Workbook vs NuminaMath versions, quantify rename impact
- [ ] **0.3** Download datasets (1h) — Lean Workbook (single JSON, 25.2K pre-traced tactic pairs + 57K+83K problems), Goedel Workbook proofs (29.8K), LEAN-GitHub (218K tactics from 28.6K theorems), NuminaMath-LEAN (104K) from HuggingFace
- [ ] **0.3a** Survey all Lean 4 datasets (1.5h) — catalog every HF dataset: LeanDojo-v2, LEAN-GitHub, Herald, DeepSeek-Prover-V1, Kimina-Prover-Promptset, FormalMATH, AI4M/less-proofnet-lean4, LeanTree, plus our four targets. For each: rows, format, Lean/Mathlib version, proof type, license, relevance. Check for newer versions. Write findings to `docs/datasets.md`
- [ ] **0.3b** Data inventory & quality report (1h) — write `python/data/inspect_datasets.py`: load four target HF datasets, report total rows/schema, sorry/admit/cheat contamination rates, Lean Workbook depth distribution + tactic length stats + dedup check (InternLM-Math-Plus/StepProver overlap), Goedel overlap with Lean Workbook `id` → net-new count + audit import lines for non-Mathlib imports, LEAN-GitHub quality distribution + state length stats, NuminaMath non-empty `formal_proof` count + `ground_truth_type` distribution. Update `docs/datasets.md`
- [ ] **0.3c** Pre-traced data format validation (0.5h) — for 25.2K Lean Workbook tactic pairs: verify `state_before`/`tactic` format vs our DeepSeek-native prompt format (see `docs/data_format_spec.md`), write converter if needed, compute depth distribution (need depth ≥ 3 for contrastive pool), run sorry/admit filter, determine if usable for immediate SFT
- [ ] **0.3d** Pantograph validation — Lean Workbook (1.5h) — use Rust `lean-repl` crate (`LeanPool`/`ProofSession`): extend `prover-core` with `validate-tactics` subcommand, sample 50 theorems stratified by depth, replay tactic sequences via `goal.start` + `goal.tactic`, compare replayed goals against pre-traced `state_after` to detect formatting drift (Mathlib v4.8→v4.26). Categorize failures: lemma renames, tactic API changes, missing imports, pretty-printer divergence. If tactic success < 80%: re-tracing needed. If ≥ 80% but text differs: re-extract states during replay
- [ ] **0.3e** Pantograph validation — Goedel proofs (1.5h) — same `lean-repl` approach: sample 30 non-overlapping Goedel proofs, parse `full_proof` to extract statement + tactics (handle imports, `set_option`), check for non-Mathlib imports, replay through Pantograph stratified by depth, measure compilation rate, categorize failures. If Lean 4.9→4.26 breaks too much: try import fixups or skip source
- [ ] **0.3f** Pantograph validation — NuminaMath (1.5h) — sample 30 proofs where `formal_proof` non-empty and `ground_truth_type` is "complete", check tactic-style vs term-style (term proofs can't be replayed step-by-step), replay tactic-style through Pantograph, check Lean version compatibility, estimate tracing yield
- [ ] **0.3g** Data strategy decision (0.5h, decision point) — based on 0.3a–0.3f: determine immediate SFT data (pre-traced pairs passing Pantograph → use directly; tactics pass but states differ → replay all for consistent states; tactics fail >20% → need full re-tracing). Check if survey found better sources. Set tracing priority. If pre-traced pairs fully pass, Task 0.4 seed batch may be unnecessary. Update task descriptions for 0.4/0.5/0.6
- [ ] **0.3h** Download + filter LEAN-GitHub (1h) — download `internlm/Lean-Github`, apply quality filtering (state length < 4096, trivial tactic subsampling), source-prefixed dedup against Goedel, convert to SFT format. Expected yield: ~100-150K pairs
- [ ] **0.4** Trace Goedel Workbook proofs — parallel + chunked (Day 2, 8–12h wall) — **prerequisite: Phase M migration or 0.3e Pantograph validation** determines if Goedel proofs compile under our Lean v4.26. Seed 20% first (~6K, ~2–3h), start SFT on seed, remainder overnight. ABORT if error rate >20%
- [ ] **0.5** Trace NuminaMath-LEAN proved subset — **deferred to Phase 2** (after Goedel + LEAN-GitHub validated; Goedel + LEAN-GitHub already provide ~210-350K pairs)
- [ ] **0.6** Filter and format tactic pairs (1h) — combine sources including pre-traced Lean Workbook pairs (format-converted per 0.3c), LEAN-GitHub filtered pairs (from 0.3h), and Goedel traced pairs (from 0.4). Source-prefixed dedup, split by theorem name (Gotcha 11), sorry filter (Gotcha 12), depth≥3 contrastive pool

### Phase 1: Iter 0 SFT + EBM Baseline (Days 3–4)

- [ ] **1.0** Port GoalConditionedEnergyHead to PyTorch (3h) — implement `python/joint/ebm_head.py` with unit tests: output shape, spectral norms ~1.0, first-layer weight ~0.1x, temperature divides forward(), EmbeddingExtractor grabs correct token
- [ ] **1.1** Train iter_0 LoRA r=32 (5–7h GPU) — seed data first, retrain on full when ready. Completion-only loss masking (Gotcha 13). Check special tokens (Gotcha 14)
- [ ] **1.2** Merge LoRA and deploy to SGLang (0.5h)
- [ ] **1.3** Search on contrastive pool — 2K theorems × 800 nodes × 300s (6–8h GPU) — depth over breadth (Gotcha 8), T=1.3, 16 candidates
- [ ] **1.4** Quick miniF2F sanity check (1.5h GPU) — LLM-only baseline, compare to old iter_4

### Phase 2: Embedding + EBM Baseline (Days 5–6)

- [ ] **2.1** Extract embeddings from iter_0 — STATES AND GOALS (2h GPU) — save to `iterations/iter_0/embeddings/` (Gotcha 9: extract both z_state and z_goal)
- [ ] **2.2** Compute embedding baseline metrics (2h, no GPU) — centroid_l2, linear_probe, norm_gap, sibling_l2, variance spectrum, depth-stratified clustering, dual-label rate
- [ ] **2.3** Generate embedding visualizations (1h) — t-SNE/UMAP, depth coloring, sibling histograms, eigenvalue spectrum
- [ ] **2.4** Train decoupled goal-conditioned EBM on frozen iter_0 embeddings (3h GPU) — SAME architecture as iter_1. Save to `iterations/iter_0/ebm/`
- [ ] **2.5** Compute EBM baseline metrics (1h) — rank-1 accuracy (overall + by depth), energy gap, sibling discrimination, active ratio
- [ ] **2.6** EBM-augmented miniF2F search (2–3h GPU) — needs Python encode server wrapping GoalConditionedEnergyHead
- [ ] **2.7** Compile baseline report (1h) — `iterations/iter_0/baselines/baseline_report.md`

### Phase 3: Joint Training Loop (Days 7–8)

- [ ] **3.1** JointDataset for competition data (3h) — implement `python/joint/dataset.py`: SFT + contrastive interleaved streams
- [ ] **3.2** JointProver model (2.5h) — implement `python/joint/model.py`: backbone + LoRA r=64 + GoalConditionedEnergyHead
- [ ] **3.3** InfoNCE loss — no temperature (0.5h) — implement `python/joint/losses.py` (Gotcha 1)
- [ ] **3.4** Training loop with monitoring (2h) — implement `python/joint/train.py`: joint SFT+InfoNCE backward, separation probe every 500 steps, temperature logging every 50 steps (Gotcha 3)
- [ ] **3.5** Smoke test — 1 batch forward+backward (0.5h GPU) — verify finite losses, gradients flow to both LoRA and EBM head, peak VRAM <36GB

### Phase 4: Joint Training + Evaluation (Days 9–11)

- [ ] **4.1** Train iter_1 joint (6–8h GPU) — LoRA r=64, lr_lora=2e-5, lr_ebm=3e-5, λ_ebm=0.1, 6K steps. Monitor abort conditions: centroid_l2 <3.0, SFT loss >0.5 after 2K steps, temperature floor/ceiling
- [ ] **4.2** Export and deploy (0.5h) — merge LoRA, save EBM head to `iterations/iter_1/`
- [ ] **4.3** Repeat ALL baseline measurements on iter_1 (4h GPU) — 3-config comparison: A (iter_0 decoupled), B (iter_1 LoRA + decoupled EBM), C (iter_1 full joint)
- [ ] **4.4** Comparative analysis (2h) — `iterations/iter_1/analysis/comparison_report.md`: did competition data help? did joint training protect embeddings? did goal conditioning help? end-to-end miniF2F delta?
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