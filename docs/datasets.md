# Lean 4 Training Datasets — Survey & Pipeline Plan

Last updated: 2026-02-26

## Overview

This document catalogs all known Lean 4 theorem-proving datasets on HuggingFace and beyond, evaluates them for our competition-focused v2 training pipeline, and documents format/compatibility details for our three target datasets.

**Our setup:** Pantograph on Lean v4.26.0 / Mathlib v4.26.0. DeepSeek-Prover-V2-7B backbone. SFT format: DeepSeek-native prompt with tactic state as Lean comment (see `docs/data_format_spec.md`).

**v2.1 additions:** LEAN-GitHub (218K pre-traced tactic pairs) promoted to high-relevance target. Goedel Workbook proofs migrated to Lean 4.26 (new Phase M). miniF2F-v2s as primary evaluation benchmark. Lean Workbook (InternLM) tactic pairs deprecated — fully subsumed by Goedel 4.26 migration. NuminaMath deferred to Phase 2.

---

## 1. Complete Lean 4 Dataset Landscape

### 1.1 Our Three Target Datasets

| Dataset | HF Path | Rows | Lean Version | Format | License |
|---------|---------|------|-------------|--------|---------|
| Goedel Workbook Proofs | `Goedel-LM/Lean-workbook-proofs` | 29.8K full proofs | Lean 4.9 → **4.26 (Phase M)** | Full proofs (`problem_id`, `full_proof`) | Apache 2.0 |
| LEAN-GitHub | `internlm/Lean-Github` | 28.6K theorems, 218K tactics | Mixed (pre-4.26) | Pre-traced tactic pairs (`state_before`, `tactic`, `state_after`) | Apache 2.0 |
| NuminaMath-LEAN | `AI-MO/NuminaMath-LEAN` | 104K rows | v4.15.0 (Mathlib v4.15.0) | Formal statements + proofs (`formal_statement`, `formal_proof`) | Apache 2.0 |

### 1.2 Other Lean 4 Datasets

| Dataset | HF Path | Size | Lean Version | Format | Notes |
|---------|---------|------|-------------|--------|-------|
| Lean Workbook tactic pairs | `internlm/Lean-Workbook` | 25.2K tactic pairs | v4.8.0-rc1 | Pre-traced (`state_before`, `tactic`, `state_after`) | **Deprecated.** Subset of Goedel (15.7K theorems). Superseded by Phase M Goedel 4.26 migration which covers all 29.7K at the correct Lean version. |
| LeanDojo Benchmark 4 | `kaist-ai/LeanDojo-Benchmark-4` | 122K theorems, 259K tactics | v4.3.0+ (Mathlib4) | Tactic pairs from Mathlib4 tracing | Used in v1. Abstract math, not competition-focused. |
| Herald | `xueliangz/Herald` | 580K statements from 110K Mathlib4 theorems | Mathlib4 | Natural language + formal statement pairs | Useful for autoformalization, not direct SFT. |
| DeepSeek-Prover-V1 Data | `deepseek-ai/DeepSeek-Prover-V1` | ~8K proved | v4.3.0 | Full proofs | Older, smaller. Superceded by V1.5/V2 data. |
| Kimina-Prover-Promptset | `AI-MO/Kimina-Prover-Preview-Distill-Promptset` | ~7K | Unknown | Prompt-completion pairs | Distilled from larger model, may have quality issues. |
| FormalMATH-Bench | `Sphere-AI-Lab/FormalMATH-Bench` | 5.5K benchmark | ~v4.15 | Formalized competition problems | Evaluation benchmark (HS olympiad to undergrad), not training data. |
| AI4M less-proofnet-lean4 | `AI4M/less-proofnet-lean4-top1M` | Up to 1M | Unknown | Lean 4 proof terms | Massive but unverified quality. Needs investigation. |
| LeanTree | Various | Proof trees | Unknown | Tree-structured proofs | Specialized format, may need conversion. |
| LeanNavigator | — | 4.7M generated theorems | Unknown | Generated theorems | Feb 2025. Huge but synthetic, needs quality check. |
| IMO-Steps | `roozbeh-yz/IMO-Steps` | 1.3K lemmas (20 IMO problems) | v4.17.0 | Structured step-by-step | Small but high-quality IMO-specific. |
| Kimina-Lean-Server data | (via `project-numina/kimina-lean-server`) | 9.4K sorry-free | v4.15.0 | Filtered NuminaMath proofs | Verified subset of NuminaMath. |
| Lean Workbook (full) | `internlm/Lean-Workbook` | 57K + 83K problems | v4.8.0-rc1 | Problem statements (many unproved) | Raw problem set. Goedel proved 29.7K of these; use Goedel for proofs. |
| InternLM-Math-Plus | Part of Lean-Workbook | Overlaps with Lean Workbook | v4.8.0-rc1 | Extended annotations | StepProver data also overlaps. Check dedup. |

### 1.3 Relevance Assessment

**High relevance (competition math, tactic-style, Lean 4):**
- Goedel Workbook proofs — large, competition-focused, Phase M migrates to 4.26 + LeanDojo trace
- LEAN-GitHub — 218K pre-traced tactic pairs from 28.6K theorems, human-written proof diversity that complements machine-generated Goedel proofs. Already in (state, tactic) format — no tracing needed.
- NuminaMath-LEAN — IMO/USAMO/AMC/AIME, largest collection, needs quality audit (deferred to Phase 2)

**Medium relevance:**
- LeanDojo Benchmark 4 — proven pipeline but wrong domain (Mathlib abstract math)
- AI4M less-proofnet-lean4 — huge but unknown quality, needs investigation

**Deprecated:**
- Lean Workbook tactic pairs (InternLM) — 25.2K pairs from 15.7K theorems at v4.8. Fully subsumed by Goedel 4.26 migration (29.7K theorems, same problem set, correct Lean version).

**Low relevance for v2:**
- Herald — NL↔formal pairs, not tactic proofs
- DeepSeek-Prover-V1 — superseded
- Kimina-Prover-Promptset — small, unknown provenance

---

## 2. Target Dataset Details

### 2.1 Goedel Workbook Proofs (Goedel-LM/Lean-workbook-proofs)

**Source:** Goedel-LM team, proofs generated by DeepSeek-Prover-V1.5.

**Schema (29.8K rows):**
```
{
  "problem_id": str,       # e.g. "lean_workbook_0" — links to Lean Workbook
  "full_proof": str,       # Complete Lean 4 proof including imports and theorem statement
}
```

**Key characteristics:**
- Full proofs, not pre-traced tactic pairs — needs Pantograph tracing to extract tactic pairs
- Lean 4.9 — 17 minor versions behind our Pantograph
- ~14K theorems are net-new beyond InternLM's 15.7K proved subset (based on `problem_id` overlap)
- Generated by DeepSeek-Prover-V1.5 — same model family as our backbone, tactics in natural vocabulary
- **Risk:** `full_proof` may contain `import LeanWorkbook.Utils` or other non-Mathlib imports our env lacks
- **Risk:** Lean 4.9 → 4.26 may break some proofs due to Mathlib lemma renames

**Tracing approach:**
1. Parse `full_proof` to extract theorem statement + tactic sequence
2. Check for non-Mathlib imports (reject or fixup)
3. Replay through Pantograph to get tactic pairs with our Mathlib v4.26 goal states

#### Goedel → Lean 4.26 Migration (Phase M)

No one has ported Goedel proofs to Lean v4.20+. We plan to migrate the full 29.7K to Lean 4.26 and release as a community contribution.

**Why this works for competition math:** Core tactics (`nlinarith`, `ring`, `omega`, `norm_num`, `field_simp`) are stable across versions. The proofs are short (median ~5 tactics) with limited Mathlib surface area. Known 4.26 breaking changes mostly don't apply:
- `sorted` → `pairwise` renames — unlikely in competition math
- `Data.List.MinMax` typeclass changes — unlikely in these proofs
- `Nat.sqrt` definitional reduction changes — possible in number theory problems
- General Mathlib theorem name shuffles — main risk, handled by automated rename pass

Most breakage is in topology, category theory, and order theory areas. **Expected survival: 90-97%.** If below 90%, LEAN-GitHub supplements the gap.

**Migration pipeline (Phase M, ~4 days before Phase 0):**

| Day | Task | Deliverable |
|-----|------|-------------|
| M.0 | Clone Goedel, update toolchain to 4.26, attempt `lake build` | Error log, survival count |
| M.1 | Automated fixes (renames, instance patches), rebuild | Improved survival count |
| M.2 | Manual triage of remaining failures, drop or fix | ≥95% compilation (≥28,270 of 29,759) |
| M.3 | Port miniF2F-v2s/v2c statements to 4.26, verify all 488 compile | Eval benchmark ready |
| M.4 | LeanDojo trace on compiled Goedel 4.26 proofs | Tactic pairs parquet |
| M.5 | Download + filter LEAN-GitHub, merge with Goedel pairs | Unified SFT dataset |
| M.6 | Release Goedel-4.26 on HuggingFace, write migration notes | Community contribution |

**Detailed migration phases:**

```
Phase 1: Toolchain + bulk compile (M.0–M.1)
├── Clone Goedel-LM/Lean-workbook-proofs
├── Update lean-toolchain → leanprover/lean4:4.26.0
├── Update lakefile.lean Mathlib dependency → 4.26-compatible tag
├── lake update (resolve Reservoir deps)
├── lake build (attempt bulk compile)
└── Log: compiled/total, error categories

Phase 2: Automated fixes (M.1–M.2)
├── Parse compilation errors
├── Apply known renames (sorted→pairwise, etc.)
├── Fix Nat/Int instance changes
├── Re-run lake build
└── Log: improvement count

Phase 3: Manual triage (M.2)
├── For remaining failures: categorize
│   ├── Mathlib API rename → regex fix
│   ├── Definitional change → manual proof patch
│   └── Fundamental incompatibility → drop from dataset
├── Target: ≥95% survival rate (≥28,270 of 29,759)
└── Document all changes for community PR

Phase 4: Trace + release (M.4–M.6)
├── Run LeanDojo trace on compiled 4.26 proofs
│   → Extract (state_before, tactic, state_after) pairs
├── Apply sorry filter (should find ~0 in already-proved corpus)
├── Package as Parquet: {theorem, state, tactic, depth, source}
├── Release on HuggingFace with:
│   ├── lean-toolchain version
│   ├── Mathlib commit hash
│   ├── Migration changelog
│   └── List of dropped theorems + reasons
└── Open PR upstream to Goedel-LM if they want it
```

### 2.2 NuminaMath-LEAN (AI-MO/NuminaMath-LEAN) — Deferred to Phase 2

**Source:** AI-MO team, competition math formalization.

**Schema (104K rows):**
```
{
  "uuid": str,                # Unique identifier
  "problem": str,             # Natural language problem statement
  "question_type": str,       # Problem type classification
  "answer": str,              # Expected answer
  "author": str,              # "human" or "autoformalizer"
  "formal_statement": str,    # Lean 4 formal statement (always present)
  "formal_ground_truth": str, # Human-written proof (only when author == "human")
  "ground_truth_type": str,   # "complete" or "with_sorry"
  "formal_proof": str,        # Machine-generated proof (from Kimina-Prover RL, when available)
  "rl_data": str,             # RL training metadata
  "source": str,              # Competition source
  "problem_type": str,        # Problem classification
  "exam": str,                # Competition name (e.g. "IMO", "USAMO", "AMC", "AIME")
}
```

**Critical filters (from data_format_spec.md):**
- `ground_truth_type == "with_sorry"` → **SKIP** (contains sorry, our filter catches this)
- `author == "autoformalizer"` with no `formal_proof` → statements only, no proof to trace
- `formal_proof` is present → can trace for tactic pairs (machine-generated by Kimina-Prover RL)
- `formal_ground_truth` with `ground_truth_type == "complete"` → **highest quality**, human-written

**Processing steps (Phase 2):**
1. Filter to rows that have either `formal_proof` or complete `formal_ground_truth`
2. Reconstruct full Lean 4 files (statement + proof)
3. Trace with LeanDojo to get (state, tactic) pairs
4. Lean version must match: compiled against Mathlib v4.15.0, migrated to 4.26

**Key characteristics:**
- 104K rows total, but many are statements-only (no proof available)
- Kimina Lean Server (Apr 2025) verified 9,419 sorry-free proofs from this dataset at v4.15
- Competition math from IMO, USAMO, AMC, AIME — directly relevant to miniF2F (use `exam` field for stratification)
- Mix of tactic-style and term-style proofs — term proofs can't be replayed step-by-step
- Two proof sources: `formal_proof` (machine-generated by Kimina-Prover RL) and `formal_ground_truth` (human-written, highest quality)
- Lean v4.15.0 / Mathlib v4.15.0 (confirmed from README) — 11 minor versions behind our v4.26, closest of the three
- `ground_truth_type` must be checked: "with_sorry" rows are contaminated and must be filtered

**Deferral rationale:** Goedel + LEAN-GitHub already gives ~210-350K pairs. NuminaMath uses Lean 4.15 (fewer migration issues), and the proofs were machine-generated by Kimina-Prover, adding yet another style. Will be integrated in Phase 2 after Goedel is validated.

### 2.3 LEAN-GitHub (internlm/Lean-Github)

**Source:** `internlm/Lean-Github` on HuggingFace — tactic-level extraction from open-source Lean 4 GitHub repos using LeanDojo.

**Schema (218,866 tactic steps from 28,597 theorems):**
```
{
  "url":          str,   # GitHub repo URL
  "commit":       str,   # pinned commit
  "file_path":    str,   # e.g. "Mathlib/Topology/Basic.lean"
  "full_name":    str,   # theorem name (e.g. "rot_add_assoc")
  "start":        str,   # source location [line, col]
  "end":          str,   # source location
  "tactic":       str,   # the tactic text
  "state_before": str,   # proof state before tactic (pretty-printed)
  "state_after":  str,   # proof state after tactic
}
```

**Key characteristics:**
- **Already in (state, tactic) pair format** — no LeanDojo tracing needed
- Can directly convert to our SFT format: `format_sft_pair(row["state_before"], row["tactic"])`
- Repos span multiple Lean versions. Tactic states are already extracted as strings, so they're version-agnostic for SFT training purposes
- **Human-written proof diversity** that Goedel lacks — custom lemmas, `have` chains, `calc` blocks, domain-specific tactics vs Goedel's machine-generated repetitive `nlinarith`/`ring` style

**Integration challenges:**
1. **Quality filtering needed**: Some repos are educational (trivial `rfl`/`simp` on toy types). Some `state_before` strings up to 2MB. Need filtering.
2. **Deduplication with Goedel**: LEAN-GitHub includes some Mathlib proofs. Check overlap on `full_name`.
3. **Lean version mismatch**: Pre-traced strings are version-agnostic for SFT. Re-tracing on 4.26 would be a separate effort if needed.

**Filtering strategy:**
```python
def should_include(row):
    state = row["state_before"]
    tactic = row["tactic"]

    # Skip trivially long states (likely generated / expanded)
    if len(state) > 4096:
        return False

    # Skip trivial single-token tactics (for diversity)
    # But keep them at 10% rate for calibration
    if tactic.strip() in {"rfl", "simp", "trivial", "exact?", "decide"}:
        return random.random() < 0.1

    # Skip if state is "no goals" (shouldn't exist in state_before but check)
    if "no goals" in state:
        return False

    return True
```

**Expected yield after filtering:** ~100-150K usable tactic pairs.

---

## 3. Version Compatibility Matrix

| Component | Lean Version | Mathlib Version | Notes |
|-----------|-------------|-----------------|-------|
| **Our Pantograph** | **v4.26.0** | **v4.26.0** | Source of truth |
| Goedel Proofs | v4.9 → **v4.26 (Phase M)** | Unknown → 4.26-compatible | Phase M migrates all 29.7K proofs |
| LEAN-GitHub | Mixed | Mixed | Pre-traced strings are version-agnostic for SFT |
| NuminaMath | v4.15.0 | Mathlib v4.15.0 | 11 minor versions behind — closest. Phase 2 migration |
| LeanDojo Benchmark 4 | v4.3.0+ → **v4.19.0** | Mathlib4 | v4.19 traced version on Zenodo (May 2025). Abstract math, not competition. |
| miniF2F-lean4 | varies → **v4.21.0** | Mathlib4 | Eval benchmark only (244 problems). Closest to our v4.26. |
| miniF2F-v2s/v2c | evaluated at v4.9.0 | — | Phase M.3 ports statements to 4.26 |
| IMO-Steps | v4.17.0 | — | 20 problems, 1.3K lemmas |

### Version Drift Risks

**Primary risk: Mathlib lemma renames.** Between Lean v4.9 and v4.26 (17 minor versions of development):
- Hundreds of Mathlib lemmas were renamed for consistency
- Example: `Nat.foo_bar` → `Nat.bar_foo` or namespace reorganizations
- Tactic `exact Nat.foo_bar` in v4.9 data fails if lemma was renamed in v4.26
- `simp` and `omega` tactics are more robust (work on goal structure, not lemma names)
- Phase M automated rename pass handles most of these

**Secondary risk: Tactic API changes.** Less likely but possible:
- New tactic options or changed defaults
- Pretty-printer format changes (goal text differs even when tactic succeeds)

**Mitigation strategy:**
1. Phase M migration proactively ports Goedel proofs to 4.26 before tracing — bulk compile validates survival
2. LEAN-GitHub pre-traced strings are version-agnostic for SFT — sidesteps version issues entirely
3. Re-extract goal states from Pantograph after migration to ensure consistent `state_before` format
4. NuminaMath (Phase 2): migrate from v4.15 to v4.26 using same Phase M pipeline — fewer renames expected

---

## 4. Version Port Status (Researched 2026-02-26)

**No one has ported any of our target datasets to Lean v4.20+.** The entire field is stuck at old versions — even Goedel-Prover-V2 and DeepSeek-Prover-V2 (both 2025) still use Lean 4.9.

### Port Status by Dataset

| Dataset | Current Version | Any Newer Port? | Notes |
|---------|----------------|-----------------|-------|
| Goedel Proofs | v4.9.0 | **NO** → **PLANNED** | Our Phase M migrates to 4.26; Goedel-V2 achieved 88% miniF2F but stayed at v4.9 |
| LEAN-GitHub | Mixed | N/A | Pre-traced strings, no compilation needed for SFT |
| NuminaMath-LEAN | v4.15.0 | **NO** | Cleaned versions exist (ChristianZ97, iiis-lean) but same Lean version |
| LeanDojo Benchmark 4 | varies | **v4.19.0** (Zenodo, May 2025) | Abstract math, not competition-focused |
| miniF2F-lean4 | varies | **v4.21.0** (yangky11, Nov 2024) | Eval only, 244 problems |
| miniF2F-v2s/v2c | v4.9.0 | **NO** → **PLANNED** | Phase M.3 ports to 4.26 |
| IMO-Steps | v4.17.0 | **NO** | 20 problems, 1.3K lemmas |

### Implications for Our Pipeline

1. **NuminaMath at v4.15 is our best bet for easy porting** — only 11 versions behind (not 17). Mathlib renames between v4.15→v4.26 are significantly fewer than v4.9→v4.26.
2. **Goedel (v4.9) has the largest version gap** — expect substantial breakage from Mathlib lemma renames over 17 minor versions. Phase M migration addresses this with automated rename pass + manual triage.
3. **LEAN-GitHub sidesteps version issues entirely** — pre-traced strings are used as-is for SFT, making it the easiest data source to integrate.
4. **LeanDojo-v2** can retrace any Lean repo at any version, but the source code must compile first at the target version.
5. **LeanInteract** (Python, `augustepoiroux/LeanInteract`) supports Lean v4.8 through v4.29 — potential multi-version validation tool.

### Alternative Strategy Considered

Running Pantograph at an older Lean version (e.g., v4.15 matching NuminaMath) instead of porting data forward. **Rejected** — conflicts with our existing Mathlib v4.26 setup and miniF2F eval environment.

---

## 5. Overlap Analysis

### 5.1 Lean Workbook ↔ Goedel Proofs (Resolved)

- Both reference the same problem set via `problem_id` / `id`
- Lean Workbook had 25.2K pre-traced tactic pairs from ~15.7K proved theorems (at v4.8)
- Goedel has 29.8K proofs — a strict superset of the Lean Workbook proved subset
- **Decision:** Lean Workbook tactic pairs are deprecated. Phase M migrates all 29.7K Goedel proofs to 4.26 and traces them with LeanDojo, producing tactic pairs at the correct Lean version. This completely supersedes the InternLM pre-traced pairs.

### 5.2 Goedel ↔ NuminaMath

- Different problem sources: Goedel from competition practice (Lean Workbook), NuminaMath from actual competitions
- Minimal overlap expected (different formalization efforts)
- NuminaMath provides competition metadata (`exam` field: competition name, year) useful for stratified evaluation

### 5.3 LEAN-GitHub ↔ Goedel

- LEAN-GitHub includes some Mathlib proofs and other GitHub repos
- Goedel proofs are all from the Lean Workbook problem set
- Overlap check: match on `full_name` field in LEAN-GitHub against Goedel `problem_id`
- Expected overlap: low (LEAN-GitHub is scraped from diverse GitHub repos, not the Workbook)
- **Decision:** Deduplicate by source-prefixed theorem name during combine step (Task 0.6)

### 5.4 InternLM-Math-Plus / StepProver Overlap

- The Lean Workbook tactic pairs (now deprecated) may have included data from InternLM-Math-Plus and StepProver
- Not a concern for v2.1 since we use Goedel proofs directly, not the InternLM pre-traced pairs
- If any InternLM data is used in future as validation cross-check, dedup by `(theorem_id, tactic_index)` pair

---

## 6. Data Pipeline Decisions

### Immediate SFT Candidates (no tracing needed)

1. **LEAN-GitHub tactic pairs** — pre-traced, no tracing needed
   - Apply quality filtering (state length < 4096, trivial tactic subsampling)
   - Source-prefixed dedup against Goedel to avoid name collisions
   - Convert to DeepSeek-native prompt format (see `docs/data_format_spec.md`)

### Needs Tracing

2. **Goedel Workbook proofs (all 29.7K)** — full proofs migrated to 4.26 in Phase M, then LeanDojo traced to extract tactic pairs. Expected ~28K+ surviving proofs → ~110-140K tactic pairs.
3. **NuminaMath tactic-style proofs** — deferred to Phase 2; filter to rows with `formal_proof` or complete `formal_ground_truth`, migrate from 4.15 to 4.26, then LeanDojo trace
4. **NuminaMath term-style proofs** — cannot be replayed step-by-step, may need different approach

### Quality Gates

- Sorry/admit/cheat filter on all sources
- Depth ≥ 3 subset for contrastive pool
- Validation split by theorem name (not tactic pairs)
- Phase M bulk compile validates Goedel proofs on 4.26 before tracing
- LEAN-GitHub: state length < 4096, trivial tactic subsampling, dedup against Goedel
- NuminaMath: reject `ground_truth_type == "with_sorry"`, require `formal_proof` or complete `formal_ground_truth`

### Tracing Priority

1. Goedel Workbook proofs (29.7K, Phase M migration to 4.26 → LeanDojo trace → ~110-140K pairs)
2. LEAN-GitHub pre-traced pairs (immediate — quality filter + format convert → ~100-150K pairs)
3. NuminaMath tactic-style proofs (Phase 2 — migrate from 4.15, competition metadata for stratification)

---

## 7. Evaluation Protocol

### Primary evaluation: miniF2F-v2s

| Benchmark | Source | Format | Lean Version | Size |
|-----------|--------|--------|-------------|------|
| miniF2F v1 (backward compat) | yangky11/miniF2F-lean4 | .lean files | 4.9.0 | 488 (244 test + 244 valid) |
| miniF2F-v2s (primary) | roozbeh-yz/miniF2F_v2 | JSONL | evaluated at 4.9.0 | 488 |
| miniF2F-v2c (stretch) | roozbeh-yz/miniF2F_v2 | JSONL | evaluated at 4.9.0 | 488 |

**v2s** = formal statements corrected + solutions included in informal statements. All 16 previously unprovable statements fixed. Simpler than v2c (no multi-choice structural challenges) but harder than v1 (oversimplifications reverted).

### Published baselines (whole-proof @32)

| Model | v1 test | v2s test | v2c test |
|-------|---------|----------|----------|
| DeepSeek-Prover-V1.5-RL | 50.0% | 41.0% | 38.1% |
| Goedel-Prover-SFT | 58.2% | 48.4% | 46.3% |
| DeepSeek-Prover-V2-7B | 73.4% | 68.1% | 64.4% |

Our current system: **41.7% on v1 test** (tactic-level search, 800 nodes). Expected v2s performance: ~33-36% (roughly 80% of v1 score based on the pattern above).

### Metrics reported at each iteration

| Metric | Benchmark | How |
|--------|-----------|-----|
| `v1_test` | miniF2F v1 (244 problems) | Pantograph search, 800 nodes |
| `v2s_test` | miniF2F-v2s (244 problems) | Pantograph search, 800 nodes |
| `v2c_test` | miniF2F-v2c (244 problems) | Pantograph search, 800 nodes |

### Baseline expectations (iter_0, before EBM)

| Metric | Expected | Rationale |
|--------|----------|-----------|
| v1_test | ~45-50% | Larger SFT data + 4.26 should improve over current 41.7% |
| v2s_test | ~36-42% | ~80-85% of v1 score based on published ratios |
| v2c_test | ~33-38% | ~75-80% of v1 score |

### Success criteria for v2 (EBM contribution)

The EBM is successful if:
- `v2s_test` improves by ≥3% absolute over LLM-only baseline at same node budget
- OR equivalent `v2s_test` with ≤50% of the node budget

---

## 8. Updated Data Pipeline

### Pipeline comparison (v2 → v2.1)

**Before (v2 plan):**
```
Goedel (29.7K) + NuminaMath (~10-20K)
    ↓ LeanDojo trace on 4.9.0/4.15.0
(state, tactic) pairs → SFT training
    ~80-120K pairs
```

**After (v2.1 plan):**

### Pipeline diagram

```
                    Lean 4.26 toolchain
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                  ▼
   Goedel 4.26        LEAN-GitHub        miniF2F-v2s
   (migrate proofs)   (pre-traced,       (port statements
    29.7K → ~28K+     filter for SFT)    to 4.26 for eval)
         │                 │
         ▼                 ▼
   LeanDojo trace     Direct use
   on 4.26            (already state,tactic)
         │                 │
         └────────┬────────┘
                  ▼
         Unified tactic pairs
         ~210-350K pairs
         {theorem, state, tactic, depth, source}
                  │
         ┌───────┴────────┐
         ▼                ▼
    SFT training     Contrastive pool
    (DeepSeek-V2     (for EBM after
     native format)   iter_0 search)
```

### Data source breakdown (estimated)

| Source | Raw pairs | After filtering | Notes |
|--------|-----------|----------------|-------|
| Goedel 4.26 | ~120-150K | ~110-140K | LeanDojo traced, sorry-filtered |
| LEAN-GitHub | ~219K | ~100-150K | Pre-traced, quality-filtered |
| NuminaMath (Phase 2) | ~50-80K | ~40-60K | Deferred; after Goedel validated |
| **Total** | | **~210-350K** | |

### Train/val split

Split by theorem name (Gotcha #11). Since we're combining sources:

```python
# Combine all theorems, prefix with source to avoid name collisions
all_theorems = (
    [f"goedel::{t}" for t in goedel_theorems] +
    [f"github::{t}" for t in github_theorems]
)
random.shuffle(all_theorems)
train_theorems = set(all_theorems[:int(len(all_theorems) * 0.95)])
```

---

## 9. NuminaMath Migration (Phase 2)

After Goedel is validated on 4.26, apply the same pipeline to NuminaMath-LEAN:

```
AI-MO/NuminaMath-LEAN (104K formalized)
    ↓ Filter: formal_proof present, no sorry
    ~10-20K provable
    ↓ Update lean-toolchain → 4.26
    ↓ Rebuild, apply same migration fixes
    ↓ LeanDojo trace
    → Additional ~50-80K tactic pairs
```

This is lower priority because:
1. Goedel + LEAN-GitHub already gives ~210-350K pairs (target: ~250K usable)
2. NuminaMath uses Lean 4.15 (more recent, fewer migration issues than Goedel's 4.9)
3. But the proofs were machine-generated by Kimina-Prover, adding yet another style
4. Only ~9.4K sorry-free proofs verified (Kimina Lean Server) — much smaller yield than expected from 104K rows

---

## 10. Community Contribution Checklist

### What to release

- [ ] `{org}/Lean-workbook-proofs-4.26` on HuggingFace
  - Parquet: full proofs (like original)
  - Parquet: tactic-level pairs (from LeanDojo trace)
  - `lean-toolchain` version
  - Mathlib commit hash
  - Migration changelog
  - List of dropped theorems + failure reasons

- [ ] `{org}/miniF2F-v2-lean426` on GitHub
  - miniF2F-v2s and v2c formal statements ported to 4.26
  - Verification script
  - Any patches needed

- [ ] `{org}/NuminaMath-LEAN-4.26` on HuggingFace (Phase 2)
  - Same structure as Goedel release
  - Filter: `formal_proof` or complete `formal_ground_truth` present, no sorry
  - Document `ground_truth_type` filtering and survival rate

### What to document

- Migration script (automated renames)
- Failure categorization (what breaks and why)
- Survival statistics by error type
- Recommendations for others migrating to 4.26