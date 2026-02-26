# Lean 4 Training Datasets — Survey & Pipeline Plan

Last updated: 2026-02-26

## Overview

This document catalogs all known Lean 4 theorem-proving datasets on HuggingFace and beyond, evaluates them for our competition-focused v2 training pipeline, and documents format/compatibility details for our four target datasets.

**Our setup:** Pantograph on Lean v4.26.0 / Mathlib v4.26.0. DeepSeek-Prover-V2-7B backbone. SFT format: DeepSeek-native prompt with tactic state as Lean comment (see `docs/data_format_spec.md`).

**v2.1 additions:** LEAN-GitHub (218K pre-traced tactic pairs) promoted to high-relevance target. Goedel Workbook proofs migrated to Lean 4.26 (new Phase M). miniF2F-v2s as primary evaluation benchmark. NuminaMath deferred to Phase 2.

---

## 1. Complete Lean 4 Dataset Landscape

### 1.1 Our Four Target Datasets

| Dataset | HF Path | Rows | Lean Version | Format | License |
|---------|---------|------|-------------|--------|---------|
| Lean Workbook (InternLM) | `internlm/Lean-Workbook` | 25.2K tactic pairs | v4.8.0-rc1 | Pre-traced tactic pairs (`state_before`, `tactic`, `state_after`) | Apache 2.0 |
| Goedel Workbook Proofs | `Goedel-LM/Lean-workbook-proofs` | 29.8K full proofs | Lean 4.9 | Full proofs (`problem_id`, `full_proof`) | Apache 2.0 |
| LEAN-GitHub | `internlm/Lean-Github` | 28.6K theorems, 218K tactics | Mixed (pre-4.26) | Pre-traced tactic pairs (`state_before`, `tactic`, `state_after`) | Apache 2.0 |
| NuminaMath-LEAN | `AI-MO/NuminaMath-LEAN` | 104K rows | v4.15.0 (Mathlib v4.15.0) | Formal statements + proofs (`formal_statement`, `formal_proof`) | Apache 2.0 |

### 1.2 Other Lean 4 Datasets

| Dataset | HF Path | Size | Lean Version | Format | Notes |
|---------|---------|------|-------------|--------|-------|
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
| Lean Workbook (full) | `internlm/Lean-Workbook` | 57K + 83K problems | v4.8.0-rc1 | Problem statements (many unproved) | The 25.2K tactic pairs are a proved subset. |
| InternLM-Math-Plus | Part of Lean-Workbook | Overlaps with Lean Workbook | v4.8.0-rc1 | Extended annotations | StepProver data also overlaps. Check dedup. |

### 1.3 Relevance Assessment

**High relevance (competition math, tactic-style, Lean 4):**
- Lean Workbook tactic pairs — pre-traced, competition-focused, closest to our SFT format
- Goedel Workbook proofs — large, competition-focused, needs tracing
- LEAN-GitHub — 218K pre-traced tactic pairs from 28.6K theorems, human-written proof diversity that complements machine-generated Goedel proofs. Already in (state, tactic) format — no tracing needed.
- NuminaMath-LEAN — IMO/USAMO/AMC/AIME, largest collection, needs quality audit (deferred to Phase 2)

**Medium relevance:**
- LeanDojo Benchmark 4 — proven pipeline but wrong domain (Mathlib abstract math)
- AI4M less-proofnet-lean4 — huge but unknown quality, needs investigation

**Low relevance for v2:**
- Herald — NL↔formal pairs, not tactic proofs
- DeepSeek-Prover-V1 — superseded
- Kimina-Prover-Promptset — small, unknown provenance

---

## 2. Target Dataset Details

### 2.1 Lean Workbook (internlm/Lean-Workbook)

**Source:** InternLM team, derived from competition math problems.

**Schema (tactic pairs subset, 25.2K rows):**
```
{
  "url": str,              # HuggingFace problem URL
  "id": str,               # e.g. "lean_workbook_0"
  "state_before": str,     # Goal state before tactic
  "tactic": str,           # Applied tactic
  "state_after": str,      # Goal state after tactic (or "no goals" if proved)
}
```

**Key characteristics:**
- Pre-traced tactic pairs — potentially usable for immediate SFT without re-tracing
- Lean v4.8.0-rc1 — 18 minor versions behind our Pantograph (v4.26.0)
- Competition math focus — algebra, number theory, combinatorics
- Multiple tactic pairs per theorem (one per proof step)
- `state_before` format needs validation against our DeepSeek-native prompt format (see `docs/data_format_spec.md`)
- Overlap with InternLM-Math-Plus and StepProver data — need dedup check

**Format validation needed:**
- Does `state_before` match Pantograph's pretty-printed goal format?
- Depth distribution: how many theorems have depth ≥ 3? (needed for contrastive pool)
- Sorry/admit contamination rate

### 2.2 Goedel Workbook Proofs (Goedel-LM/Lean-workbook-proofs)

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

**Why this works for competition math:** Core tactics (`nlinarith`, `ring`, `omega`, `norm_num`, `field_simp`) are stable across versions. Breaking changes (`sorted` → `pairwise`, `Data.List.MinMax` typeclass changes) affect topology/category theory, not competition algebra/number theory. **Expected survival: 90-97%.**

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

### 2.3 NuminaMath-LEAN (AI-MO/NuminaMath-LEAN) — Deferred to Phase 2

**Source:** AI-MO team, competition math formalization.

**Schema (104K rows):**
```
{
  "problem": str,           # Natural language problem statement
  "formal_statement": str,  # Lean 4 formal statement
  "formal_proof": str,      # Lean 4 proof (may be empty)
  "ground_truth_type": str, # "complete", "partial", etc.
  "competition": str,       # "IMO", "USAMO", "AMC", "AIME", etc.
  "year": int,              # Competition year
}
```

**Key characteristics:**
- ~70K+ rows with non-empty `formal_proof` (needs exact count)
- Competition math from IMO, USAMO, AMC, AIME — directly relevant to miniF2F
- Mix of tactic-style and term-style proofs — term proofs can't be replayed step-by-step
- Lean v4.15.0 / Mathlib v4.15.0 (confirmed from README) — 11 minor versions behind our v4.26, closest of the three
- Largest single source of competition-focused Lean 4 proofs
- Kimina Lean Server (Apr 2025) verified 9,419 sorry-free proofs from this dataset at v4.15
- `ground_truth_type` distribution needs auditing — "complete" vs "partial" vs other

**Deferral rationale:** Goedel + LEAN-GitHub already gives ~210-350K pairs. NuminaMath uses Lean 4.15 (fewer migration issues), and the proofs were machine-generated by Kimina-Prover, adding yet another style. Will be integrated in Phase 2 after Goedel is validated.

### 2.4 LEAN-GitHub (internlm/Lean-Github)

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
| Lean Workbook | v4.8.0-rc1 | Unknown | 18 minor versions behind |
| Goedel Proofs | v4.9 | Unknown | 17 minor versions behind; Phase M migrates to 4.26 |
| LEAN-GitHub | Mixed | Mixed | Pre-traced strings are version-agnostic for SFT |
| NuminaMath | v4.15.0 | Mathlib v4.15.0 | 11 minor versions behind — closest of the three |
| LeanDojo Benchmark 4 | v4.3.0+ → **v4.19.0** | Mathlib4 | v4.19 traced version on Zenodo (May 2025). Abstract math, not competition. |
| miniF2F-lean4 | varies → **v4.21.0** | Mathlib4 | Eval benchmark only (244 problems). Closest to our v4.26. |
| miniF2F-v2s/v2c | evaluated at v4.9.0 | — | Phase M.3 ports statements to 4.26 |

### Version Drift Risks

**Primary risk: Mathlib lemma renames.** Between Lean v4.8 and v4.26 (18 months of development):
- Hundreds of Mathlib lemmas were renamed for consistency
- Example: `Nat.foo_bar` → `Nat.bar_foo` or namespace reorganizations
- Tactic `exact Nat.foo_bar` in v4.8 data fails if lemma was renamed in v4.26
- `simp` and `omega` tactics are more robust (work on goal structure, not lemma names)

**Secondary risk: Tactic API changes.** Less likely but possible:
- New tactic options or changed defaults
- Pretty-printer format changes (goal text differs even when tactic succeeds)

**Mitigation strategy:**
1. Pantograph validation (Tasks 0.3d/e/f) measures actual failure rate
2. If tactic success ≥ 80%: use pre-traced data, accept some loss
3. If tactic success < 80%: must re-trace under our Mathlib v4.26
4. Even if tactics succeed, re-extract goal states from Pantograph to ensure consistent `state_before` format
5. Phase M migration proactively ports Goedel proofs to 4.26 before tracing

---

## 4. Version Port Status (Researched 2026-02-26)

**No one has ported any of our target datasets to Lean v4.20+.** The entire field is stuck at old versions — even Goedel-Prover-V2 and DeepSeek-Prover-V2 (both 2025) still use Lean 4.9.

### Port Status by Dataset

| Dataset | Current Version | Any Newer Port? | Notes |
|---------|----------------|-----------------|-------|
| Lean Workbook | v4.8.0-rc1 | **NO** | Paper revised Jun 2025, version unchanged |
| Goedel Proofs | v4.9.0 | **NO** → **PLANNED** | Our Phase M migrates to 4.26; Goedel-V2 achieved 88% miniF2F but stayed at v4.9 |
| LEAN-GitHub | Mixed | N/A | Pre-traced strings, no compilation needed for SFT |
| NuminaMath-LEAN | v4.15.0 | **NO** | Cleaned versions exist (ChristianZ97, iiis-lean) but same Lean version |
| LeanDojo Benchmark 4 | varies | **v4.19.0** (Zenodo, May 2025) | Abstract math, not competition-focused |
| miniF2F-lean4 | varies | **v4.21.0** (yangky11, Nov 2024) | Eval only, 244 problems |
| miniF2F-v2s/v2c | v4.9.0 | **NO** → **PLANNED** | Phase M.3 ports to 4.26 |
| IMO-Steps | v4.17.0 | **NO** | 20 problems, 1.3K lemmas |

### Implications for Our Pipeline

1. **NuminaMath at v4.15 is our best bet for easy porting** — only 11 versions behind (not 18). Mathlib renames between v4.15→v4.26 are significantly fewer than v4.8→v4.26.
2. **Lean Workbook and Goedel (v4.8/4.9) have the largest version gap** — expect substantial breakage from Mathlib lemma renames over 17-18 minor versions. Phase M migration addresses this for Goedel.
3. **LEAN-GitHub sidesteps version issues entirely** — pre-traced strings are used as-is for SFT, making it the easiest data source to integrate.
4. **LeanDojo-v2** can retrace any Lean repo at any version, but the source code must compile first at the target version.
5. **LeanInteract** (Python, `augustepoiroux/LeanInteract`) supports Lean v4.8 through v4.29 — potential multi-version validation tool.

### Alternative Strategy Considered

Running Pantograph at an older Lean version (e.g., v4.15 matching NuminaMath) instead of porting data forward. **Rejected** — conflicts with our existing Mathlib v4.26 setup and miniF2F eval environment.

---

## 5. Overlap Analysis

### 5.1 Lean Workbook ↔ Goedel Proofs

- Both reference the same problem set via `problem_id` / `id`
- Lean Workbook has 25.2K tactic pairs from ~15.7K proved theorems
- Goedel has 29.8K proofs
- Overlap: ~15.7K theorems appear in both → ~14K net-new from Goedel
- For overlapping theorems: Goedel proofs may differ from InternLM proofs (different provers)
- **Decision:** Use both, but deduplicate by theorem. For overlaps, prefer InternLM pre-traced pairs (less work), use Goedel for net-new theorems.

### 5.2 Lean Workbook ↔ NuminaMath

- Different problem sources: Lean Workbook from competition practice, NuminaMath from actual competitions
- Minimal overlap expected (different formalization efforts)
- NuminaMath provides competition metadata (competition name, year) useful for stratified evaluation

### 5.3 LEAN-GitHub ↔ Goedel

- LEAN-GitHub includes some Mathlib proofs and other GitHub repos
- Goedel proofs are all from the Lean Workbook problem set
- Overlap check: match on `full_name` field in LEAN-GitHub against Goedel `problem_id`
- Expected overlap: low (LEAN-GitHub is scraped from diverse GitHub repos, not the Workbook)
- **Decision:** Deduplicate by source-prefixed theorem name during combine step (Task 0.6)

### 5.4 InternLM-Math-Plus / StepProver Overlap

- The Lean Workbook tactic pairs may include data from InternLM-Math-Plus and StepProver
- Need to check if our download includes duplicates from these sub-sources
- Dedup by `(theorem_id, tactic_index)` pair

---

## 6. Data Pipeline Decisions

### Immediate SFT Candidates (no tracing needed)

1. **Lean Workbook tactic pairs** — if Pantograph validation (0.3d) shows ≥80% tactic success
   - Convert `state_before`/`tactic` to DeepSeek-native prompt format (see `docs/data_format_spec.md`)
   - Apply sorry/admit/cheat filter (Gotcha 12)
   - Split by theorem name (Gotcha 11)

2. **LEAN-GitHub tactic pairs** — pre-traced, no tracing needed
   - Apply quality filtering (state length < 4096, trivial tactic subsampling)
   - Source-prefixed dedup against Goedel to avoid name collisions
   - Convert to DeepSeek-native prompt format

### Needs Tracing

3. **Goedel net-new proofs (~14K)** — full proofs need Pantograph replay (after Phase M migration to 4.26)
4. **NuminaMath tactic-style proofs** — deferred to Phase 2; full proofs need Pantograph replay
5. **NuminaMath term-style proofs** — cannot be replayed step-by-step, may need different approach

### Quality Gates

- Sorry/admit/cheat filter on all sources
- Depth ≥ 3 subset for contrastive pool
- Validation split by theorem name (not tactic pairs)
- Version compatibility check via Pantograph replay (for Goedel/NuminaMath)

### Tracing Priority (if needed)

1. Lean Workbook pre-traced pairs (immediate if validation passes)
2. LEAN-GitHub pre-traced pairs (immediate — quality filter + format convert)
3. Goedel net-new proofs (14K, competition-focused, after Phase M migration)
4. NuminaMath tactic-style proofs (Phase 2 — largest source, competition metadata)

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
1. Goedel + LEAN-GitHub already gives ~210-350K pairs
2. NuminaMath uses Lean 4.15 (more recent, fewer migration issues)
3. But the proofs were machine-generated by Kimina-Prover, adding yet another style

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

### What to document

- Migration script (automated renames)
- Failure categorization (what breaks and why)
- Survival statistics by error type
- Recommendations for others migrating to 4.26
