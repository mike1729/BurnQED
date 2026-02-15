# EBM Overhaul: Architecture Upgrade + Generate-Negatives Pipeline

Saved from planning session 2026-02-15. Parts 2 and 3 implemented 2026-02-15.

## Part 2: EnergyHead Architecture Upgrade (DONE)

```
Old: 4096 → 512 → 256 → 1     (~2.2M params, 3 layers, k=4)
New: 4096 → 2048 → 1024 → 512 → 1  (~11M params, 4 layers, k=7)
```

### Files

| File | Change |
|------|--------|
| `crates/ebm/src/model/energy_head.rs` | Add `d_hidden3` config field (default 512), `sn_linear4`, `dropout3`, update forward pass, update doc |
| `crates/ebm/src/training/trainer.rs` | Change `k_negatives` default: 4 → 7 |
| `crates/prover-core/src/main.rs` | Change `k_negatives` CLI default: 4 → 7 |
| `crates/ebm/src/model/energy_head.rs` tests | Update `test_parameter_count` to ~11M |

**Breaking:** Old 3-layer checkpoints incompatible. Must retrain from scratch.

---

## Part 3: Generate-Negatives Pipeline (DONE)

Walks LeanDojo proof traces in Pantograph, generates LLM candidates at each step. Classifies results into three categories:

```
depth 0:  state₀ ──[ground-truth tactic]──→ state₁  (POSITIVE, depth=0)
              ├──[alt tactic A]──→ stateₐ ──→ goals=[] (POSITIVE — alternative proof!)
              ├──[alt tactic C]──→ stateₓ ──→ goals≠[] (NEGATIVE — diverges)
              └──[alt tactic B]──→ ✗ failed             (skip)
depth 1:  state₁ ──[ground-truth tactic]──→ state₂  (POSITIVE, depth=1)
              └──[alt tactic D]──→ state_d ──→ goals≠[] (NEGATIVE)
...
```

### Critical Design Decisions

**1. False Negative Prevention:** When an LLM-generated tactic differs from ground-truth but succeeds in Lean, we must check `goals.is_empty()` on the resulting state. If goals are empty → proof complete → label as **Positive** (alternative proof), NOT Negative. Only label as Negative when the tactic succeeds but leaves remaining goals (divergent path).

**2. Namespace Mismatch Mitigation:** Pantograph's `copyFrom` creates a goal from the theorem's type but runs in a base namespace context — no `open` directives from the original Lean file. LeanDojo ground-truth tactics often use short names (e.g., `exact X_sub_C_ne_zero z` instead of `exact Polynomial.X_sub_C_ne_zero z`). Mitigation:
   - Try ground-truth tactic as-is first
   - On failure, retry with `open <inferred_namespace> in <tactic>` where namespace is derived from the theorem's fully qualified name (e.g., `Polynomial.natDegree_cyclotomic'` → `open Polynomial in <tactic>`)
   - Accept ~20-30% skip rate — with 60K available theorems, even 30% failure gives ~42K usable theorems
   - **Note:** LLM-generated candidates work from the pretty-printed proof state (fully qualified names), so they're less affected by namespace issues

### New Subcommand

```
cargo run -p prover-core -- generate-negatives \
    --tactic-pairs data/tactic_pairs/train.jsonl \
    --server-url http://localhost:30000 \
    --encode-url http://localhost:30001 \
    --output negatives.parquet \
    --num-theorems 5000 \
    --candidates-per-step 8 \
    --target-negatives 15 \
    --temperature 1.0
```

### Pipeline Logic

`run_generate_negatives()` in `crates/prover-core/src/pipeline.rs`:

1. Load tactic pairs, group by theorem → `HashMap<String, Vec<TacticStep>>` sorted by depth
2. Filter: `--min-steps N` skips theorems with fewer than N proof steps
3. Sample `--num-theorems` theorems
4. For each theorem (concurrent via `JoinSet`, bounded by semaphore):
   a. Start proof in Pantograph (`copyFrom` by name)
   b. Record ALL ground-truth steps as Positive **upfront** (guarantees balanced depth distribution regardless of Pantograph replay success)
   c. Walk proof path depth by depth:
      - **Normal path** (`is_zombie == false`):
        - Generate LLM candidates via SGLang
        - Normalize whitespace before comparing with ground-truth
        - For each non-ground-truth candidate that succeeds in Lean:
          - If `goals.is_empty()` → **Positive** (alternative proof found)
          - If `goals.len() > 0` → **Negative** (divergent path)
          - Multi-tactic chains: walk sequentially, relabel all as Positive if chain completes proof
        - Try 18 **probe tactics** (simp, ring, omega, norm_num, constructor, left, right, intro _, etc.):
          - Cheap (no LLM call), deduped against LLM candidates
          - ProofComplete/empty goals → Positive; non-empty goals → **Negative**
          - Save last successful probe state for potential advancement
        - Apply ground-truth tactic to advance to next depth
          - On failure: retry with `open <all_namespace_prefixes> in <tactic>`
          - If still fails → **enter zombie walk** (if a probe state is available)
      - **Zombie path** (`is_zombie == true`):
        - **Skip LLM candidates** (avoids unreliable hard negatives, saves GPU)
        - Try probe tactics only:
          - Non-empty goals → **Positive** (valid context for representation learning)
          - Save last successful probe state for advancement
        - Advance via probe state; stop when no probe can advance
   d. Stop early if `target_negatives` reached
5. Write Parquet via `TrajectoryWriter` (auto-save every 50 theorems)
6. Print summary: depth distribution table, survival rate

### Zombie Walk Design

When ground-truth replay fails (~98% of theorems due to namespace mismatch), the pipeline doesn't stop — it "resurrects" using a probe tactic's resulting state.

**Safe labeling rules** (avoids confusing the EBM with false value signals):
- **Normal path negatives** = Hard Negatives: "This divergent tactic from the LLM is worse than ground-truth." (Reliable signal — ground-truth is known-good.)
- **Zombie path positives** = Context: "This state belongs to theorem A." (Representation learning — the EBM learns what theorem A's states look like at various depths, not which tactic is better.)
- **Never generate Hard Negatives on zombie paths**: No LLM candidates → no "intro > induction" false signals.

```
Step 0 (normal):  GT state₀ ──[LLM candidates]──→ Negatives at depth 1
                             ──[probe tactics]──→ Negatives at depth 1
                  GT fails → pick probe state → is_zombie = true

Step 1 (zombie):  Probe state ──[probe tactics]──→ Positives at depth 2
                               pick probe state → advance

Step 2 (zombie):  Probe state ──[probe tactics]──→ Positives at depth 3
                               no probe advances → stop
```

### Helper Functions

`infer_open_prefix(theorem_name: &str) -> Option<String>`

Opens **all namespace prefixes** from a fully qualified theorem name:
- `Polynomial.natDegree_cyclotomic'` → `"open Polynomial in "`
- `MeasureTheory.Measure.hahn` → `"open MeasureTheory MeasureTheory.Measure in "`
- `CategoryTheory.ShortComplex.cycles_ext_iff` → `"open CategoryTheory CategoryTheory.ShortComplex in "`
- `simple_name` → `None`

Rule: For name `A.B.C.theorem`, generate prefixes `[A, A.B, A.B.C]` and format as `"open A A.B A.B.C in "`.

`normalize_tactic(tactic: &str) -> String`

Normalizes whitespace for deduplication: trim, collapse runs of whitespace to single space.

`PROBE_TACTICS: &[&str]` — 18 built-in Lean 4 tactics tried at each step:
`simp, ring, omega, norm_num, decide, trivial, rfl, tauto, linarith, push_neg, contradiction, exfalso, constructor, left, right, ext, simp_all, intro _`

### Output Format

Standard `TrajectoryRecord`, compatible with existing `ContrastiveSampler`. Both ground-truth positives and alternative-proof positives use `label: "positive"`. Negatives use `label: "negative"`.

### Error Handling

- Ground-truth fails in Pantograph (even with namespace retry) → **enter zombie walk** if probe state available, otherwise skip rest of theorem (~98% enter zombie, ~2% complete normally)
- SGLang fails → skip step's candidates (probe tactics still run)
- Per-theorem timeout (120s) → skip theorem
- Alternative tactic produces `goals=[]` → record as Positive (alternative proof)
- Zombie walk: no probe can advance → stop walking, keep records collected so far

### Files Affected

| File | Change |
|------|--------|
| `crates/prover-core/src/main.rs` | `generate-negatives` subcommand |
| `crates/prover-core/src/pipeline.rs` | `run_generate_negatives()`, `process_theorem()`, `infer_open_prefix()`, `normalize_tactic()`, `PROBE_TACTICS`, zombie walk |
| `crates/ebm/src/training/data.rs` | `load_tactic_pairs_grouped()` |

### Unit Tests

- `test_infer_open_prefix`: multi-prefix namespace extraction ("Polynomial.X" → "open Polynomial in", "CategoryTheory.ShortComplex.X" → "open CategoryTheory CategoryTheory.ShortComplex in")
- `test_normalize_tactic`: whitespace normalization for dedup
- `test_probe_tactics_no_duplicates`: validates PROBE_TACTICS list has no duplicates and ≥10 entries

### Time Estimates (A100, with encoding sidecar)

| Theorems | Generation + Lean | Encoding (~5-10/s) | Training | **Total** |
|----------|------------------|---------------------|----------|-----------|
| 2,000 | ~2h | ~45min-1.5h | ~5m | ~3-4h |
| 5,000 | ~5h | ~2-4h | ~5m | ~7-9h |
| 8,000 | ~7h | ~2-4h | ~5m | ~9-11h |

---

## Implementation Notes (2026-02-15)

### Part 2
Implemented as planned. Commit `0ee71b4`.

### Part 3
Implemented with these enhancements beyond the original plan:

1. **Multi-tactic chain walking**: When the LLM generates a multi-step proof (e.g., `intro n\nsimp\nring`), each tactic is walked sequentially. Intermediate states are buffered and labeled based on outcome:
   - Chain reaches `ProofComplete` → all intermediates (and the first divergent step) relabeled **Positive**
   - Chain fails/stops → intermediates stay **Negative** (medium-difficulty negatives)

2. **`--min-steps` filter**: Filters theorems by minimum proof depth, targeting multi-step proofs for better depth distribution of positives.

3. **`raw_text` field on `GeneratedTactic`**: Preserves full model output before tactic extraction, enabling `extract_all_tactics()` for chain walking.

4. **SGLang resilience**: Individual HTTP 500 errors on candidate generation are caught and skipped instead of killing all candidates for a step.

5. **Progress bar interleaving**: `try_join_next()` during spawn loop drains completed tasks incrementally, keeping the progress bar live.

6. **Depth distribution tracking**: Summary prints a table of positives/negatives per proof depth.

7. **Upfront positive recording**: All ground-truth steps recorded as Positive before Pantograph loop, ensuring balanced depth distribution regardless of replay success.

8. **Multi-prefix namespace opens**: `infer_open_prefix()` opens all namespace segments (e.g., `open CategoryTheory CategoryTheory.ShortComplex in <tactic>`). Applied to ground-truth retries, LLM candidates, and probe tactics.

9. **Probe tactics**: 18 built-in Lean 4 tactics tried at each step alongside LLM candidates. Cheap (no GPU), frequently produce valid divergent states where LLM candidates mostly fail (~95% TacticResult::Failed). Deduped against LLM candidates and ground-truth.

10. **Zombie walk**: When ground-truth fails, advance via successful probe state and continue exploring. On zombie paths: skip LLM candidates (saves GPU, avoids unreliable hard negatives), label probe-reached states as Positive context (representation learning). See "Zombie Walk Design" section above.

11. **Survival rate monitoring**: Tracks % of theorems with negatives at depth > 2 (>30% safe, 10-30% marginal, <10% warning).

12. **ContrastiveSampler resilience**: All theorems with positives are eligible (not just those with both pos and neg). Cross-theorem backfill when same-theorem negatives are sparse.

### Run History (100 theorems, --min-steps 3, RTX 4090)

| Run | Positives | Negatives | Alt Proofs | Survival | Key Change |
|-----|-----------|-----------|------------|----------|------------|
| 1 (initial) | 120 | 55 | 24 | — | Baseline pipeline |
| 2 (upfront pos) | 557 | 38 | — | 1.1% | Upfront positives, depth balanced |
| 3 (namespace fix) | 557 | 56 | 24 | 4.6% | Multi-prefix opens + candidate wrapping |
| 4 (probe tactics) | 599 | 427 | 42 | 7.0% | 16 probe tactics, 7.6x negative boost |
| 5 (zombie walk) | — | — | — | — | Pending: expected significant survival boost |
