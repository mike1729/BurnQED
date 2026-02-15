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
2. Filter: skip theorems with non-contiguous depths
3. Sample N theorems
4. For each theorem (concurrent via `JoinSet`, bounded by Lean pool):
   a. Start proof in Pantograph (`copyFrom` by name)
   b. Walk proof path depth by depth:
      - Record ground-truth step as Positive
      - Generate candidates via SGLang
      - Normalize whitespace before comparing with ground-truth
      - For each non-ground-truth candidate that succeeds in Lean:
        - If `goals.is_empty()` → **Positive** record (alternative proof found)
        - If `goals.len() > 0` → **Negative** record (divergent path)
      - Apply ground-truth tactic to advance to next depth
        - On failure: retry with `open <namespace> in <tactic>`
        - If still fails: skip rest of theorem, log warning
   c. Stop early if `target_negatives` reached
5. Write Parquet via `TrajectoryWriter`
6. Print summary: theorems attempted/succeeded/skipped, positives (ground-truth + alternative), negatives

### Helper Function

`infer_open_namespace(theorem_name: &str) -> Option<String>`

Extracts the most likely `open` namespace from a fully qualified theorem name:
- `Polynomial.natDegree_cyclotomic'` → `Polynomial`
- `MeasureTheory.hahn_decomposition` → `MeasureTheory`
- `Nat.add_comm` → `Nat`

Rule: Take all dot-separated segments except the last (the theorem name), rejoin with `.`.

### Output Format

Standard `TrajectoryRecord`, compatible with existing `ContrastiveSampler`. Both ground-truth positives and alternative-proof positives use `label: "positive"`. Negatives use `label: "negative"`.

### Error Handling

- Ground-truth fails in Pantograph (even with namespace retry) → skip rest of theorem, log warning (~20-30% expected)
- SGLang fails → retry once, then skip step
- Per-theorem timeout (120s) → skip theorem
- Alternative tactic produces `goals=[]` → log info, record as positive (expected to be rare but valuable)

### Files Affected

| File | Change |
|------|--------|
| `crates/prover-core/src/main.rs` | `generate-negatives` subcommand |
| `crates/prover-core/src/pipeline.rs` | `run_generate_negatives()`, `infer_open_namespace()`, alternative-proof detection |
| `crates/ebm/src/training/data.rs` | `load_tactic_pairs_grouped()` |

### Unit Tests

- `test_infer_open_namespace`: "Polynomial.natDegree_cyclotomic'" → "Polynomial", "Nat.add_comm" → "Nat", "simple_name" → None
- `test_alternative_proof_labeling`: Mock tactic that returns empty goals → labeled positive
- `test_divergent_tactic_labeling`: Mock tactic that returns non-empty goals → labeled negative
- `test_namespace_retry`: Ground-truth tactic wrapped with `open X in <tactic>` on first failure

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

### First Run Results (100 theorems, RTX 4090)
- 42 completed, 7 skipped (namespace mismatch)
- 120 positives, 55 negatives, 24 alternative proofs, 175 total records
- Depth distribution: 0→115pos, 1→4pos+37neg, 2→1pos+14neg, 3→2neg, 4→2neg
- Ground-truth failure rate ~55% at step 0 (LeanDojo/Pantograph namespace mismatch)
- Use `--min-steps 3` for future runs to target multi-step theorems
