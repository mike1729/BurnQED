# Phase 5: Expert Iteration + Evaluation — Execution Plan

## Status

| Chunk | Status | Summary |
|-------|--------|---------|
| 0 | Done | CLAUDE.md updated: Phase 4 marked complete, current phase = 5 |
| 1 | Done | Results types, eval redesign (eval/summary/compare), helper extraction |
| 2 | Done | Search enhancements: --temperature, --resume-from, flush_partial, auto-save |
| 3 | Pending | Python: trace_mathlib.py, prepare_tactic_pairs.py, requirements.txt |
| 4 | Pending | Python: train_llm.py (LoRA), export_llm.py (merge+safetensors) |
| 5 | Pending | Shell: run_iteration.sh, lean_start.sh, setup_cloud.sh |
| 6 | Pending | Docs: cloud_deployment.md |
| 7 | Pending | Docs: Final CLAUDE.md update, results.md (post-GPU-runs) |

## Implemented Changes (Chunks 0-2)

### Chunk 0: CLAUDE.md
- Phase 4 marked `[x]` with deliverable summary
- Current phase updated to 5
- Phase 4 deliverable section added (model, training, inference, CLI, test coverage)

### Chunk 1: Results Types + Eval Redesign

**New file**: `crates/prover-core/src/results.rs`
- `IterationResult`, `BudgetResult`, `TheoremResult` — serde-capable types for evaluation tracking
- `median()` helper for time statistics
- 3 unit tests (serde roundtrip, rate computation, median)

**Renamed**: `eval` → `summary` (the Parquet statistics command)
- `EvalArgs` → `SummaryArgs`, `run_eval` → `run_summary`

**New `eval` subcommand**: Multi-budget evaluation
- Loads model + optional EBM via shared `load_policy_and_ebm()` helper
- Iterates budgets × theorems × pass_n attempts
- Prints formatted ASCII table
- Writes JSON `IterationResult` to `--output` (default: `eval_results/eval.json`)

**New `compare` subcommand**: Cross-iteration comparison
- Reads multiple `IterationResult` JSON files
- Finds common budgets, prints table with delta row for 2-result comparison

**Extracted helper**: `load_policy_and_ebm()` — shared setup code for `run_search` and `run_eval`
- Loads config TOML, builds Lean pool, loads LLM with optional temperature override, optional EBM scorer

**Dependencies**: Added `chrono` (workspace + prover-core)

### Chunk 2: Search Enhancements

**`--temperature`**: Overrides `PolicyConfig.temperature` at CLI level

**`--resume-from`**: Skip already-searched theorems
- `TrajectoryReader::read_theorem_names()` reads unique theorem names from Parquet
- Filters theorem list before search loop
- After search, merges old + new records into final output

**`TrajectoryWriter::flush_partial()`**: Periodic checkpointing
- Redesigned writer with `pending` + `flushed` buffers
- `flush_partial()` moves pending→flushed, rewrites entire file
- Auto-save every 50 theorems during search

**3 new trajectory tests**: flush_partial checkpoint, flush empty noop, read_theorem_names

**4 new prover-core integration tests**: eval mock budgets, compare two results, resume from partial

### Test Summary After Implementation

| Crate | Unit | Integration | Total |
|-------|------|-------------|-------|
| trajectory | 21 | 7 | 28 |
| prover-core | 6 | 13 (+1 ignored) | 19+1 |
| All workspace | 146 | 53 (+35 ignored) | 199+35 |

## Remaining Chunks (3-7)

### Chunk 3: Python Data Pipeline (Prompt 5.1)
- `python/data/trace_mathlib.py` — Extract theorem statements from Mathlib
- `python/data/prepare_tactic_pairs.py` — Create (state, tactic) training pairs
- `python/requirements.txt` — Python dependencies

### Chunk 4: LLM Training (Prompts 5.2 + 5.3)
- `python/training/train_llm.py` — LoRA fine-tuning with HuggingFace PEFT
- `python/training/export_llm.py` — Merge LoRA + export safetensors for candle

### Chunk 5: Shell Orchestration (Prompts 5.5 + 5.7)
- `scripts/run_iteration.sh` — Full expert iteration loop
- `scripts/lean_start.sh` — Cloud Lean environment bootstrap
- `scripts/setup_cloud.sh` — GPU cloud provisioning

### Chunk 6: Documentation (Prompt 5.9)
- `docs/cloud_deployment.md` — Cloud deployment guide

### Chunk 7: Final Documentation (Prompt 5.10)
- Final CLAUDE.md update marking Phase 5 complete
- `docs/results.md` with actual evaluation results (post-GPU-runs)
