# Phase 5: Expert Iteration + Evaluation — Execution Plan

## Status

| Chunk | Status | Summary |
|-------|--------|---------|
| 0 | Done | CLAUDE.md updated: Phase 4 marked complete, current phase = 5 |
| 1 | Done | Results types, eval redesign (eval/summary/compare), helper extraction |
| 2 | Done | Search enhancements: --temperature, --resume-from, flush_partial, auto-save |
| 3 | Done | Python data pipeline: trace_mathlib.py, prepare_tactic_pairs.py, requirements.txt |
| 4 | Done | Python LLM training: train_llm.py (QLoRA), export_llm.py (merge+safetensors) |
| 5 | Done | Shell scripts: run_iteration.sh, lean_start.sh, setup_cloud.sh, resume_search.sh, run_all_iterations.sh |
| 6 | Done | Docs: cloud_deployment.md (providers, hardware, setup, Docker, costs, spot resilience) |
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

## Implemented Changes (Chunks 3-6)

### Chunk 3: Python Data Pipeline

**New file**: `python/requirements.txt`
- lean-dojo, torch, transformers, peft, datasets, safetensors, accelerate, bitsandbytes, pyarrow, tqdm

**New file**: `python/data/trace_mathlib.py`
- `trace_mathlib()`: LeanDojo tracing of Mathlib4 at a given commit
- `extract_theorems()`: Filter to tactic-proof theorems, output `{name, statement, file_path}`
- `extract_tactic_pairs()`: Extract `{state, tactic, theorem, depth}` per tactic step
- `extract_minif2f()`: Separate miniF2F test/valid sets
- Default mode: Download pre-traced LeanDojo benchmark data; `--trace` for local LeanDojo
- `--skip-trace` mode: Load existing cached trace
- Outputs: `theorem_index.json` (TheoremIndex format), `tactic_pairs/{train,val}.jsonl`, `minif2f_{test,valid}.json`

**New file**: `python/data/prepare_tactic_pairs.py`
- Formats raw tactic pairs as `[GOAL]{state}[PROOFSTEP]{tactic}` (matches Rust `format_prompt()`)
- Token length filtering with HuggingFace tokenizer (or char-based fallback)
- Output: JSONL with `{text, theorem, depth}` fields

### Chunk 4: Python LLM Training

**New file**: `python/training/train_llm.py`
- QLoRA fine-tuning: 4-bit quantization (nf4), bfloat16 compute, double quantization
- LoRA targets: q/k/v/o_proj + gate/up/down_proj (rank 16, alpha 32, dropout 0.05)
- `--extra-data`: Reads trajectory Parquet files, filters to `label == "positive"`, formats as training examples
- `--base`: Load previous LoRA adapter for continued training (iterations > 0)
- HuggingFace Trainer with cosine LR, warmup, gradient checkpointing
- Saves training summary JSON alongside adapter weights

**New file**: `python/training/export_llm.py`
- Merges LoRA into base model in float32 on CPU for precision
- Exports as sharded safetensors (5GB shards) consumed by `TacticGenerator::load()`
- Copies tokenizer files (tokenizer.json, tokenizer_config.json, config.json)
- `--verify`: Runs forward pass on merged model to validate

### Chunk 5: Shell Orchestration

**Replaced**: `scripts/run_iteration.sh` (was 6-line stub)
- Full expert iteration: LLM fine-tune → export → EBM train → search → noise injection → eval → compare
- LR halving: `2e-4 / 2^iter`
- Iter 0: 3 epochs on Mathlib tactic pairs + noise injection search (temp 1.2)
- Iter N>0: 1 epoch with trajectory augmentation + EBM-guided search
- Uses correct CLI flags (`--model-path` not `--llm-path` for search/eval)

**New file**: `scripts/lean_start.sh`
- Quick validation: LLM-only search → train EBM → EBM-guided search → compare summaries
- Auto-detects full theorem_index.json for 500-theorem subset, falls back to test_theorems.json

**New file**: `scripts/setup_cloud.sh`
- Cloud bootstrap: system packages → Rust → elan → submodules → Pantograph → Python venv → cargo build
- Model weight download instructions (HuggingFace CLI or git lfs)
- Smoke test: 2-theorem search to verify full pipeline

**New file**: `scripts/resume_search.sh`
- Spot instance recovery: checks partial Parquet → resumes or starts fresh
- Reports theorem progress before resuming

**New file**: `scripts/run_all_iterations.sh`
- Sequential loop over N iterations with per-iteration logging
- Prints final compare command on completion

### Chunk 6: Documentation

**New file**: `docs/cloud_deployment.md`
- Provider comparison: Lambda Labs, RunPod, Vast.ai, AWS, GCP with pricing
- Hardware requirements per task: fine-tune (A100 80GB), search (A100 40GB + 8 cores), EBM (any 8GB GPU)
- Setup checklist: automated (setup_cloud.sh) and manual paths
- Docker / snapshot workflow with Dockerfile example
- Persistent storage layout and volume sizing (~200GB for 5 iterations)
- Cost estimate: ~$55-60 for full 5-iteration run on A100 spot
- Monitoring: tmux layout, GPU/disk alerts
- Spot instance resilience: auto-save, resume protocol, SIGTERM handling

## Remaining Chunks (7)

### Chunk 7: Final Documentation (Prompt 5.10) — Deferred
- Final CLAUDE.md update marking Phase 5 complete
- `docs/results.md` with actual evaluation results (post-GPU-runs)
