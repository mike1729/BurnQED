# burn-qed

# Global Claude Code Instructions

## Gemini Tool Usage Rules

For structured reviews, always use the designated review workflows:
- For plan reviews → follow `gemini-plan-reviewer` agent instructions
- For code reviews → follow `gemini-reviewer` agent instructions
- For security reviews → follow `gemini-security-reviewer` agent instructions

The `ask-gemini` MCP tool may be used directly for quick one-off questions that don't fit any review workflow.

## Custom Agent Definitions

Review agents are defined in `.claude/agents/`. Each agent file contains step-by-step instructions
for gathering context and invoking `mcp__gemini-cli__ask-gemini` with `model: "gemini-3-pro-preview"`.

Available agents:
- **`gemini-plan-reviewer`** (`.claude/agents/gemini-plan-reviewer.md`) — Reviews plans before implementation
- **`gemini-reviewer`** (`.claude/agents/gemini-reviewer.md`) — Reviews code changes after implementation
- **`gemini-security-reviewer`** (`.claude/agents/gemini-security-reviewer.md`) — Security audit on changes

**How to invoke**: Read the agent `.md` file, follow its execution steps (gather context, compose
prompt with `@file` includes), then call `mcp__gemini-cli__ask-gemini` with the specified model
and prompt template. Present findings as described in the agent file.

## Planning Protocol

When planning any non-trivial task (new features, refactors, architecture changes, bug fixes
that touch multiple files, or anything requiring more than ~50 lines of changes):

1. **Think deeply first.** Before writing any plan, understand the full scope of the request.
   Read relevant files, check dependencies, and understand existing patterns.

2. **Write the plan** to the active plan mode file (Claude Code stores plans at `~/.claude/plans/`) with this structure:

   ### Goal
   One clear sentence describing what we're trying to achieve and why.

   ### Context
   - What currently exists (relevant files, functions, patterns)
   - What problem this solves or what value it adds

   ### Approach
   Numbered steps, each concrete and actionable. For each step:
   - What file(s) are affected
   - What specifically changes
   - Why this approach over alternatives

   ### Files Affected
   List every file to be created, modified, or deleted.

   ### Dependencies
   New libraries, services, environment variables, or config changes needed.

   ### Edge Cases & Risks
   - What could go wrong
   - Breaking changes
   - Migration needs
   - Backward compatibility concerns

   ### Testing Strategy
   - What tests to write or update
   - How to verify the changes work
   - Manual testing steps if applicable

3. **MANDATORY: Invoke the `gemini-plan-reviewer` subagent** after writing the plan.
   **You MUST NOT call ExitPlanMode or present the plan to the user until the reviewer has run.**
   This is a hard gate — no exceptions. If you are in plan mode, the sequence is always:
   write plan → run reviewer → handle verdict → then (and only then) ExitPlanMode.

4. If the reviewer returns **NEEDS_REVISION**:
   - Read every concern carefully
   - Update the plan file to address all high and medium severity concerns
   - Re-invoke the reviewer
   - Maximum 3 revision cycles — if still not approved, present concerns to the user

5. If the reviewer returns **APPROVE**:
   - **High complexity** (as rated by reviewer): present the full plan with a summary to the user and **wait for explicit confirmation** before implementing
   - **Medium complexity**: present a brief summary and **wait for confirmation**
   - **Low complexity**: briefly state what will be done and proceed unless the user objects
   - When in doubt, default to waiting for confirmation

6. **Do not skip planning** for non-trivial tasks. When in doubt, plan.
   **Do not skip the plan reviewer.** Skipping it is a protocol violation.

## Implementation Protocol

When implementing after an approved plan:

1. Follow the plan step by step in the order specified
2. Do not deviate from the plan without stating why
3. If you discover the plan needs adjustment mid-implementation, update the plan file
   and note what changed and why

## Code Review Protocol

After implementation is complete (before committing):

1. **Invoke the `gemini-reviewer` subagent**

2. If verdict is **REQUEST_CHANGES**:
   - Fix all "error" severity issues immediately
   - Fix all "warning" severity issues unless there's a clear reason not to
   - "Suggestion" severity items: apply if quick, otherwise note for later
   - Re-invoke the reviewer after fixes
   - Maximum 5 review cycles

3. If verdict is **NEEDS_DISCUSSION**:
   - Present the findings to the user with your assessment
   - Wait for the user's decision before proceeding

4. If verdict is **APPROVE**:
   - Safe to commit

## Security Review Protocol

**Automatically invoke the `gemini-security-reviewer` subagent** when changes touch ANY of:

- Authentication or authorization (login, signup, tokens, sessions, roles, permissions)
- Payment or financial logic
- User data handling (PII, passwords, emails, addresses)
- API endpoints (new or modified routes)
- Database queries or schema changes
- File uploads or user-generated content
- Environment variables, secrets, or credentials
- Infrastructure config (Docker, CI/CD, deployment)
- New dependency additions
- CORS, CSP, or security header changes

If the security reviewer returns **FAIL**:
- Do NOT commit under any circumstances
- Fix all critical and high severity vulnerabilities
- Re-invoke the security reviewer
- Only proceed after verdict changes to PASS or WARN

## General Coding Standards

- Write clean, readable code. Prioritize clarity over cleverness.
- Use meaningful names for variables, functions, and files.
- Handle errors explicitly — no silent failures or empty catch blocks.
- Add comments only when the "why" isn't obvious from the code.
- Keep functions focused — one function, one responsibility.
- Don't leave dead code, unused imports, or TODO comments without context.
- Match existing project patterns and conventions.
- When modifying existing code, maintain the established style.

## Git Commit Standards

- Write clear, descriptive commit messages
- Use conventional commit format when the project uses it
- Keep commits atomic — one logical change per commit
- Don't commit generated files, build artifacts, or secrets

## Communication Style

- Be direct. State what you're doing and why.
- When unsure, say so and explain what you'd need to clarify.
- If a task is ambiguous, ask before assuming.
- When presenting review findings, lead with the most important issues.
- Don't repeat yourself or pad responses with unnecessary explanation.


Lean 4 theorem prover combining LLM policy (tactic generation) with Energy-Based Model value function (proof state scoring) to guide best-first proof search. Trained via expert iteration.

## Architecture Overview

```
DeepSeek-Prover-V2-7B (candle, frozen)
├── Policy head: autoregressive tactic generation (LM head)
└── Mean-pool hidden states → detached Vec<f32>
                                    │
                                    ▼
                    Energy Head (burn-rs, trainable)
                    SpectralNorm MLP: 4096 → 512 → 256 → 1
                    Output: scalar energy (lower = more provable)

Lean 4 REPL Pool (tokio, Pantograph JSON protocol)
└── Verifies tactics against proof states, returns new goals
```

Single shared 7B backbone serves both policy and value function (AlphaZero-style). The energy head (~5M params) is the only component trained in Rust via burn-rs. LLM fine-tuning happens in Python with HuggingFace PEFT/LoRA.

## Settled Architecture Decisions — Do NOT Change

These were decided after two external review rounds. Do not revisit without explicit instruction.

1. **Shared 7B backbone**, not separate 1.3B encoder. DeepSeek-Prover-V2-7B serves both policy and value via `encode_only()`. Saves 3GB VRAM, one tokenizer, Lean-native representations. `EncoderBackend` enum allows switching to dedicated encoder via config.

2. **No ONNX import.** candle loads HuggingFace safetensors directly. No Python encoder export.

3. **SpectralNorm Option C** for our code (random reinit per forward, 5 power iterations). Simpler than persisting u/v. Switch to Param+detach for upstream PR only.

4. **Single optimizer for EBM.** Only energy head trains in burn-rs. 7B encoder frozen in candle. No per-parameter-group optimizer.

5. **No "skip easy" EBM bypass.** Always score with EBM when available. LLMs are confidently wrong on formal math.

6. **No dual tokenizer / bag-of-embeddings.** Smart truncation if context too long.

7. **Worker recycling for Lean.** 1000 requests OR 30 minutes TTL. Lean processes leak memory from cached environments, elaboration state, type-class resolution.

8. **Pantograph protocol:** JSON lines terminated by `\n`. Missing newline hangs process. 30s timeout on all reads. On timeout, kill and recycle worker.

9. **ProofHandle pattern for state ID routing.** Pantograph `stateId` values are process-local. `pool.start_proof()` returns a `ProofHandle` that holds the worker for the proof's lifetime — never release between tactics. No standalone `pool.run_tactic()`. Use `ProofHandleOwned` / `WorkerGuardOwned` (with `Arc<LeanPool>`) for `tokio::spawn`. Use `checkout()` / `WorkerGuard` only for advanced multi-proof-on-one-worker scenarios.

## Code Conventions

### Rust
- Edition 2021
- `anyhow::Result` for application code (prover-core, scripts)
- Concrete error enums for library crates:
  ```rust
  // crates/lean-repl/src/types.rs
  #[derive(Debug, thiserror::Error)]
  pub enum LeanError {
      #[error("Lean process exited unexpectedly")]
      ProcessDied,
      #[error("Tactic timed out after {0}s")]
      Timeout(u64),
      #[error("Protocol error: {0}")]
      Protocol(String),
      #[error("Lean error: {0}")]
      LeanMessage(String),
      #[error(transparent)]
      Io(#[from] std::io::Error),
  }
  ```
- `tracing` for all logging. Never `println!` in library code. `tracing::debug!` for per-request detail, `tracing::info!` for lifecycle events, `tracing::warn!` for recoverable issues.
- `tokio` for all async. No `async-std`.
- All public types and functions get `///` doc comments
- Tests in `#[cfg(test)] mod tests` within each file
- Integration tests in `crates/*/tests/`
- `ordered-float::OrderedFloat<f64>` for priority queues
- `serde::Deserialize` on all config types, load from TOML
- burn-rs: `#[derive(Module, Debug)]` on models, `#[derive(Config, Debug)]` on configs
- burn-rs testing: `type TestBackend = burn::backend::NdArray<f32>;`

### File Organization
- One major type per file. `worker.rs` has `LeanWorker`, `pool.rs` has `LeanPool`.
- `mod.rs` files only contain `pub mod` declarations and re-exports.
- `lib.rs` re-exports the public API for the crate.

### Naming
- Crate names: `lean-repl`, `burn-contrib`, `prover-core` (kebab-case)
- Module names: `snake_case`
- Types: `PascalCase`
- Config structs: `{Thing}Config` (e.g., `LeanPoolConfig`, `EnergyHeadConfig`)

## Key Dependencies

```toml
# PHASE 1 (lean-repl) dependencies:
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
tracing = "0.1"
thiserror = "2"

# Full workspace (pinned — do not upgrade without testing):
burn = "0.16"
candle-core = "0.8"
candle-transformers = "0.8"
arrow = "53"
parquet = "53"
tokenizers = "0.21"
```

## Pantograph Protocol Reference

Pantograph is Lean 4's programmatic REPL interface. Communication is via JSON lines over stdin/stdout of a child process.

### Setup (Bundled Submodule)

Pantograph is bundled as a git submodule at `vendor/Pantograph/` (pinned to commit `d047b1d`).

```bash
# One-time setup: init submodule + build
./scripts/setup_pantograph.sh

# Or manually:
git submodule update --init vendor/Pantograph
cd vendor/Pantograph && lake build
```

`LeanPoolConfig::with_bundled_pantograph()` auto-discovers the vendor path — no env var needed. Override with `PANTOGRAPH_PROJECT=/other/path` if desired.

### Launching

```bash
# The binary is typically at .elan/toolchains/.../bin/pantograph
# Or built from source: https://github.com/lenianiva/Pantograph
pantograph --env-path /path/to/mathlib/build/
```

### Request/Response Format

Every request is a single JSON object on one line, terminated by `\n`.
Every response is a single JSON object on one line.

**CRITICAL:** The `\n` terminator is mandatory. Without it, Pantograph blocks forever waiting for more input. This is the #1 source of hangs.

### Key Commands

```jsonc
// Start a new proof environment for a theorem (by expression)
{"cmd": "goal.start", "payload": {"expr": "∀ (n : Nat), n + 0 = n"}}
// Response: {"stateId": 0, "root": "_uniq.7"}

// Start a proof by looking up a Mathlib theorem name
{"cmd": "goal.start", "payload": {"copyFrom": "Nat.add_comm"}}
// Response: {"stateId": 0, "root": "_uniq.42"}

// Apply a tactic to a proof state
{"cmd": "goal.tactic", "stateId": 0, "goalId": 0, "tactic": "intro n"}
// Response: {"stateId": 1, "goals": ["n : Nat\n⊢ n + 0 = n"]}

// Success: empty goals list means proof complete
{"cmd": "goal.tactic", "stateId": 1, "goalId": 0, "tactic": "simp"}
// Response: {"stateId": 2, "goals": []}

// Error: tactic failed
{"cmd": "goal.tactic", "stateId": 0, "goalId": 0, "tactic": "bad_tactic"}
// Response: {"error": "unknown tactic 'bad_tactic'"}
```

### Important Protocol Details

- `stateId` is monotonically increasing and immutable — you can revisit any previous state
- `goalId` is 0-indexed within a state's goal list (usually 0 for single-goal states)
- The `goals` array in responses contains pretty-printed proof state strings
- Error responses have an `"error"` field instead of `"stateId"`/`"goals"`
- Pantograph may also output diagnostic messages on stderr — always redirect stderr to null
- Some tactics cause Lean to loop or take very long — always use a read timeout (30s)

### Proof State Format

Goals are pretty-printed as:
```
hyp1 : Type1
hyp2 : Type2
⊢ goal_type
```

The `⊢` symbol separates hypotheses from the goal. This is the string you'll tokenize and feed to the LLM/EBM.

## Phase Status

- [x] **Phase 0: Setup** — Cargo workspace, all crate stubs compile
- [x] **Phase 1: Lean REPL** — Pantograph client with worker pool, ProofHandle pattern, and recycling
- [x] **Phase 2: LLM in candle** — TacticGenerator with generate, encode_only, forked Llama
- [x] **Phase 3: Search + trajectory + CLI** — Search engine, trajectory Parquet I/O, prover-core CLI
- [x] **Phase 4: EBM** — EnergyHead, training loop, inference, CLI integration
- [x] **Phase 5: Expert iteration** — eval/compare/resume CLI, Python training pipeline, shell orchestration, cloud deployment
- [ ] Phase 6: burn-rs PRs

## Current Phase: Experiment Execution

### Phase 0 Deliverable (DONE)
All crates exist as stubs. `cargo check --workspace` passes.

### Phase 1 Deliverable (DONE)
A working async Lean 4 REPL client that can:
1. Spawn Pantograph child processes
2. Send tactic commands via JSON-line protocol
3. Parse responses into typed Rust structs
4. Manage a pool of N workers with semaphore-based concurrency
5. Guarantee state ID routing via `ProofHandle` (worker held for proof lifetime)
6. Support concurrent proofs via `ProofHandleOwned` with `Arc<LeanPool>`
7. Recycle workers after 1000 requests or 30 minutes
8. Handle timeouts (30s) and crashes gracefully
9. Start proofs by theorem name via Pantograph `copyFrom` (for Mathlib theorems)
10. Pass 10 integration tests including concurrent multi-step branching proofs

### Phase 2 Deliverable (DONE)
A working LLM inference crate that can:
1. Load DeepSeek-Prover-V2-7B (Llama architecture) from HuggingFace safetensors
2. Deserialize the model's config.json (including YaRN rope_scaling fields)
3. Run autoregressive tactic generation with temperature + top-p sampling
4. Batched candidate generation: single prefill, N-way parallel decode via KV cache expansion
5. Extract mean-pooled hidden states via `encode_only()` for the EBM
6. Manage KV cache for efficient generation, fresh cache for deterministic encoding
7. Handle multi-shard safetensors loading
8. Extract first tactic from code-fenced model output (strips fences, declarations, comments)
9. Pass 34 unit tests (types, tokenizer, llama config, sampling, prompt formatting, cache ops, extraction)
10. 10 integration tests ready (require MODEL_PATH env var)

### Phase 2 Architecture Notes

- **Forked llama.rs**: Copied from candle-transformers 0.8.4 with modifications:
  - Private `Llama` fields (accessed only through methods)
  - Added `forward_hidden_states()` returning `(batch, seq_len, hidden_size)` before lm_head
  - Added `Cache::snapshot()`, `Cache::restore()`, `Cache::expand_batch(n)` for batched generation
  - `DeepSeekConfig` deserializes model's config.json with YaRN fields, converts to runtime `Config` with `rope_scaling: None`
  - Inlined `with_tracing` wrappers (Linear, RmsNorm) to avoid depending on candle-transformers internals
  - Removed flash-attn support (Windows/CPU only)
- **YaRN RoPE ignored**: Standard RoPE used. YaRN is backwards-compatible within original 4096 context, and our sequences are ≤2048 tokens.
- **Prompt format**: Chat format via `encode_chat()` wrapping `format_tactic_message()` — tactic-state comment block in a lean4 code fence, matching DeepSeek-Prover-V2 training data.
- **Generation stopping**: EOS token or max_tactic_tokens only. No newline-based stopping — model outputs code-fenced multi-line completions; `extract_first_tactic()` handles parsing.
- **f32 on CPU, bf16 on CUDA**: CPU doesn't support bf16 well in candle.

### Phase 3 Deliverable (DONE)

**Trajectory crate (Part 1):**
- `TrajectoryRecord`, `SearchResult`, `TheoremTask`, `TheoremIndex`, `TrajectoryLabel`
- `TrajectoryWriter`: buffer records, write Parquet, label from search results
- `TrajectoryReader`: read Parquet back, compute `TrajectorySummary`
- Arrow schema (12 columns), roundtrip Parquet I/O

**Search crate (Parts 2 & 3):**
A trait-based best-first search engine that can:
1. Search for proofs using priority queue (combined LLM + EBM score)
2. Generate tactic candidates via `PolicyProvider` trait (sync, matches candle)
3. Score states via `ValueScorer` trait (sync, optional — EBM comes in Phase 4)
4. Verify tactics via `ProofEnvironment` / `TacticRunner` traits (async, matches Lean)
5. Synthesize root node goals from theorem statement (uses `⊢` as-is if present, prepends `"⊢ "` otherwise)
6. Respect node budget, depth limit, and wall-clock timeout
7. Build `SearchResult` with trajectory records for EBM training
8. Bridge to real types via adapters (`Arc<LeanPool>`, `ProofHandleOwned`, `MutexPolicyProvider`)
9. Instrument `SearchStats` (timing, pruning, frontier size) per search
10. Pass 35 unit tests using mocks (no Lean, no LLM)

### Phase 3 Architecture Notes

- **Trait-based abstraction**: `ProofEnvironment`, `TacticRunner` (async) and `PolicyProvider`, `ValueScorer` (sync) allow testing with mocks.
- **Root node synthesis**: `goal.start` returns empty goals. Search uses the statement as root: if it contains `⊢` (Mathlib proof states), uses as-is; otherwise prepends `"⊢ "` (simple expressions).
- **Arena indexing**: Nodes stored in `Vec<SearchNode>`, parent references by index. `ScoredNode` wraps `OrderedFloat<f64>` for `BinaryHeap`.
- **Adapters**: `Arc<LeanPool>` implements `ProofEnvironment` (tries `copyFrom` by name, falls back to `expr`), `ProofHandleOwned` implements `TacticRunner`, `MutexPolicyProvider` wraps `TacticGenerator` with `Mutex` for `Send + Sync`.
- **`ProofEnvironment::start_proof(name, statement)`**: Accepts both theorem name and statement. The adapter tries `copyFrom(name)` first (works for Mathlib theorems loaded in the environment), falling back to `expr(statement)` on `LeanMessage` error. This enables both `theorem_index.json` (proof states) and `test_theorems.json` (expressions) as input.

### Phase 3 Part 4+5: prover-core CLI + test data (DONE)

**prover-core CLI:**
- `config.rs`: `SearchToml`, `LeanPoolOverrides`, `load_search_toml()`, `build_lean_pool_config()`
- `pipeline.rs`: `run_search()` (async, CTRL-C handling, EBM loading, `--dry-run`), `run_eval()` (sync), `run_train_ebm()` (sync, trains EBM from Parquet)
- `main.rs`: clap CLI with `search` (incl. `--dry-run`, `--ebm-path`), `eval`, `train-ebm` subcommands
- Priority chain for config: `with_bundled_pantograph()` defaults < TOML values < `--num-workers` CLI flag
- EBM integration: `--ebm-path` loads `EnergyHeadConfig` + checkpoint, shares `TacticGenerator` via `Arc<Mutex>` for both policy and EBM encoding

**Test data:**
- `data/test_theorems.json`: 16 Init-only theorems (True, False→False, nat_refl, and_comm, eq_trans, etc.)

### Phase 4 Deliverable (DONE)

**EBM Model (`crates/ebm/`):**
- `SpectralNormLinear` (Option C: random reinit, 5 power iterations)
- `EnergyHead`: 4096→512→256→1 MLP with SiLU, dropout, learnable log_temperature
- `bridge.rs`: Vec<f32> ↔ burn Tensor conversions
- `EncoderBackend` enum (Shared/Dedicated) for config-driven encoder selection

**Training (`crates/ebm/src/training/`):**
- `ContrastiveSampler`: hard/medium/easy negative mining from Parquet trajectories
- `info_nce_loss` + `depth_regression_loss`
- `EBMMetrics` + `MetricsHistory` + `health_check()` (gradient norm, loss, temperature monitoring)
- Training loop: AdamW optimizer, warmup+cosine LR schedule, periodic checkpointing
- `EmbeddingCache`: precompute embeddings once, save/load as Parquet for reuse

**Inference (`crates/ebm/src/inference.rs`):**
- `EBMScorer<B>`: loads checkpoint + encode_fn, batch scoring
- `EBMValueFn`: backend-erased wrapper implementing `search::ValueScorer` trait

**CLI Integration (`crates/prover-core/`):**
- `train-ebm` subcommand with `--embeddings-cache`, `--save-embeddings`, `--resume-from`
- `search --ebm-path` loads EBM scorer, shares `TacticGenerator` via `Arc<Mutex>` for both policy and EBM encoding

**Test Coverage:**
- 41 unit tests + 30 integration tests (ebm: 32+9, prover-core: 3+8+4 TinyLlama)

### Cross-Crate Integration

- **Running search**: `cargo run -p prover-core -- search --model-path ... --theorems ... --output ...`
- **Search with EBM**: `cargo run -p prover-core -- search --ebm-path checkpoints/ebm ...` — loads `energy_head_config.json` + `final.mpk` from the checkpoint dir, shares the `TacticGenerator` between policy and EBM encode closure via `Arc<Mutex<TacticGenerator>>`
- **Dry-run validation**: `cargo run -p prover-core -- search --dry-run --model-path ... --theorems ...` — loads model, pool, theorems, verifies setup, exits without searching
- **CTRL-C handling**: Graceful interruption during search loop. Partial results are written to Parquet with an enhanced summary noting the interruption.
- **Reading trajectories**: `TrajectoryReader::read_all(path)` returns `Vec<TrajectoryRecord>`
- **Label logic**: `TrajectoryWriter::from_search_result()` — proved theorems get Positive labels on the proof path (root→QED), Negative on dead ends. Unproved theorems are all Negative.
- **SearchStats**: Each `SearchResult` includes `stats: SearchStats` with per-search timing (Lean time, generation time), node counts (expanded, pruned, terminal), and peak frontier size.
- **Parquet → EBM training**: `cargo run -p prover-core -- train-ebm --trajectories ... --llm-path ...` reads trajectory Parquet files, creates `ContrastiveSampler`, trains EBM via `ebm::train()`, saves checkpoint + `energy_head_config.json`.
- **MutexPolicyProvider sharing**: `new_shared(Arc<Mutex<TacticGenerator>>)` and `shared_generator()` methods allow the same generator to be used for both policy and EBM encoding.

## Testing Policy

**Always run integration tests after implementing or editing them.** Integration tests (`#[ignore]`) exercise real Lean/Pantograph and catch issues that unit tests with mocks cannot. Before running, confirm with the user if runtime is expected to be long (e.g., >30 seconds). Typical runtimes:

- `cargo test -p lean-repl -- --ignored --test-threads=1` — ~60-90s (10 tests, spawns Pantograph)
- `cargo test -p search -- --ignored --test-threads=1` — ~60s (11 tests, spawns Pantograph)
- `cargo test -p prover-core -- --ignored --test-threads=1` — ~15s (1 test, spawns Pantograph)
- `cargo test -p prover-core --test integration_llm -- --ignored --test-threads=1` — ~10-35min per test (4 tests, require TinyLlama model weights)
- `cargo test -p policy -- --ignored --test-threads=1` — requires MODEL_PATH env var, ~30-60s

All integration tests that use Pantograph **must** run with `--test-threads=1` to avoid resource contention.

## Documentation Maintenance

When making architectural changes, adding new public types, or completing phases, **always update these docs to match**:

- `CLAUDE.md` (this file) — project layout, phase status, settled decisions, API summaries
- `docs/burn-qed_plan.md` — full plan with code samples, architecture diagrams, risk table, cost analysis
- `docs/phase1_instructions.md` (and future `phase*_instructions.md`) — step-by-step prompts and verification checklists

If a code change would make any code sample, type signature, or architectural description in these docs incorrect, fix the docs in the same change.

## Experiment Execution

### Quick Start

```bash
# 1. Data preparation (CPU, ~5 min download or hours with --trace)
./scripts/prepare_data.sh              # Downloads pre-traced data (default)
./scripts/prepare_data.sh --trace      # Local LeanDojo trace (optional)

# 2. Cloud bootstrap (on GPU instance)
bash scripts/setup_cloud.sh

# 3. Run full experiment
NUM_WORKERS=64 ./scripts/run_all_iterations.sh
```

### Scripts

| Script | Purpose | GPU? |
|--------|---------|------|
| `scripts/setup_cloud.sh` | Install Rust, elan, Python deps, build Pantograph + prover-core | No |
| `scripts/prepare_data.sh` | Trace Mathlib + format tactic pairs + validate outputs | No |
| `scripts/run_baseline.sh` | Phase B: raw model baseline on test_theorems + miniF2F + theorem_index + baseline EBM | Yes |
| `scripts/run_iteration.sh N` | One expert iteration: fine-tune → export → EBM → search → eval + ablation | Yes |
| `scripts/run_all_iterations.sh` | Full experiment: baseline + iters 0-4 + final analysis | Yes |
| `scripts/resume_search.sh N` | Resume interrupted search from partial Parquet file | Yes |
| `scripts/lean_start.sh` | Smoke test: 16 theorems, 100-node budget, 32 candidates, EBM train+search | Yes |

### Throughput Tuning

Cross-prompt batching (different proof states in one GPU batch) is not feasible without padding, attention masking, and ragged KV cache management.

**Key finding:** Batched decode scales ~linearly in N on this model (not constant as hoped). 32 candidates takes ~19s vs ~2.5s for 4 candidates (~7.5× slower). After dedup at T=0.6, 32 candidates yield only 2-4 unique tactics — same as 4-8 candidates. High candidate counts waste GPU time.

**Recommended defaults** (set in `configs/search.toml` and scripts):

| Parameter | Value | Reason |
|-----------|-------|--------|
| `num_candidates` | 4 | 2-3 unique after dedup at T=0.6; ~2.5s GPU time. |
| `num_workers` | 6 | Enough to overlap Lean verification with GPU generation |
| `concurrency` | 6 | Match workers — each needs one active search |
| `max_nodes` | 100 | At 4% prove rate, proofs found within 2 nodes. 100 ≈ 3 expansions with backtrack. |

The generation service (`policy::spawn_generation_service`) processes requests FIFO via an mpsc channel, eliminating mutex contention entirely. Workers queue requests while the GPU is busy; the 64-slot channel buffer absorbs bursts.

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NUM_WORKERS` | 6 | Number of Lean worker processes |
| `CONCURRENCY` | 6 | Number of theorems searched in parallel |
| `MAX_ITER` | 4 | Maximum iteration number (0-indexed) |
| `LLM_BASE` | `deepseek-ai/DeepSeek-Prover-V2-7B` | Base model for fine-tuning |
| `SKIP_BASELINE` | 0 | Set to 1 to skip Phase B baseline |
| `MODEL_PATH` | (none) | Local model dir for tokenizer in data prep |
| `MATHLIB_COMMIT` | `v4.26.0` | Mathlib4 tag to trace (matches Pantograph lean-toolchain) |

### Experiment Outputs

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

models/llm/iter_0 ... iter_4        # Merged safetensors for candle
logs/iter_0.log ... iter_4.log      # Per-iteration logs
```

### Go/No-Go Checkpoints

1. **After B2** (raw baseline): If <5% on miniF2F → investigate model loading or search config
2. **After C3** (iter 0): If no improvement over baseline → check training data + loss curves
3. **After D4 vs D5** (EBM ablation): Key result — if EBM shows no improvement → investigate embeddings
4. **After each iteration**: If solve rate plateaus or decreases → stop early

## Reference

Full plan with architecture diagrams, all code samples, loss functions, training loop, cost analysis, and risk mitigation: `docs/burn-qed_plan.md`
