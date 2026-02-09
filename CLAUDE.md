# burn-qed

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

## Project Layout

```
burn-qed/
├── CLAUDE.md                           # THIS FILE
├── Cargo.toml                          # Workspace root
├── vendor/
│   └── Pantograph/                     # Git submodule (pinned commit d047b1d)
├── docs/
│   ├── burn-qed_plan.md               # Full architecture plan with all code samples
│   ├── phase1_instructions.md          # Phase 1 step-by-step prompts
│   ├── phase2_instructions.md          # Phase 2 step-by-step prompts
│   ├── phase3_instructions.md          # Phase 3 step-by-step prompts
│   ├── phase4_instructions.md          # Phase 4 step-by-step prompts
│   ├── phase5_instructions.md          # Phase 5 step-by-step prompts
│   └── week2_summary.md               # Week 2 progress summary
├── crates/
│   ├── lean-repl/                      # Phase 1: Lean 4 REPL async client
│   │   ├── Cargo.toml
│   │   ├── build.rs                    # Emits LEAN_REPL_MANIFEST_DIR for path discovery
│   │   └── src/
│   │       ├── lib.rs                  # Public API: LeanPool, ProofHandle, ProofHandleOwned
│   │       ├── worker.rs               # LeanWorker: spawn, communicate, recycle
│   │       ├── pool.rs                 # LeanPool, ProofHandle, WorkerGuard (+Owned variants)
│   │       ├── session.rs              # ProofSession: stateful proof tracking (holds ProofHandle)
│   │       ├── protocol.rs             # Pantograph JSON request/response serde types
│   │       └── types.rs                # ProofState, TacticResult, LeanError, Goal, discover_pantograph()
│   │
│   ├── policy/                         # Phase 2: LLM tactic generator (candle)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs                  # Public API: TacticGenerator, LeanTokenizer, PolicyConfig
│   │   │   ├── llama.rs                # Forked Llama with forward_hidden_states(), DeepSeekConfig
│   │   │   ├── model.rs                # TacticGenerator: load, generate, encode_only
│   │   │   ├── tokenizer.rs            # LeanTokenizer: HuggingFace tokenizer wrapper
│   │   │   └── types.rs                # PolicyConfig, DeviceConfig, GeneratedTactic, Embedding
│   │   └── tests/
│   │       └── integration.rs          # 10 #[ignore] tests (need MODEL_PATH)
│   │
│   ├── ebm/                            # Phase 4: Energy-Based Model (burn-rs)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model/
│   │       │   ├── mod.rs
│   │       │   ├── spectral_norm.rs    # SpectralNormLinear (Option C: random reinit)
│   │       │   ├── energy_head.rs      # 4096 → 512 → 256 → 1
│   │       │   └── encoder.rs          # EncoderBackend enum
│   │       ├── training/
│   │       │   ├── mod.rs
│   │       │   ├── loss.rs             # InfoNCE + depth regression
│   │       │   ├── data.rs             # Parquet dataset + contrastive batcher
│   │       │   ├── trainer.rs          # Training loop (single optimizer on head only)
│   │       │   └── metrics.rs          # EBMMetrics + health_check()
│   │       └── inference.rs            # Batch scoring for search
│   │
│   ├── search/                         # Phase 3: Best-first search
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # Public API: SearchEngine, SearchStats, traits, nodes
│   │       ├── config.rs               # SearchConfig (from TOML, with defaults)
│   │       ├── node.rs                 # SearchNode, ScoredNode, path extraction
│   │       ├── engine.rs               # Traits + SearchEngine::search_one + SearchStats instrumentation
│   │       ├── mocks.rs               # MockPolicy, MockEnvironment for testing
│   │       └── adapters.rs            # Real impls: LeanPool, ProofHandleOwned, TacticGenerator
│   │
│   ├── trajectory/                     # Phase 3: Parquet I/O
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── writer.rs               # Arrow schema, write trajectory records
│   │       ├── reader.rs               # Read + deserialize
│   │       └── types.rs                # TrajectoryRecord
│   │
│   ├── burn-contrib/                   # Phase 6: Reusable burn-rs modules → upstream PRs
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── spectral_norm.rs
│   │       ├── info_nce_loss.rs
│   │       ├── warmup_scheduler.rs
│   │       ├── parquet_dataset.rs
│   │       └── pooling.rs
│   │
│   └── prover-core/                    # CLI binary
│       ├── Cargo.toml
│       ├── src/
│       │   ├── main.rs                 # clap: search, eval, train-ebm subcommands
│       │   ├── config.rs               # SearchToml, LeanPoolOverrides, TOML loading
│       │   └── pipeline.rs             # run_search() (async), run_eval() (sync), run_train_ebm()
│       ├── examples/
│       │   └── gen_synthetic_parquet.rs # Generate synthetic trajectory Parquet for testing
│       └── tests/
│           └── integration.rs          # 7 mock tests + 1 #[ignore] Lean test
│
├── data/
│   └── test_theorems.json              # 12 Init-only theorems for testing
│
├── python/
│   ├── training/
│   │   ├── train_llm.py                # LoRA fine-tuning
│   │   └── export_llm.py              # safetensors for candle
│   └── data/
│       ├── trace_mathlib.py
│       └── prepare_tactic_pairs.py
│
├── configs/
│   ├── models.toml
│   └── search.toml
│
└── scripts/
    ├── setup_pantograph.sh             # One-time: init submodule + lake build
    ├── setup_cloud.sh                  # Cloud GPU bootstrap (Rust, elan, Python, build)
    ├── prepare_data.sh                 # Data pipeline: trace Mathlib, format tactic pairs, validate
    ├── run_baseline.sh                 # Phase B: raw model baseline evaluation
    ├── run_iteration.sh                # One expert iteration (fine-tune, EBM, search, eval)
    ├── run_all_iterations.sh           # Full experiment: baseline + N iterations + analysis
    ├── resume_search.sh                # Resume interrupted search from partial Parquet
    └── lean_start.sh                   # Quick end-to-end validation pipeline
```

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
# The binary is typically at ~/.elan/toolchains/.../bin/pantograph
# Or built from source: https://github.com/lenianiva/Pantograph
pantograph --env-path /path/to/mathlib/build/
```

### Request/Response Format

Every request is a single JSON object on one line, terminated by `\n`.
Every response is a single JSON object on one line.

**CRITICAL:** The `\n` terminator is mandatory. Without it, Pantograph blocks forever waiting for more input. This is the #1 source of hangs.

### Key Commands

```jsonc
// Start a new proof environment for a theorem
{"cmd": "goal.start", "expr": "∀ (n : Nat), n + 0 = n"}
// Response: {"stateId": 0, "goals": ["⊢ ∀ (n : Nat), n + 0 = n"]}

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
9. Pass 10 integration tests including concurrent multi-step branching proofs

### Phase 1 API Summary

```rust
// Basic proof (worker held for lifetime of `proof`):
let pool = LeanPool::new(config).await?;
let mut proof = pool.start_proof("∀ (n : Nat), n = n").await?;
let sid = proof.state_id();
let r = proof.run_tactic(sid, None, "intro n").await?;

// Concurrent proofs via tokio::spawn (owned, 'static):
let pool = Arc::new(pool);
tokio::spawn({
    let pool = pool.clone();
    async move {
        let mut proof = pool.start_proof_owned("∀ n, n = n").await.unwrap();
        // ...
    }
});

// Advanced: checkout raw worker for multi-proof scenarios:
let mut guard = pool.checkout().await?;
let w = guard.worker();
let s1 = w.start_proof("True").await?;
let s2 = w.start_proof("False → False").await?;
// both proofs share the same worker
```

### Phase 1 Test Coverage

```
cargo test -p lean-repl           # 20 unit + 2 doc tests
cargo test -p lean-repl -- --ignored --test-threads=1  # 10 integration tests

# Integration tests (require Pantograph built):
# - Single-worker simple proof (intro + rfl)
# - Tactic error handling
# - Multi-goal proofs (And.intro)
# - ProofSession (stateful tracking)
# - Sequential 100 proofs (4 workers)
# - Concurrent 20 proofs (4 workers, owned handles)
# - Concurrent multi-step isolation (10 proofs, branching, state immutability)
# - Worker recycling after N requests
# - Timeout recovery
# - Proof search simulation (branching, goalId, 20 theorems, error recovery)
```

### Phase 2 Deliverable (DONE)
A working LLM inference crate that can:
1. Load DeepSeek-Prover-V2-7B (Llama architecture) from HuggingFace safetensors
2. Deserialize the model's config.json (including YaRN rope_scaling fields)
3. Run autoregressive tactic generation with temperature + top-p sampling
4. Extract mean-pooled hidden states via `encode_only()` for the EBM
5. Manage KV cache for efficient generation, fresh cache for deterministic encoding
6. Handle multi-shard safetensors loading
7. Pass 21 unit tests (types, tokenizer, llama config, sampling, prompt formatting)
8. 10 integration tests ready (require MODEL_PATH env var)

### Phase 2 API Summary

```rust
// Load model from HuggingFace directory:
let config = PolicyConfig::new(PathBuf::from("./models/deepseek-prover-v2-7b"));
let mut gen = TacticGenerator::load(&config)?;

// Generate candidate tactics for a proof state:
let candidates = gen.generate_candidates("n : Nat\n⊢ n + 0 = n", 32)?;
for c in &candidates {
    println!("{}: log_prob={:.4}", c.text, c.log_prob);
}

// Extract mean-pooled embeddings for EBM scoring:
let embedding = gen.encode_only("n : Nat\n⊢ n + 0 = n")?;
assert_eq!(embedding.dim, 4096);

// Batch encoding:
let embeddings = gen.encode_batch(&["⊢ True", "⊢ False → False"])?;
```

### Phase 2 Architecture Notes

- **Forked llama.rs**: Copied from candle-transformers 0.8.4 with modifications:
  - Private `Llama` fields (accessed only through methods)
  - Added `forward_hidden_states()` returning `(batch, seq_len, hidden_size)` before lm_head
  - `DeepSeekConfig` deserializes model's config.json with YaRN fields, converts to runtime `Config` with `rope_scaling: None`
  - Inlined `with_tracing` wrappers (Linear, RmsNorm) to avoid depending on candle-transformers internals
  - Removed flash-attn support (Windows/CPU only)
- **YaRN RoPE ignored**: Standard RoPE used. YaRN is backwards-compatible within original 4096 context, and our sequences are ≤2048 tokens.
- **Prompt format**: `[GOAL]{proof_state}[PROOFSTEP]` — simple structured format for DeepSeek-Prover-V2.
- **f32 on CPU, bf16 on CUDA**: CPU doesn't support bf16 well in candle.

### Phase 2 Test Coverage

```
cargo test -p policy                    # 21 unit tests
cargo test -p policy -- --ignored --nocapture --test-threads=1  # 10 integration tests

# Unit tests (no model needed):
# - types: DeviceConfig, PolicyConfig deserialization (8 tests)
# - tokenizer: truncate operations (4 tests)
# - llama: DeepSeekConfig deserialization, EosToks (4 tests)
# - model: prompt formatting, top-p sampling (5 tests)

# Integration tests (default: TinyLlama-1.1B at models/tinyllama-1.1b):
# - Model loading + hidden_size verification
# - Tokenizer roundtrip + special tokens
# - Forward logits shape verification
# - Single tactic generation
# - Candidate generation (sorted by log_prob)
# - Embedding shape (dim == hidden_size, dynamic)
# - Embedding distinctness (different states → different embeddings)
# - Embedding determinism (same state → same embedding)
# - Batch encoding
#
# TinyLlama-1.1B is the default test model — same Llama architecture as DeepSeek-7B
# but loads in seconds on CPU. All assertions use gen.hidden_size() dynamically.
# Override: MODEL_PATH=models/deepseek-prover-v2-7b cargo test -p policy -- --ignored
```

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
5. Synthesize root node goals from theorem statement (`"⊢ {statement}"`)
6. Respect node budget, depth limit, and wall-clock timeout
7. Build `SearchResult` with trajectory records for EBM training
8. Bridge to real types via adapters (`Arc<LeanPool>`, `ProofHandleOwned`, `MutexPolicyProvider`)
9. Instrument `SearchStats` (timing, pruning, frontier size) per search
10. Pass 31 unit tests using mocks (no Lean, no LLM)

### Phase 3 API Summary

```rust
// Configure search:
let config = SearchConfig::default(); // max_nodes=600, max_depth=50, alpha=0.5, beta=0.5

// Search with mocks (testing):
let engine = SearchEngine::new(config);
let result = engine.search_one(&env, &policy, scorer.as_deref(), "thm_name", "True").await?;
assert!(result.proved);
assert_eq!(result.proof_tactics, vec!["trivial"]);

// Search with real Lean + LLM:
let pool = Arc::new(LeanPool::new(lean_config).await?);
let policy = MutexPolicyProvider::new(TacticGenerator::load(&policy_config)?);
let result = engine.search_one(&pool, &policy, None, "nat_refl", "∀ (n : Nat), n = n").await?;

// Write trajectory to Parquet:
let labeled = TrajectoryWriter::from_search_result(&result);
let mut writer = TrajectoryWriter::new("output.parquet".into());
writer.record_all(labeled);
writer.finish()?;
```

### Phase 3 Architecture Notes

- **Trait-based abstraction**: `ProofEnvironment`, `TacticRunner` (async) and `PolicyProvider`, `ValueScorer` (sync) allow testing with mocks.
- **Root node synthesis**: `goal.start` returns empty goals, so search builds `"⊢ {statement}"` and uses `Goal::parse(0, &root_pp)`.
- **Arena indexing**: Nodes stored in `Vec<SearchNode>`, parent references by index. `ScoredNode` wraps `OrderedFloat<f64>` for `BinaryHeap`.
- **Adapters**: `Arc<LeanPool>` implements `ProofEnvironment`, `ProofHandleOwned` implements `TacticRunner`, `MutexPolicyProvider` wraps `TacticGenerator` with `Mutex` for `Send + Sync`.
- **No standalone `pool.run_tactic()`**: Search uses `ProofEnvironment::start_proof()` returning a `TacticRunner` that holds the worker for the proof's lifetime (ProofHandle pattern).

### Phase 3 Part 4+5: prover-core CLI + test data (DONE)

**prover-core CLI:**
- `config.rs`: `SearchToml`, `LeanPoolOverrides`, `load_search_toml()`, `build_lean_pool_config()`
- `pipeline.rs`: `run_search()` (async, CTRL-C handling, EBM loading, `--dry-run`), `run_eval()` (sync), `run_train_ebm()` (sync, trains EBM from Parquet)
- `main.rs`: clap CLI with `search` (incl. `--dry-run`, `--ebm-path`), `eval`, `train-ebm` subcommands
- Priority chain for config: `with_bundled_pantograph()` defaults < TOML values < `--num-workers` CLI flag
- EBM integration: `--ebm-path` loads `EnergyHeadConfig` + checkpoint, shares `TacticGenerator` via `Arc<Mutex>` for both policy and EBM encoding

**Test data:**
- `data/test_theorems.json`: 12 Init-only theorems (True, False→False, nat_refl, and_comm, etc.)

### prover-core API Summary

```bash
# Run proof search:
cargo run -p prover-core -- search \
  --model-path models/deepseek-prover-v2-7b \
  --theorems data/test_theorems.json \
  --output output/trajectory.parquet \
  --num-workers 4

# Run proof search with EBM value guidance:
cargo run -p prover-core -- search \
  --model-path models/deepseek-prover-v2-7b \
  --theorems data/test_theorems.json \
  --output output/trajectory.parquet \
  --ebm-path checkpoints/ebm

# Verify environment setup without searching:
cargo run -p prover-core -- search --dry-run \
  --model-path models/deepseek-prover-v2-7b \
  --theorems data/test_theorems.json \
  --output output/trajectory.parquet

# Print trajectory statistics:
cargo run -p prover-core -- eval --input output/trajectory.parquet

# Train EBM from trajectory data:
cargo run -p prover-core -- train-ebm \
  --trajectories output/trajectory.parquet \
  --llm-path models/deepseek-prover-v2-7b \
  --output-dir checkpoints/ebm \
  --steps 50000

# Resume EBM training from checkpoint:
cargo run -p prover-core -- train-ebm \
  --trajectories output/trajectory.parquet \
  --llm-path models/deepseek-prover-v2-7b \
  --output-dir checkpoints/ebm_v2 \
  --resume-from checkpoints/ebm
```

### Phase 3 Test Coverage

```
cargo test -p trajectory         # 18 unit tests + 6 integration tests
cargo test -p search             # 31 unit tests (config: 4, node: 8, engine: 10, mocks: 8, + 1 doc)
cargo test -p search -- --ignored --test-threads=1  # 11 integration tests (need Pantograph, ~60s)
cargo test -p prover-core        # 3 unit tests (config) + 13 integration tests (mocks + JSON + TOML + EBM)
cargo test -p prover-core -- --ignored --test-threads=1  # 1 Lean integration test (~15s)
cargo test -p prover-core --test integration_llm -- --ignored --test-threads=1  # 4 TinyLlama tests (~10-35min each on CPU)

# Trajectory unit tests:
# - types: label display/serde, TheoremTask/Index deserialize, record defaults
# - writer: schema, empty file, write+verify, from_search_result (proved/unproved/
#           single-node/no-terminal/multiple-terminals)
# - reader: roundtrip, nullable parent, summary, read_for_theorem

# Trajectory integration tests (no Lean, no model):
# - Label roundtrip through Parquet (proved tree with dead ends)
# - TheoremIndex from JSON file / invalid JSON / missing file
# - Read multiple Parquet files
# - Multi-theorem pipeline (proved + unproved + trivial, summary verification)

# Search unit tests (all use mocks, no Lean/LLM):
# - config: defaults, partial/full TOML, alpha+beta sum
# - node: combined_score, ScoredNode ordering, goals_as_text, path extraction, branching
# - engine: 1-step proof, 2-step proof, node budget, depth limit, tactic failure,
#           empty frontier, trajectory records, proof_tactics, timeout=0, empty candidates
# - mocks: make_tactic, MockPolicy exact/default/empty/contains/exact-before-contains,
#           MockEnvironment canned/unknown

# Search integration tests (require Pantograph, use MockPolicy + real LeanPool):
# - One-step proof (True via trivial) with trajectory verification
# - Two-step proof (∀ n, n = n via intro n + rfl) with parent chain
# - Survives tactic failures (bad tactic + good tactic)
# - Unproved with bad tactics (budget exhaustion)
# - Trajectory record field-level validation (3 records, depths, labels)
# - Concurrent two proofs (2 workers, tokio::join!, no cross-contamination)
# - Timeout exits early (3s timeout with high node budget)
# - Medium arithmetic batch (5 theorems via omega: n+0=n, 0+n=n, a+b=b+a, n*1=n, 0≤n)
# - And commutativity multi-goal 4-step (intro → constructor → exact h.2 → exact h.1)
# - Medium logic batch (eq_symm via h.symm, eq_trans via h1.trans h2, modus_ponens via f hp)
# - Hard arithmetic with backtracking (wrong tactics before omega)

# Prover-core unit tests:
# - config: full TOML deserialize, optional lean_pool, CLI override priority

# Prover-core integration tests (mocks, no Lean/LLM):
# - Mock pipeline: search 2 theorems + write/read Parquet + verify summary
# - Mock unproved: all tactics fail, verify negative labels
# - Load test_theorems.json: verify >= 10 theorems
# - Eval reads Parquet: manually write + read_summary
# - Real search.toml is valid TOML
# - Train EBM mock pipeline: synthetic Parquet + mock encode_fn, verify checkpoint + config
# - Search with mock EBM: small EnergyHead saved/loaded, mock search with scorer active

# Prover-core Lean integration test (#[ignore], requires Pantograph):
# - Real LeanPool + MockPolicy: search 3 theorems, verify >= 2 proved, Parquet output

# Prover-core TinyLlama integration tests (#[ignore], require models/tinyllama-1.1b):
# - test_tinyllama_encode_and_ebm_train: real encode_only() → EBM train, dim=2048
# - test_tinyllama_ebm_scorer_roundtrip: train → save → load → score with real dimensions
# - test_tinyllama_shared_generator_policy_and_ebm: Arc<Mutex> interleaving policy + encode
# - test_tinyllama_full_pipeline_with_lean: search → Parquet → train EBM → search with EBM (also needs Pantograph)
```

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
# 1. Data preparation (CPU, ~30-60 min)
./scripts/prepare_data.sh              # or: ./scripts/prepare_data.sh --fallback

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
| `scripts/lean_start.sh` | Quick end-to-end validation on test_theorems (no training data needed) | Yes |

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NUM_WORKERS` | 64 | Number of Lean worker processes |
| `MAX_ITER` | 4 | Maximum iteration number (0-indexed) |
| `LLM_BASE` | `deepseek-ai/DeepSeek-Prover-V2-7B` | Base model for fine-tuning |
| `SKIP_BASELINE` | 0 | Set to 1 to skip Phase B baseline |
| `MODEL_PATH` | (none) | Local model dir for tokenizer in data prep |
| `MATHLIB_COMMIT` | `v4.27.0` | Mathlib4 tag to trace |

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
