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
│   └── spindle_final_plan.md           # Full architecture plan with all code samples
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
│   │       ├── lib.rs
│   │       ├── engine.rs               # Priority queue, node expansion, scoring
│   │       ├── node.rs                 # SearchNode, ScoredNode (OrderedFloat)
│   │       └── config.rs
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
│       └── src/
│           ├── main.rs                 # clap: search, train-ebm, eval subcommands
│           ├── config.rs               # TOML config loading
│           └── pipeline.rs             # Expert iteration orchestrator
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
    └── run_iteration.sh
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
- [ ] Phase 3: Search engine + trajectory
- [ ] Phase 4: EBM in burn-rs
- [ ] Phase 5: Expert iteration
- [ ] Phase 6: burn-rs PRs

## Current Phase: 3 (Search engine + trajectory)

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

## Documentation Maintenance

When making architectural changes, adding new public types, or completing phases, **always update these docs to match**:

- `CLAUDE.md` (this file) — project layout, phase status, settled decisions, API summaries
- `docs/burn-qed_plan.md` — full plan with code samples, architecture diagrams, risk table, cost analysis
- `docs/phase1_instructions.md` (and future `phase*_instructions.md`) — step-by-step prompts and verification checklists

If a code change would make any code sample, type signature, or architectural description in these docs incorrect, fix the docs in the same change.

## Reference

Full plan with architecture diagrams, all code samples, loss functions, training loop, cost analysis, and risk mitigation: `docs/burn-qed_plan.md`
