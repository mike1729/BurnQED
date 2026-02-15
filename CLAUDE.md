# burn-qed

Lean 4 theorem prover combining LLM policy (tactic generation) with Energy-Based Model value function (proof state scoring) to guide best-first proof search. Trained via expert iteration.

## Architecture Overview

```
DeepSeek-Prover-V2-7B (SGLang server)
├── Policy head: autoregressive tactic generation (LM head)
└── Mean-pool hidden states → Vec<f32> via /encode endpoint
                                    │
                                    ▼
                    Energy Head (burn-rs, trainable)
                    SpectralNorm MLP: 4096 → 2048 → 1024 → 512 → 1
                    Output: scalar energy (lower = more provable)

Lean 4 REPL Pool (tokio, Pantograph JSON protocol)
└── Verifies tactics against proof states, returns new goals
```

Single shared 7B backbone serves both policy and value function (AlphaZero-style). The energy head (~11M params) is the only component trained in Rust via burn-rs. LLM fine-tuning happens in Python with HuggingFace PEFT/LoRA.

## Settled Architecture Decisions — Do NOT Change

These were decided after two external review rounds. Do not revisit without explicit instruction.

1. **Shared 7B backbone**, not separate 1.3B encoder. `EncoderBackend` enum allows switching via config.
2. **SGLang inference server.** No in-process model loading.
3. **SpectralNorm Option C** (random reinit per forward, 5 power iterations).
4. **Single optimizer for EBM.** 7B encoder frozen (served by SGLang).
5. **No "skip easy" EBM bypass.** Always score with EBM when available.
6. **No dual tokenizer / bag-of-embeddings.** Smart truncation if context too long.
7. **Worker recycling for Lean.** 1000 requests OR 30 minutes TTL.
8. **Pantograph protocol:** JSON lines terminated by `\n`. Missing newline hangs process. 30s timeout on all reads.
9. **ProofHandle pattern.** `stateId` is process-local — hold worker for proof lifetime. Use `ProofHandleOwned` for `tokio::spawn`.

## Code Conventions

### Rust
- Edition 2021, `tokio` for async, `tracing` for logging (never `println!` in library code)
- `anyhow::Result` for app code (prover-core); concrete `thiserror` enums for library crates
- `serde::Deserialize` on config types, load from TOML
- burn-rs: `#[derive(Module, Debug)]` on models, `#[derive(Config, Debug)]` on configs
- Tests: `#[cfg(test)] mod tests` per file; integration tests in `crates/*/tests/`

### File Organization
- One major type per file (`worker.rs` → `LeanWorker`, `pool.rs` → `LeanPool`)
- `mod.rs` only for `pub mod` declarations; `lib.rs` re-exports public API
- Crate names: kebab-case; modules: snake_case; types: PascalCase; configs: `{Thing}Config`

## Key Dependencies (pinned — do not upgrade without testing)

```toml
tokio = "1", serde = "1", serde_json = "1", anyhow = "1", tracing = "0.1", thiserror = "2"
burn = "0.16", arrow = "53", parquet = "53", reqwest = "0.12"
```

## Pantograph Protocol

Bundled at `vendor/Pantograph/` (submodule, pinned `d047b1d`). Auto-discovered by `LeanPoolConfig::with_bundled_pantograph()`.

**Critical:** Every request is a JSON object on one line, terminated by `\n`. Missing newline = hang.

```jsonc
{"cmd": "goal.start", "payload": {"copyFrom": "Nat.add_comm"}}  // by name
{"cmd": "goal.start", "payload": {"expr": "∀ (n : Nat), n + 0 = n"}}  // by expr
{"cmd": "goal.tactic", "stateId": 0, "goalId": 0, "tactic": "intro n"}
// Success: {"stateId": 1, "goals": ["n : Nat\n⊢ n + 0 = n"]}
// Proof done: {"stateId": 2, "goals": []}
// Error: {"error": "unknown tactic 'bad_tactic'"}
```

- `stateId` is monotonically increasing and immutable — revisit any previous state
- Goals: `hyp1 : Type1\nhyp2 : Type2\n⊢ goal_type`
- Always redirect stderr to null; always use 30s read timeout

## Phase Status

- [x] Phase 0-5: All implemented (lean-repl, policy, search, trajectory, ebm, prover-core, expert iteration)
- [ ] Phase 6: burn-rs upstream PRs
- **Current:** Experiment execution — generating EBM training data, tuning pipeline

## CLI Quick Reference

```bash
# Proof search
cargo run -p prover-core -- search --server-url URL --theorems FILE --output FILE [--ebm-path DIR] [--dry-run]

# EBM training from trajectories
cargo run -p prover-core -- train-ebm --trajectories FILE --server-url URL --output-dir DIR

# Generate contrastive training data (see docs/ebm_overhaul.md Part 3 for full design)
cargo run -p prover-core -- generate-negatives --tactic-pairs FILE --server-url URL --output FILE [--min-steps N]

# Evaluation
cargo run -p prover-core -- eval --server-url URL --theorems FILE --budgets 50,100,200
cargo run -p prover-core -- summary --trajectories FILE
cargo run -p prover-core -- compare --baseline FILE --experiment FILE
```

## Testing Policy

Integration tests (`#[ignore]`) spawn Pantograph — **must** use `--test-threads=1`. Typical runtimes:
- `cargo test -p lean-repl -- --ignored --test-threads=1` — ~60-90s
- `cargo test -p search -- --ignored --test-threads=1` — ~60s
- `cargo test -p prover-core -- --ignored --test-threads=1` — ~15s
- `cargo test -p policy -- --ignored --test-threads=1` — requires SGLang server

## Reference Docs

- `docs/burn-qed_plan.md` — Full plan: architecture diagrams, code samples, loss functions, cost analysis
- `docs/ebm_overhaul.md` — EBM architecture upgrade + generate-negatives pipeline design
- `docs/experiment_guide.md` — Scripts, environment variables, throughput tuning, output layout, go/no-go checkpoints
- `docs/cloud_deployment.md` — RunPod/Lambda setup, migration
- `docs/sglang.md` — SGLang server configuration
