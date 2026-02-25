# Phase 3: Search Engine + Trajectory Collection — Execution Plan

## Context

Phase 3 is where the BurnQED project becomes a working theorem prover. We wire together the lean-repl (Phase 1) and policy (Phase 2) crates into a best-first proof search pipeline that generates training data (Parquet trajectories) for the future EBM (Phase 4).

By the end: `cargo run -p prover-core -- search --llm-path ... --theorems ... --output trajectories.parquet`

**Source instructions**: `docs/phase3_instructions.md` (Prompts 3.1–3.11)

## Critical Design Issues

### 1. ProofHandle / ProofEnvironment trait mismatch

The instructions define `ProofEnvironment` with `start_proof()` + `apply_tactic()` as separate trait methods on `LeanPool`. But lean-repl has **no standalone `pool.apply_tactic()`** — stateIds are process-local, so `pool.start_proof()` returns a `ProofHandle` that must be used for ALL subsequent tactics.

**Solution**: Split into two traits:
```rust
#[async_trait]
pub trait ProofEnvironment: Send + Sync {
    async fn start_proof(&self, statement: &str) -> Result<Box<dyn TacticRunner + Send>>;
}

#[async_trait]
pub trait TacticRunner: Send {
    fn initial_state(&self) -> &ProofState;
    async fn apply_tactic(&mut self, state_id: u64, goal_id: usize, tactic: &str) -> Result<TacticResult>;
}
```
- Name is `TacticRunner`, NOT `ProofSession` — avoids collision with `lean_repl::ProofSession` (which is a different pattern: linear state tracking). The search engine needs tree-based access to arbitrary state IDs.
- `impl ProofEnvironment for Arc<LeanPool>` calls `start_proof_owned()`, returns `Box<ProofHandleOwned>`
- `impl TacticRunner for ProofHandleOwned` delegates to `run_tactic(state_id, goal_id, tactic)` with explicit state_id for tree-based search (revisiting any previous state)
- Search engine: calls `env.start_proof()` once per theorem, then uses the TacticRunner for all tactics
- Mock: `MockEnvironment` returns `MockTacticRunner` with canned responses

### 2. TacticGenerator `&mut self` vs trait `&self`

`TacticGenerator::generate_candidates()` takes `&mut self` (KV cache). The `PolicyProvider` trait needs `&self` for shared access.

**Solution**: `impl PolicyProvider for std::sync::Mutex<TacticGenerator>` — lock is held only during sync generation (no await points), so `std::sync::Mutex` is safe and efficient.

### 3. Missing workspace dependencies

Need to add: `async-trait` (for object-safe async traits), `indicatif` (progress bars).

### 4. Cross-crate type placement

`SearchResult`, `TheoremTask`, `TrajectoryRecord` live in the `trajectory` crate. The `search` crate depends on `trajectory` for these types — this is already in the stub's `Cargo.toml`.

---

## Delivery Parts (6 parts, each a focused implementation session)

---

## Part 1: trajectory crate — types, writer, reader ✅

**Goal**: Complete data layer for storing search results as Parquet files.

### Files to modify/create

| File | Action | Content |
|------|--------|---------|
| `Cargo.toml` (workspace root) | Edit | Add `async-trait = "0.1"`, `indicatif = "0.17"` to `[workspace.dependencies]` |
| `crates/trajectory/Cargo.toml` | Edit | Add `serde_json` dep (needed for TheoremIndex::from_json) |
| `crates/trajectory/src/types.rs` | Rewrite | TrajectoryRecord, TrajectoryLabel, SearchResult, TheoremTask, TheoremIndex, TrajectorySummary |
| `crates/trajectory/src/writer.rs` | Rewrite | TrajectoryWriter with Arrow schema, buffer, flush, finish, from_search_result |
| `crates/trajectory/src/reader.rs` | Rewrite | TrajectoryReader with read_all, read_multiple, read_summary, read_for_theorem |
| `crates/trajectory/src/lib.rs` | Rewrite | Module declarations + public API re-exports |

### Verification
```bash
cargo test -p trajectory
cargo check --workspace
```

---

## Part 2: search crate — node types and config ✅

**Goal**: Search data structures, priority queue ordering, proof path extraction.

### Files to modify/create

| File | Action | Content |
|------|--------|---------|
| `crates/search/Cargo.toml` | Edit | Add `async-trait`, `serde`, `anyhow`, `thiserror`, `serde_json` deps |
| `crates/search/src/config.rs` | Rewrite | SearchConfig with defaults matching configs/search.toml |
| `crates/search/src/node.rs` | Rewrite | SearchNode, ScoredNode (Ord for BinaryHeap), extract_proof_path |
| `crates/search/src/lib.rs` | Rewrite | Module declarations + re-exports |

### Verification
```bash
cargo test -p search
cargo check --workspace
```

---

## Part 3: search crate — engine, traits, mocks ✅

**Goal**: Best-first search algorithm with trait-based abstraction and mock tests.

### Files to modify/create

| File | Action | Content |
|------|--------|---------|
| `crates/search/src/engine.rs` | Rewrite | SearchEngine, PolicyProvider trait, ProofEnvironment/TacticRunner traits, ValueScorer trait |
| `crates/search/src/adapters.rs` | Create | Trait impls for real types (Mutex<TacticGenerator>, Arc<LeanPool>, ProofHandleOwned) |
| `crates/search/src/mocks.rs` | Create | MockPolicy, MockEnvironment, MockTacticRunner for unit testing |
| `crates/search/src/lib.rs` | Edit | Add `pub mod adapters; pub mod mocks;` and re-exports |

### Verification
```bash
cargo test -p search
cargo check --workspace
```

---

## Part 4: prover-core CLI + pipeline

**Goal**: Wire everything into a CLI binary with `search` and `eval` subcommands.

### Files to modify/create

| File | Action | Content |
|------|--------|---------|
| `crates/prover-core/Cargo.toml` | Edit | Add `indicatif` dep |
| `crates/prover-core/src/main.rs` | Rewrite | clap CLI with Search, Eval subcommands |
| `crates/prover-core/src/config.rs` | Rewrite | load_search_config, load_lean_config |
| `crates/prover-core/src/pipeline.rs` | Rewrite | run_search orchestration |

### Verification
```bash
cargo build -p prover-core
cargo run -p prover-core -- search --help
cargo check --workspace
```

---

## Part 5: Test data + integration tests

**Goal**: Sample theorems for local testing, end-to-end integration test.

### Files to create

| File | Action | Content |
|------|--------|---------|
| `data/test_theorems.json` | Create | 10 theorems of varying difficulty |
| `data/minif2f_sample.json` | Create | Placeholder with 2-3 known problems |
| `crates/prover-core/tests/integration.rs` | Create | Full pipeline integration tests (#[ignore]) |

### Verification
```bash
cargo test -p trajectory
cargo test -p search
cargo test -p prover-core -- --ignored --nocapture --test-threads=1
```

---

## Part 6: Polish — progress, error handling, statistics

**Goal**: Production-ready robustness for real search runs.

### Changes

| File | Changes |
|------|---------|
| `crates/prover-core/src/pipeline.rs` | CTRL-C handling, per-theorem error recovery, statistics summary block |
| `crates/search/src/engine.rs` | SearchStats struct, timing instrumentation |
| `CLAUDE.md` | Update Phase 3 status, add API summaries, cross-crate integration notes |

### Verification
```bash
cargo clippy --workspace
cargo test --workspace
cargo test -p prover-core -- --ignored --nocapture --test-threads=1
```

---

## Dependency Changes Summary

### Workspace `Cargo.toml` additions
```toml
async-trait = "0.1"
indicatif = "0.17"
```

### Crate-level additions
- `trajectory/Cargo.toml`: add `serde_json = { workspace = true }`
- `search/Cargo.toml`: add `async-trait`, `serde`, `anyhow`, `thiserror`, `serde_json`
- `prover-core/Cargo.toml`: add `indicatif`

---

## Execution Order

Parts 1 -> 2 -> 3 must be sequential (each depends on the prior).
Part 4 depends on Parts 1-3.
Part 5 depends on Part 4.
Part 6 depends on Parts 4-5.

**Recommended commit points**:
- After Part 1: "Add trajectory crate with Parquet writer/reader"
- After Parts 2+3: "Add search crate with best-first search engine"
- After Parts 4+5: "Add prover-core CLI and integration tests"
- After Part 6: "Polish search pipeline with progress and error handling"
