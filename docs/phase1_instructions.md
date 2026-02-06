# Phase 0 + 1: Claude Code Instructions

Step-by-step prompts to paste into `claude` CLI. Run each one, review output, fix if needed, move to the next.

Before starting, ensure your environment has:
- Rust (latest stable via `rustup`)
- Lean 4 via `elan` (`elan default leanprover-lean4:stable`)
- Pantograph built from source: `https://github.com/lenianiva/Pantograph`
- Mathlib built locally (for Pantograph's `--env-path`): `git clone https://github.com/leanprover-community/mathlib4 && cd mathlib4 && lake build`

---

## Phase 0: Workspace Scaffold

### Prompt 0.1 — Create workspace and all crate stubs

```
Read CLAUDE.md carefully. Then create the full Cargo workspace for burn-qed with all crates listed in the project layout. Every crate should have:
- A Cargo.toml with appropriate dependencies (only what that crate needs — don't add burn deps to lean-repl, don't add tokio to ebm)
- Stub source files with the module structure from CLAUDE.md
- Empty pub structs/traits/functions with todo!() bodies so `cargo check --workspace` passes

For Phase 0, the crate dependency relationships are:
- prover-core depends on: lean-repl, policy, ebm, search, trajectory
- ebm depends on: burn-contrib
- search depends on: lean-repl, policy, ebm, trajectory
- lean-repl, policy, burn-contrib, trajectory are leaf crates (no internal deps)

The workspace Cargo.toml should define workspace.dependencies for shared versions.

Do NOT implement any real logic yet — just make the skeleton compile. Run `cargo check --workspace` at the end to verify.
```

### Prompt 0.2 — Add config files and scripts directory

```
Create the configs/ and scripts/ directories:

configs/models.toml:
- [encoder] section: mode = "shared", shared_hidden_dim = 4096
- [energy_head] section: d_hidden1 = 512, d_hidden2 = 256, dropout = 0.1, n_power_iterations = 5
- [llm] section: model_name = "deepseek-ai/DeepSeek-Prover-V2-7B", max_seq_len = 2048, num_candidates = 32, temperature = 0.8

configs/search.toml:
- [search] section: max_nodes = 600, max_depth = 50, beam_width = 8, alpha = 0.5, beta = 0.5, timeout_per_theorem = 600
- [search.iteration_0] section: normal_temperature = 0.8, noise_temperature = 1.2, noise_fraction = 0.3
- [lean_pool] section: num_workers = 64, max_requests_per_worker = 1000, max_lifetime_secs = 1800, tactic_timeout_secs = 30

Also create an empty scripts/run_iteration.sh placeholder and python/ directory stubs.

Copy docs/spindle_final_plan.md from the plan document (it's already generated, just create the docs/ dir and note it should be placed there).
```

### Checkpoint

```bash
cargo check --workspace  # Must pass
```

---

## Phase 1: Lean REPL Client

### Prompt 1.1 — Types and error definitions

```
Implement crates/lean-repl/src/types.rs with the core types for the Lean REPL client:

1. LeanError enum using thiserror:
   - ProcessDied — Lean process exited unexpectedly
   - Timeout(u64) — tactic timed out after N seconds
   - Protocol(String) — JSON parse error or unexpected response format
   - LeanMessage(String) — Lean reported an error (tactic failed, unknown identifier, etc.)
   - Io(std::io::Error) — IO error (transparent)

2. Goal struct:
   - goal_id: usize
   - hypothesis: Vec<String> (parsed from pretty-printed state)
   - target: String (the part after ⊢)
   - raw: String (full pretty-printed state as returned by Pantograph)

3. TacticResult enum:
   - Success { state_id: u64, goals: Vec<Goal> } — tactic applied, new goals
   - ProofComplete { state_id: u64 } — goals list is empty, proof done
   - Failed { message: String } — Lean error message

4. ProofState struct:
   - state_id: u64
   - goals: Vec<Goal>

5. LeanPoolConfig struct (derives Deserialize, Debug, Clone):
   - num_workers: usize (default 4)
   - max_requests_per_worker: u64 (default 1000)
   - max_lifetime_secs: u64 (default 1800)
   - tactic_timeout_secs: u64 (default 30)
   - pantograph_path: PathBuf
   - lean_env_path: PathBuf (path to Mathlib build dir)

Parse goals from raw Pantograph strings: split on `⊢` — everything before is hypotheses (split by newlines), everything after is the target.

Add unit tests for Goal parsing from sample Pantograph output strings like:
- "n : Nat\n⊢ n + 0 = n"
- "⊢ ∀ (n : Nat), n + 0 = n" (no hypotheses)
- "h : p\nh2 : q\n⊢ p ∧ q" (multiple hypotheses)
```

### Prompt 1.2 — Pantograph protocol types

```
Implement crates/lean-repl/src/protocol.rs with serde types for Pantograph JSON communication:

1. PantographRequest enum (tagged by "cmd" field):
   - GoalStart { expr: String }
     → serializes to: {"cmd": "goal.start", "expr": "..."}
   - GoalTactic { state_id: u64, goal_id: usize, tactic: String }
     → serializes to: {"cmd": "goal.tactic", "stateId": 0, "goalId": 0, "tactic": "..."}

   Use #[serde(tag = "cmd", rename_all = "camelCase")] or implement custom Serialize
   since Pantograph uses camelCase field names (stateId, goalId) and the cmd field
   uses dot notation ("goal.start", "goal.tactic").

   IMPORTANT: Pantograph uses "stateId" and "goalId" (camelCase), not snake_case.

2. PantographResponse enum (untagged, try each variant):
   - Success { state_id: u64, goals: Vec<String> }
     → from: {"stateId": 2, "goals": ["n : Nat\n⊢ n + 0 = n"]}
   - Error { error: String }
     → from: {"error": "unknown tactic"}

   Use #[serde(untagged)] and match on presence of "error" field.
   The "stateId" field uses camelCase. Use #[serde(rename = "stateId")].

Add thorough unit tests for serialization/deserialization:
- Serialize GoalStart and verify exact JSON string matches what Pantograph expects
- Serialize GoalTactic with state_id=5, goal_id=0, tactic="simp" and verify JSON
- Deserialize a success response with goals
- Deserialize a success response with empty goals (proof complete)
- Deserialize an error response
- Deserialize malformed JSON → should return a parse error, not panic
```

### Prompt 1.3 — LeanWorker implementation

```
Implement crates/lean-repl/src/worker.rs with the LeanWorker struct.

Read the CLAUDE.md sections on Pantograph protocol and worker recycling carefully before coding.

LeanWorker manages a single Pantograph child process:

Fields:
- child: tokio::process::Child
- stdin: tokio::io::BufWriter<tokio::process::ChildStdin>
- stdout: tokio::io::BufReader<tokio::process::ChildStdout>
- requests_handled: u64
- started_at: std::time::Instant
- config: LeanPoolConfig (shared reference or clone of relevant fields)

Methods:

1. async fn spawn(config: &LeanPoolConfig) -> Result<Self, LeanError>
   - Launch pantograph with: Command::new(&config.pantograph_path).arg("--env-path").arg(&config.lean_env_path)
   - Set stdin(Stdio::piped()), stdout(Stdio::piped()), stderr(Stdio::null())
   - Wrap stdin in BufWriter, stdout in BufReader
   - Initialize counters

2. fn needs_recycling(&self) -> bool
   - true if requests_handled >= config.max_requests_per_worker
   - OR started_at.elapsed().as_secs() >= config.max_lifetime_secs

3. async fn recycle(&mut self) -> Result<(), LeanError>
   - Kill old process (child.kill().await, child.wait().await — ignore errors)
   - Spawn fresh process with same config
   - Reset counters and timestamp
   - tracing::debug!("Recycled Lean worker")

4. async fn send_raw(&mut self, request: &PantographRequest) -> Result<PantographResponse, LeanError>
   - Serialize request to JSON string
   - CRITICAL: Write json bytes + b"\n" to stdin, then flush
   - Read one line from stdout with tokio::time::timeout(Duration::from_secs(config.tactic_timeout_secs))
   - On timeout: recycle self, return LeanError::Timeout
   - On Ok(0) bytes read: return LeanError::ProcessDied
   - Parse JSON line into PantographResponse
   - Increment requests_handled
   - Return result

5. async fn start_proof(&mut self, expr: &str) -> Result<ProofState, LeanError>
   - Send GoalStart request
   - Convert response to ProofState (parse goals)

6. async fn apply_tactic(&mut self, state_id: u64, goal_id: usize, tactic: &str) -> Result<TacticResult, LeanError>
   - Send GoalTactic request
   - Convert response:
     - Error response → TacticResult::Failed
     - Success with empty goals → TacticResult::ProofComplete
     - Success with goals → TacticResult::Success (parse each goal string)
   - If self.needs_recycling(), log but DON'T recycle mid-proof. Set a flag so pool recycles between proofs.

Add unit tests:
- Test that send_raw correctly appends \n (mock or verify serialization)
- Test needs_recycling logic with various counter/time combinations
```

### Prompt 1.4 — LeanPool + ProofHandle implementation

```
Implement crates/lean-repl/src/pool.rs with the LeanPool worker pool and ProofHandle pattern.

IMPORTANT: Pantograph stateId values are process-local. Never release a worker between
start_proof and run_tactic — the next acquire might return a different worker that doesn't
have that stateId. Instead, start_proof() returns a ProofHandle that holds the worker.

Use tokio::sync::Semaphore to limit concurrent access and a Mutex<Vec<LeanWorker>> as a simple free list.

Main types:
- LeanPool: manages N workers with semaphore
- ProofHandle<'a>: holds WorkerGuard for proof lifetime (borrowed)
- ProofHandleOwned: same but 'static for tokio::spawn (holds Arc<LeanPool>)
- WorkerGuard<'a>: RAII guard for raw worker checkout (borrowed)
- WorkerGuardOwned: same but 'static (holds Arc<LeanPool>)

LeanPool methods:
1. async fn new(config) -> Result<Self, LeanError> — spawn workers
2. async fn start_proof(&self, expr) -> Result<ProofHandle<'_>, LeanError> — checkout + goal.start
3. async fn start_proof_owned(self: &Arc<Self>, expr) -> Result<ProofHandleOwned, LeanError>
4. async fn checkout(&self) -> Result<WorkerGuard<'_>, LeanError> — raw worker access
5. async fn checkout_owned(self: &Arc<Self>) -> Result<WorkerGuardOwned, LeanError>
6. fn available_workers(&self) -> usize
7. async fn shutdown(&self)

ProofHandle methods:
- fn state_id(&self) -> u64 — initial state ID
- fn initial_state(&self) -> &ProofState
- async fn run_tactic(&mut self, state_id, goal_id, tactic) -> Result<TacticResult, LeanError>
- fn worker(&mut self) -> &mut LeanWorker — advanced access

There is NO standalone pool.run_tactic() — it's always wrong with >1 worker.

Workers are returned to the pool via Drop on the guard/handle.
Use try_lock in Drop (can't do async) — if contended, leak the worker.

Add unit tests (that don't require Lean):
- Semaphore correctly limits concurrency
```

### Prompt 1.5 — Lib.rs public API and re-exports

```
Wire up crates/lean-repl/src/lib.rs to export the public API:

pub mod worker;
pub mod pool;
pub mod protocol;
pub mod session;
pub mod types;

// Re-export main types at crate root
pub use pool::{LeanPool, ProofHandle, ProofHandleOwned, WorkerGuard, WorkerGuardOwned};
pub use session::ProofSession;
pub use types::{LeanError, LeanPoolConfig, ProofState, Goal, TacticResult, discover_pantograph};
pub use protocol::{PantographRequest, PantographResponse};
pub use worker::LeanWorker;

Make sure `cargo check -p lean-repl` passes with no warnings.
Run `cargo test -p lean-repl` — all unit tests should pass.
Fix any compilation errors.
```

### Prompt 1.6 — Integration tests

```
Create crates/lean-repl/tests/integration.rs with real Pantograph integration tests.

All tests should be marked #[ignore] so they don't run in CI without Lean installed.
They should be async (#[tokio::test]).

Pantograph is auto-discovered via LeanPoolConfig::with_bundled_pantograph():
  1. PANTOGRAPH_PROJECT env var (if set)
  2. vendor/Pantograph/ submodule (default)

IMPORTANT: All multi-step proofs must use ProofHandle to hold the worker.
The old pattern of pool.start_proof() + pool.run_tactic() is broken with >1 worker.

Write these tests:

1. test_single_worker_simple_proof
   - pool.start_proof("forall (n : Nat), n = n") → ProofHandle
   - proof.run_tactic(sid, None, "intro n") → verify hypothesis and goal
   - proof.run_tactic(s1, None, "rfl") → verify ProofComplete

2. test_single_worker_tactic_error
   - Start proof, apply "nonexistent_tactic_12345" via proof handle
   - Verify TacticResult::Failed

3. test_single_worker_multiple_goals
   - Prove "forall (p q : Prop), p -> q -> p /\ q" via proof handle
   - Apply intros + "exact And.intro hp hq"

4. test_proof_session_simple
   - ProofSession::new(&pool, ...) — internally holds ProofHandle
   - session.apply("intro n"), session.apply("rfl")

5. test_pool_sequential_100
   - 4 workers, 100 sequential proofs
   - Each proof gets its own ProofHandle; worker returned between proofs

6. test_pool_concurrent_20
   - Arc<LeanPool> with 4 workers
   - 20 concurrent proofs via tokio::spawn + pool.start_proof_owned()

7. test_concurrent_multi_step_isolation
   - 4 workers, 10 concurrent multi-step proofs (different theorems)
   - Includes branching (constructor → 2 goals → targeted goalId)
   - Verifies state immutability after proof completes
   - THE key test for the ProofHandle state-routing fix

8. test_worker_recycling
   - max_requests_per_worker = 9, run 8 proofs (3 requests each)
   - Verify recycling happens transparently

9. test_timeout_recovery
   - 2s timeout, verify pool still works after timeout

10. test_proof_search_simulation
    - checkout() raw worker, comprehensive branching/state-immutability test
    - 20 rapid-fire theorems, error recovery, multi-goal proofs

Run with:
  cargo test -p lean-repl -- --ignored --nocapture --test-threads=1

Use --test-threads=1 to avoid overwhelming the system with concurrent pool creation.
```

### Prompt 1.7 — Polish and edge cases

```
Review and harden the lean-repl crate:

1. Add tracing spans to key operations:
   - worker.rs: tracing::debug_span!("lean_worker") on spawn/recycle
   - pool.rs: tracing::instrument on acquire/release/run_tactic
   - Include state_id and tactic preview (first 50 chars) in tactic spans

2. Handle edge cases in protocol.rs:
   - What if Pantograph outputs multiple lines before we read? (shouldn't happen with JSON lines, but be defensive)
   - What if the JSON response has extra fields we don't expect? (use #[serde(deny_unknown_fields)] only if Pantograph is strict, otherwise allow unknown fields)
   - What if goals contain unicode (they will — Lean uses ∀, ⊢, →, etc.)?
     Ensure all string handling is UTF-8 safe.

3. Add a ProofSession helper that tracks state across multiple tactic applications.
   ProofSession internally holds a ProofHandle (not &LeanPool), ensuring the worker
   stays pinned for the entire session:
   ```rust
   pub struct ProofSession<'a> {
       handle: ProofHandle<'a>,               // holds worker for session lifetime
       current_state: ProofState,
       history: Vec<(String, TacticResult)>,   // (tactic, result) pairs
       completed: bool,
   }

   impl<'a> ProofSession<'a> {
       pub async fn new(pool: &'a LeanPool, expr: &str) -> Result<Self, LeanError>;
       pub async fn apply(&mut self, tactic: &str) -> Result<&TacticResult, LeanError>;
       pub async fn apply_to_goal(&mut self, goal_id: u64, tactic: &str) -> Result<&TacticResult, LeanError>;
       pub fn is_complete(&self) -> bool;
       pub fn current_goals(&self) -> &[Goal];
       pub fn current_state(&self) -> &ProofState;
       pub fn depth(&self) -> usize;
       pub fn history(&self) -> &[(String, TacticResult)];
   }
   ```

4. Make sure `cargo clippy -p lean-repl` passes with no warnings.

5. Make sure `cargo doc -p lean-repl --no-deps` generates clean documentation.

6. Run all tests one more time:
   cargo test -p lean-repl
   LEAN_ENV_PATH=... cargo test -p lean-repl -- --ignored --nocapture
```

---

## Verification Checklist

After all Phase 1 prompts, verify:

```bash
# Workspace compiles
cargo check --workspace

# lean-repl unit tests pass (20 unit + 2 doc tests)
cargo test -p lean-repl

# No clippy warnings on lean-repl
cargo clippy -p lean-repl

# Integration tests pass (requires Pantograph built: lake build -R in vendor/Pantograph/)
cargo test -p lean-repl -- --ignored --nocapture --test-threads=1

# Specific validations:
# - ProofHandle holds worker for entire proof (no state ID routing bugs)
# - Can start a proof and apply at least 3 tactics in sequence on same handle
# - Can handle tactic errors without crashing
# - Pool survives 100 sequential proofs without hanging (4 workers)
# - 20 concurrent proofs complete via ProofHandleOwned (4 workers)
# - 10 concurrent multi-step branching proofs with state isolation (4 workers)
# - Worker recycling triggers and works correctly
# - Timeout recovery works
# - ProofSession correctly holds ProofHandle internally
```

---

## Troubleshooting

### "Pantograph hangs after first command"
You forgot the `\n` after the JSON. Check that `send_raw` writes `json_bytes` then `b"\n"` then flushes.

### "Connection refused" or "No such file"
Pantograph path is wrong. Try running `pantograph --help` manually to verify it's installed. Check the PANTOGRAPH_PATH env var.

### "unknown command" errors from Pantograph
Pantograph versions differ. The command format might be `{"cmd": "goal.start", ...}` or `{"command": "goal_start", ...}` depending on version. Check Pantograph's actual protocol by running it manually:
```bash
echo '{"cmd": "goal.start", "expr": "∀ (n : Nat), n = n"}' | pantograph --env-path /path/to/mathlib/.lake/build
```

### "lake build" for Mathlib takes forever
Normal — first Mathlib build takes 30-60 minutes. Subsequent builds are cached. Make sure you have `~8GB` free RAM.

### Worker recycling tests are flaky
Timing-based tests are inherently flaky. Use request-count-based recycling for deterministic tests, not time-based.

### "unknown stateId" or wrong state in multi-worker pool
You're releasing the worker between `start_proof` and `run_tactic`. Pantograph
`stateId` values are process-local — each worker has its own counter. Always use
`ProofHandle` (returned by `pool.start_proof()`) to keep the worker pinned:
```rust
// CORRECT: worker held for entire proof
let mut proof = pool.start_proof("...").await?;
let r = proof.run_tactic(proof.state_id(), None, "intro n").await?;

// WRONG: no standalone pool.run_tactic() exists (removed)
```

### Integration tests need Pantograph built
```bash
# One-time setup:
cd vendor/Pantograph && lake build -R
# Then run:
cargo test -p lean-repl -- --ignored --nocapture --test-threads=1
```

### All integration tests timeout at pool creation
Running all 10 tests in parallel spawns many Pantograph processes simultaneously.
Use `--test-threads=1` to run them sequentially.
