# Phase 3: Search Engine + Trajectory Collection — Claude Code Instructions

Step-by-step prompts for `claude` CLI. Phases 1 (lean-repl) and 2 (policy/candle) are complete.

Phase 3 builds two crates: `search` (best-first proof search algorithm) and `trajectory` (Parquet I/O for storing search results). It also wires up `prover-core` as the CLI binary that ties everything together.

By the end of Phase 3, you can run: `cargo run -p prover-core -- search --theorems problems.json --output trajectories.parquet` and get an LLM-only theorem prover that writes training data for the EBM.

## Prerequisites

- Phase 1 complete: `lean-repl` working with Pantograph
- Phase 2 complete: `policy` crate loads DeepSeek-Prover-V2-7B, generates tactics, encode_only() works
- Know the actual numbers from Phase 2 benchmarks:
  - How long does generate_candidates(state, 8) take on your hardware?
  - How long does encode_only() take?
  - These determine search timeout budgets.

---

## Prompt 3.1 — Trajectory crate: types and Arrow schema

```
Implement the trajectory crate (crates/trajectory/). This crate handles reading and writing search results as Parquet files. It's a dependency for both the search engine and the future EBM training pipeline.

Read CLAUDE.md and docs/spindle_final_plan.md Section 5.1 for the data schema.

Start with crates/trajectory/src/types.rs:

1. TrajectoryRecord struct (Clone, Debug, Serialize, Deserialize):
   - theorem_name: String — which theorem this state belongs to
   - state_id: u64 — Pantograph state ID (unique within a proof attempt)
   - state_pp: String — pretty-printed proof state (the raw text from Pantograph)
   - tactic_applied: String — the tactic that was applied to reach this state (empty for root)
   - parent_state_id: Option<u64> — the state this was generated from (None for root)
   - label: TrajectoryLabel — positive (on proof path), negative (dead end), or unknown
   - depth_from_root: u32 — number of tactic applications from the initial goal
   - remaining_depth: i32 — steps remaining to QED on the proof path, -1 if unknown
   - llm_log_prob: f64 — log-probability of the tactic from the LLM
   - ebm_score: f64 — EBM energy score (0.0 if EBM not used)
   - is_proof_complete: bool — true if this state has no remaining goals
   - timestamp_ms: u64 — unix timestamp in milliseconds

2. TrajectoryLabel enum (Clone, Debug, Serialize, Deserialize, PartialEq):
   - Positive — on the path from root to QED
   - Negative — dead end, error, or timeout
   - Unknown — not yet labeled (for states discovered during search before outcome known)

   Implement Display for this: "positive", "negative", "unknown"

3. SearchResult struct (Clone, Debug):
   - theorem_name: String
   - proved: bool
   - proof_tactics: Vec<String> — the tactic sequence if proved, empty otherwise
   - nodes_expanded: u32
   - total_states: u32
   - max_depth_reached: u32
   - wall_time_ms: u64
   - all_records: Vec<TrajectoryRecord> — every state visited during search

4. TheoremTask struct (Clone, Debug, Deserialize):
   - name: String — theorem identifier (e.g., "Nat.add_zero")
   - statement: String — the Lean 4 type/proposition to prove (e.g., "∀ (n : Nat), n + 0 = n")
   - Optional: file_path, line_number (for tracking provenance from Mathlib)

5. TheoremIndex struct (Clone, Debug, Deserialize):
   - theorems: Vec<TheoremTask>
   With methods:
   - fn from_json(path: &Path) -> Result<Self> — load from JSON file
   - fn len(&self) -> usize

Make sure `cargo check -p trajectory` passes.
```

### Prompt 3.2 — Trajectory writer (Parquet)

```
Implement crates/trajectory/src/writer.rs — writes TrajectoryRecords to Parquet files using arrow-rs.

TrajectoryWriter struct:

Fields:
- schema: Arc<arrow::datatypes::Schema>
- records: Vec<TrajectoryRecord> — buffered in memory
- output_path: PathBuf
- flush_threshold: usize — flush to disk every N records (default 10_000)

Define the Arrow schema:

  fn trajectory_schema() -> Schema {
      Schema::new(vec![
          Field::new("theorem_name", DataType::Utf8, false),
          Field::new("state_id", DataType::UInt64, false),
          Field::new("state_pp", DataType::Utf8, false),
          Field::new("tactic_applied", DataType::Utf8, false),
          Field::new("parent_state_id", DataType::UInt64, true),  // nullable
          Field::new("label", DataType::Utf8, false),
          Field::new("depth_from_root", DataType::UInt32, false),
          Field::new("remaining_depth", DataType::Int32, false),
          Field::new("llm_log_prob", DataType::Float64, false),
          Field::new("ebm_score", DataType::Float64, false),
          Field::new("is_proof_complete", DataType::Boolean, false),
          Field::new("timestamp_ms", DataType::UInt64, false),
      ])
  }

Methods:

1. fn new(output_path: PathBuf) -> Self
   - Create writer with schema, empty buffer

2. fn record(&mut self, record: TrajectoryRecord)
   - Push to buffer
   - If buffer >= flush_threshold, call flush()

3. fn flush(&mut self) -> Result<()>
   - Convert Vec<TrajectoryRecord> to Arrow RecordBatch
   - Build each column array: StringArray, UInt64Array, etc.
   - For parent_state_id (Option<u64>): use UInt64Array with nulls
   - Append batch to Parquet file (using ArrowWriter in append mode)

4. fn finish(mut self) -> Result<PathBuf>
   - Flush remaining records
   - Close the ArrowWriter
   - Log: tracing::info!(path, num_records, "Trajectory file written")
   - Return the output path

5. fn from_search_result(result: &SearchResult) -> Vec<TrajectoryRecord>
   - Static method: label all records in a SearchResult
   - If result.proved:
     - Walk the proof path from root to QED, label those states Positive
     - Compute remaining_depth for positive states (distance to QED)
     - Label all other states Negative
   - If !result.proved:
     - Label ALL states Negative

Important implementation detail: opening a Parquet file for appending. Use:
  let file = File::create(&self.output_path)?;
  let mut writer = ArrowWriter::try_new(file, Arc::new(self.schema.clone()), None)?;
  // Write batches...
  writer.close()?;

If the file already exists and we need true append, we'd need to read and rewrite.
For simplicity, buffer everything and write once in finish(). This works fine for
per-theorem-batch writing.

Add unit tests:
- Write 100 records, finish, verify file exists and is valid Parquet
- Read the file back with ParquetRecordBatchReader, verify row count = 100
- Verify nullable parent_state_id works (some None, some Some)
- Test from_search_result with a mock proved result and verify label assignment
- Test from_search_result with an unproved result (all negative)
```

### Prompt 3.3 — Trajectory reader

```
Implement crates/trajectory/src/reader.rs — reads TrajectoryRecords from Parquet files.

TrajectoryReader with static methods:

1. fn read_all(path: &Path) -> Result<Vec<TrajectoryRecord>>
   - Open Parquet file
   - Read all record batches
   - Convert each row to TrajectoryRecord
   - Handle nullable parent_state_id (null → None)
   - Return flat Vec

2. fn read_multiple(paths: &[PathBuf]) -> Result<Vec<TrajectoryRecord>>
   - Read and concatenate records from multiple files
   - Used for combining trajectories across iterations:
     "trajectories/iter_0.parquet" + "trajectories/iter_0_noisy.parquet"

3. fn read_summary(path: &Path) -> Result<TrajectorySummary>
   - Quick stats without loading all records into memory
   - TrajectorySummary:
     - total_records: usize
     - positive_count: usize
     - negative_count: usize
     - unique_theorems: usize
     - proved_theorems: usize (theorems that have at least one is_proof_complete=true)

4. fn read_for_theorem(path: &Path, theorem_name: &str) -> Result<Vec<TrajectoryRecord>>
   - Read only records matching a specific theorem
   - Uses row group filtering if possible, or just filters after reading

Wire up crates/trajectory/src/lib.rs:
  pub mod types;
  pub mod writer;
  pub mod reader;
  pub use types::*;
  pub use writer::TrajectoryWriter;
  pub use reader::TrajectoryReader;

Add integration tests:
- Round-trip: write records → read them back → verify equality
- read_summary on a file with 50 positive and 50 negative → verify counts
- read_multiple on two files → verify combined count
- read_for_theorem filters correctly

Run: cargo test -p trajectory
```

### Prompt 3.4 — Search crate: node types and config

```
Implement the search crate types in crates/search/src/.

src/config.rs — SearchConfig:

  #[derive(Debug, Clone, Deserialize)]
  pub struct SearchConfig {
      pub max_nodes: u32,              // 600 — total node budget
      pub max_depth: u32,              // 50 — max tactic depth
      pub beam_width: usize,           // 8 — top-k candidates per expansion
      pub alpha: f64,                  // 0.5 — LLM log-prob weight in combined score
      pub beta: f64,                   // 0.5 — EBM score weight (0.0 if no EBM)
      pub timeout_per_theorem: u64,    // 600 seconds
      pub num_candidates: usize,       // 32 — tactics generated from LLM per expansion
      pub temperature: f64,            // 0.8 — LLM sampling temperature
  }

  impl Default for SearchConfig — use the values from configs/search.toml

src/node.rs — SearchNode and priority queue types:

  #[derive(Clone, Debug)]
  pub struct SearchNode {
      pub state_id: u64,              // Pantograph state ID
      pub state_pp: String,           // Pretty-printed proof state
      pub goals: Vec<Goal>,           // Parsed goals from lean-repl
      pub parent: Option<usize>,      // Index into search tree (not state_id)
      pub tactic_applied: String,     // Tactic used to reach this node
      pub depth: u32,                 // Depth from root
      pub llm_log_prob: f64,          // Log-prob of the tactic that created this node
      pub ebm_score: f64,             // EBM score (0.0 if no EBM)
      pub is_terminal: bool,          // true if goals is empty (proof complete)
  }

  impl SearchNode:
    fn combined_score(&self, alpha: f64, beta: f64) -> f64
      - alpha * self.llm_log_prob + beta * self.ebm_score
      - For the root node (no tactic), use 0.0 for llm_log_prob

  For the priority queue, create:

  #[derive(Clone, Debug)]
  pub struct ScoredNode {
      pub node_index: usize,          // Index into the search tree Vec<SearchNode>
      pub score: OrderedFloat<f64>,   // Combined score for priority ordering
  }

  impl Ord, PartialOrd, Eq, PartialEq for ScoredNode — ordered by score (max-heap)

  The search tree is stored as Vec<SearchNode>. Each node references its parent by index.
  This is simpler and faster than reference-counted tree pointers.

  Add a method to extract the proof path:

  fn extract_proof_path(tree: &[SearchNode], terminal_index: usize) -> Vec<&SearchNode>
    - Walk from terminal node to root via parent indices
    - Reverse to get root-to-QED order
    - Return the sequence of nodes

Add unit tests:
- ScoredNode ordering: higher score should come first in BinaryHeap
- extract_proof_path on a simple 3-node tree
- SearchNode::combined_score with various alpha/beta
```

### Prompt 3.5 — Search engine core

```
Implement crates/search/src/engine.rs — the best-first search algorithm.

This is the heart of the prover. Read CLAUDE.md section on search and docs/spindle_final_plan.md Section 8.1 carefully.

SearchEngine struct:

  pub struct SearchEngine {
      config: SearchConfig,
  }

The engine is stateless — it takes references to the policy model and Lean pool for each search call. This makes it easy to reuse.

Key types for the search interface (define as traits so we can test with mocks):

  /// Trait for tactic generation (implemented by TacticGenerator)
  #[async_trait]
  pub trait PolicyProvider: Send + Sync {
      async fn generate_candidates(
          &self,
          proof_state: &str,
          n: usize,
          temperature: f64,
      ) -> Result<Vec<GeneratedTactic>>;
  }

  /// Trait for proof state manipulation (implemented by LeanPool)
  #[async_trait]
  pub trait ProofEnvironment: Send + Sync {
      async fn start_proof(&self, statement: &str) -> Result<ProofState>;
      async fn apply_tactic(
          &self,
          state_id: u64,
          goal_id: usize,
          tactic: &str,
      ) -> Result<TacticResult>;
  }

  /// Optional trait for EBM scoring (None in Phase 3, added in Phase 4)
  pub trait ValueScorer: Send + Sync {
      fn score(&self, proof_state: &str) -> f64;
  }

Main method:

  pub async fn search(
      &self,
      theorem: &TheoremTask,
      policy: &dyn PolicyProvider,
      env: &dyn ProofEnvironment,
      scorer: Option<&dyn ValueScorer>,  // None for LLM-only search
  ) -> Result<SearchResult>

Algorithm:

  1. Start proof: env.start_proof(&theorem.statement)
  2. Create root SearchNode from initial ProofState
  3. Initialize:
     - tree: Vec<SearchNode> = vec![root_node]
     - frontier: BinaryHeap<ScoredNode> with root
     - nodes_expanded: 0
     - start_time = Instant::now()

  4. Loop while:
     - frontier is not empty
     - nodes_expanded < config.max_nodes
     - elapsed < config.timeout_per_theorem

     a. Pop best node from frontier
     b. Skip if node is terminal or depth >= max_depth
     c. Get the proof state string from the node
     d. Generate candidates:
        policy.generate_candidates(state_pp, config.num_candidates, config.temperature)
     e. Filter to top beam_width by log_prob (they should already be sorted, just take first beam_width)
     f. For each candidate tactic (can be concurrent with tokio::join or sequential):
        - env.apply_tactic(node.state_id, 0, &tactic.text)
        - If Failed → skip
        - If ProofComplete → create terminal node, add to tree. FOUND PROOF!
        - If Success → create child SearchNode
          - If scorer is Some: score = scorer.score(&child_state_pp)
          - Else: score = 0.0
          - Set llm_log_prob = tactic.log_prob
          - Set ebm_score = score
          - Add to tree and frontier
     g. nodes_expanded += 1
     h. Log progress every 50 expansions:
        tracing::debug!(expanded = nodes_expanded, frontier = frontier.len(), depth = node.depth)

  5. Build SearchResult:
     - If a terminal node was found: proved = true, extract proof path
     - Convert all nodes to TrajectoryRecords
     - Label: positive for proof path, negative for everything else
     - Compute remaining_depth for positive nodes

  Handle concurrent tactic application carefully:
  - Option A (simple, recommended for now): apply tactics sequentially in a for loop
  - Option B (faster, for later): use tokio::spawn for each tactic application
    But this requires careful handling of the pool semaphore and error recovery.
  Start with Option A. Add a TODO comment for concurrent optimization.

  For goal_id in apply_tactic: always use 0. Multi-goal states are handled by
  focusing on the first unsolved goal. This is a simplification — full search
  would branch on all goals. Acceptable for v1.

Add a SearchStats struct returned alongside SearchResult for logging:
  - nodes_expanded, nodes_pruned, nodes_terminal
  - avg_tactics_per_expansion
  - avg_lean_time_ms, avg_generate_time_ms
  - peak_frontier_size

Implement the PolicyProvider trait for TacticGenerator and ProofEnvironment for LeanPool
either here or in lib.rs.
```

### Prompt 3.6 — Search crate: lib.rs and trait implementations

```
Wire up crates/search/src/lib.rs:

  pub mod config;
  pub mod engine;
  pub mod node;

  pub use config::SearchConfig;
  pub use engine::{SearchEngine, PolicyProvider, ProofEnvironment, ValueScorer};
  pub use node::{SearchNode, ScoredNode};

Implement the trait bridges connecting search to the existing crates:

1. impl PolicyProvider for policy::TacticGenerator
   - Wrapper that calls self.generate_candidates(proof_state, n)
   - The temperature parameter overrides the config temperature

2. impl ProofEnvironment for lean_repl::LeanPool
   - Direct delegation to pool.start_proof() and pool.run_tactic()

These can go in a separate file like src/adapters.rs or directly in engine.rs.

Also add a MockPolicy and MockEnvironment for unit testing:

  pub struct MockPolicy {
      /// Maps state string → list of (tactic, log_prob) pairs to return
      responses: HashMap<String, Vec<(String, f64)>>,
  }

  pub struct MockEnvironment {
      /// Maps (state_id, tactic) → TacticResult
      responses: HashMap<(u64, String), TacticResult>,
      next_state_id: AtomicU64,
  }

This allows testing the search algorithm without needing Lean or a real LLM.

Add unit tests using mocks:
1. test_search_finds_one_step_proof
   - Mock: root state "⊢ True", tactic "trivial" → ProofComplete
   - Verify: proved = true, proof_tactics = ["trivial"], nodes_expanded = 1

2. test_search_finds_two_step_proof
   - Mock: root "⊢ ∀ n, n = n"
     → "intro n" succeeds → new state "n : Nat ⊢ n = n"
     → "rfl" succeeds → ProofComplete
   - Verify: proved = true, proof_tactics = ["intro n", "rfl"]

3. test_search_respects_node_budget
   - Mock: every tactic creates a new state, nothing ever proves
   - config.max_nodes = 10
   - Verify: proved = false, nodes_expanded = 10

4. test_search_respects_depth_limit
   - Mock: tactics always succeed but never complete the proof
   - config.max_depth = 3
   - Verify: no node in tree has depth > 3

5. test_search_prefers_higher_scoring_nodes
   - Mock: two child nodes with different log_probs
   - Verify the higher-scoring one is expanded first
```

### Prompt 3.7 — prover-core CLI

```
Implement the CLI binary in crates/prover-core/. This wires everything together.

Update Cargo.toml to depend on: lean-repl, policy, search, trajectory, and also:
  clap (workspace), tokio (workspace), tracing-subscriber (workspace),
  serde (workspace), toml (workspace), anyhow (workspace), indicatif (workspace)

src/main.rs — clap CLI with subcommands:

  #[derive(Parser)]
  #[command(name = "burn-qed", about = "Lean 4 theorem prover with EBM value guidance")]
  struct Cli {
      #[command(subcommand)]
      command: Command,
  }

  #[derive(Subcommand)]
  enum Command {
      /// Run proof search on theorems
      Search {
          /// Path to model directory
          #[arg(long)]
          llm_path: PathBuf,

          /// Path to theorem index JSON
          #[arg(long)]
          theorems: PathBuf,

          /// Output trajectory file
          #[arg(long)]
          output: PathBuf,

          /// Path to EBM checkpoint (optional, skip for LLM-only search)
          #[arg(long)]
          ebm_path: Option<PathBuf>,

          /// Override search config file
          #[arg(long, default_value = "configs/search.toml")]
          search_config: PathBuf,

          /// Override number of theorems to search (for testing)
          #[arg(long)]
          limit: Option<usize>,

          /// Disable EBM even if ebm_path is provided
          #[arg(long)]
          no_ebm: bool,

          /// Override temperature
          #[arg(long)]
          temperature: Option<f64>,
      },

      /// Evaluate solve rates at various budgets
      Eval {
          #[arg(long)]
          llm_path: PathBuf,

          #[arg(long)]
          ebm_path: Option<PathBuf>,

          #[arg(long, value_delimiter = ',')]
          budgets: Vec<u32>,

          #[arg(long)]
          theorems: PathBuf,
      },

      // train-ebm will be added in Phase 4
  }

src/config.rs — load configs:

  pub fn load_search_config(path: &Path) -> Result<SearchConfig>
  pub fn load_lean_config(path: &Path) -> Result<LeanPoolConfig>

  Handle missing files gracefully with defaults.

src/pipeline.rs — the search orchestration:

  pub async fn run_search(args: SearchArgs) -> Result<()>
    1. Initialize tracing subscriber (with timestamps, levels)
    2. Load search config
    3. Load theorem index from JSON
    4. Load TacticGenerator
    5. Create LeanPool
    6. Create TrajectoryWriter
    7. Create SearchEngine
    8. For each theorem (with progress bar via indicatif):
       a. Run search_engine.search(theorem, &generator, &pool, scorer)
       b. Label records via TrajectoryWriter::from_search_result()
       c. Write records to trajectory file
       d. Log result: proved/failed, nodes expanded, time
    9. Finish trajectory writer
    10. Print summary:
        - Total theorems: N
        - Proved: X (Y%)
        - Avg nodes per theorem
        - Total wall time
        - Output file path and size

The progress bar should show: [===>   ] 45/100 theorems | 12 proved | 3.2s/theorem

Handle errors per-theorem gracefully — if one theorem panics or errors out, log it and continue to the next. Don't let one failure abort the entire run.
```

### Prompt 3.8 — Theorem index: sample problems for testing

```
Create a small test theorem index for local testing. We need this before we can run the search pipeline end-to-end.

Create data/test_theorems.json with 10 theorems of varying difficulty:

Easy (should be solvable in 1-3 steps):
1. "∀ (n : Nat), n = n" — solved by "intro n; rfl"
2. "True" — solved by "trivial"
3. "∀ (p : Prop), p → p" — solved by "intro p hp; exact hp"
4. "∀ (a b : Nat), a + b = b + a" — solved by "intro a b; omega" or "intro a b; ring"

Medium (may need 3-6 steps):
5. "∀ (n : Nat), 0 + n = n" — needs "intro n; simp" or induction
6. "∀ (n : Nat), n + 0 = n" — needs "intro n; simp" or omega
7. "∀ (p q : Prop), p ∧ q → q ∧ p" — needs intro + cases + constructor + exact

Hard (may not be solvable with limited search):
8. "∀ (n : Nat), n * 1 = n" — may need simp/omega
9. "∀ (l : List Nat), l.reverse.reverse = l" — needs simp [List.reverse_reverse] or similar
10. "∀ (n : Nat), 0 ≤ n" — may need Nat.zero_le or omega

Format:
{
  "theorems": [
    {"name": "nat_refl", "statement": "∀ (n : Nat), n = n"},
    ...
  ]
}

IMPORTANT: These statements must be valid Lean 4 expressions that Pantograph can accept
in goal.start. Test by running Pantograph manually with each one BEFORE committing.
Some might need Mathlib imports — if so, note that in the JSON and handle in the code.

Also create data/minif2f_sample.json — placeholder file with a comment explaining that
the real miniF2F benchmark data will be added when we have LeanDojo tracing set up.
Just put 2-3 known miniF2F problems if you know them, or leave it as an empty list.
```

### Prompt 3.9 — End-to-end integration test

```
Create crates/prover-core/tests/integration.rs — the full pipeline integration test.

This test runs the complete search pipeline on the test theorems. Mark as #[ignore].

test_search_pipeline_small:

1. Load TacticGenerator from MODEL_PATH env var
2. Create LeanPool with 2 workers
3. Load test theorems from data/test_theorems.json
4. Create SearchEngine with a small budget:
   - max_nodes = 50 (not 600 — keep it fast for testing)
   - beam_width = 4
   - num_candidates = 8
   - timeout_per_theorem = 60
5. Create TrajectoryWriter writing to a temp file
6. Run search on all 10 test theorems
7. Finish the writer
8. Read back the Parquet file with TrajectoryReader
9. Print results table:

   Theorem               | Proved | Nodes | Time
   ----------------------|--------|-------|------
   nat_refl              | ✓      |   3   | 1.2s
   true_trivial          | ✓      |   1   | 0.8s
   ...                   | ✗      |  50   | 23.1s

10. Verify:
    - At least 2 of the easy theorems are proved
    - Trajectory file exists and has records
    - Positive labels exist for proved theorems
    - All records for unproved theorems are negative
    - TrajectoryReader::read_summary() counts match

test_trajectory_roundtrip:

1. Create 20 fake TrajectoryRecords with various labels
2. Write to a temp Parquet file
3. Read back
4. Verify field-by-field equality (within floating point tolerance for f64)

test_search_timeout:

1. Use a theorem that's too hard to prove in 5 seconds
2. Set timeout_per_theorem = 5
3. Verify search returns proved = false within ~5-7 seconds (allowing overhead)
4. Verify all trajectory records are labeled Negative

Run:
  MODEL_PATH=./models/deepseek-prover-v2-7b \
  LEAN_ENV_PATH=/path/to/mathlib4/.lake/build \
  cargo test -p prover-core -- --ignored --nocapture
```

### Prompt 3.10 — Progress reporting and robustness

```
Polish the search pipeline for real-world use:

1. Add progress reporting to run_search in pipeline.rs:
   - Use indicatif::ProgressBar with a template showing:
     [{elapsed_precise}] [{bar:40}] {pos}/{len} | proved: {msg}
   - Update the message with the running prove count
   - After each theorem, log a one-line summary:
     tracing::info!(theorem = %name, proved, nodes = expanded, time_ms, "Search complete")

2. Add error resilience:
   - If TacticGenerator fails for a theorem (e.g., tokenization error), log and skip
   - If LeanPool returns ProcessDied, try to recover (recycle worker) and retry once
   - If Parquet write fails, log error but don't abort the whole run
   - Wrap each theorem search in a tokio::time::timeout as a hard safety net

3. Add CTRL-C handling:
   - Use tokio::signal::ctrl_c() to catch interrupts
   - On interrupt: finish writing the trajectory file with whatever data we have
   - Print partial results summary
   - Exit cleanly

4. Add statistics tracking:
   After the full run, print a summary block like:

   ══════════════════════════════════════════
   burn-qed Search Results
   ──────────────────────────────────────────
   Theorems searched:  100
   Proved:             34 (34.0%)
   Failed:             66 (66.0%)
   Total nodes:        24,891
   Avg nodes/theorem:  248.9
   Avg time/theorem:   4.2s
   Total wall time:    7m 02s
   Trajectory file:    trajectories/iter_0.parquet (12.4 MB)
     - Records:        124,455
     - Positive:       1,247
     - Negative:       123,208
   ══════════════════════════════════════════

5. Add a --dry-run flag to the CLI:
   - Load everything (model, pool, theorems) but don't actually search
   - Just verify the setup works and print config summary
   - Useful for checking cloud environment before committing to a long run

6. Run cargo clippy on all modified crates. Fix all warnings.
   Run cargo test --workspace. Everything should pass.
```

### Prompt 3.11 — Update CLAUDE.md

```
Update CLAUDE.md:

1. Mark Phase 2 as complete: [x]
2. Mark Phase 3 as complete: [x]
3. Set "Current Phase" to Phase 4

4. Add "Phase 3 Results" section:
   - How many of the 10 test theorems were proved
   - Average nodes per theorem
   - Average time per theorem (on local hardware)
   - Trajectory file size for 10 theorems
   - Any issues encountered and workarounds

5. Add to "Cross-Crate Integration Notes":
   - How to run search: cargo run -p prover-core -- search --llm-path ... --theorems ... --output ...
   - TrajectoryRecord schema description
   - How to read trajectories: TrajectoryReader::read_all()
   - The label assignment logic (positive = on proof path, negative = everything else)
   - How the Parquet files feed into Phase 4 EBM training

6. Note any performance observations:
   - What's the bottleneck? (LLM generation? Lean tactic checking? Both?)
   - Does it match the latency budget from the plan?
   - Any adjustments needed for cloud deployment?
```

---

## Verification Checklist

```bash
# All crates compile
cargo check --workspace

# Unit tests pass (no external deps needed)
cargo test -p trajectory
cargo test -p search

# No clippy warnings
cargo clippy --workspace

# Integration tests (need model + Lean)
MODEL_PATH=./models/deepseek-prover-v2-7b \
LEAN_ENV_PATH=/path/to/mathlib4/.lake/build \
  cargo test -p prover-core -- --ignored --nocapture

# THE BIG TEST — full search pipeline on test theorems
MODEL_PATH=./models/deepseek-prover-v2-7b \
LEAN_ENV_PATH=/path/to/mathlib4/.lake/build \
  cargo run -p prover-core -- search \
    --llm-path ./models/deepseek-prover-v2-7b \
    --theorems data/test_theorems.json \
    --output /tmp/test_trajectories.parquet \
    --limit 5

# Verify the output
cargo run -p prover-core -- eval \  # or a quick reader script
  # Should print trajectory summary
```

### Success Criteria

1. **Search runs** end-to-end on 10 test theorems without crashes
2. **At least 2-3 easy theorems proved** (nat_refl, True, identity)
3. **Trajectory Parquet files** are correct and readable
4. **Label assignment** works: proved theorems have positive labels on proof path
5. **Progress reporting** shows running status during search
6. **Error handling** survives individual theorem failures without aborting

---

## Troubleshooting

### "No theorems proved at all"
- Check the prompt format from Phase 2 — is the LLM generating plausible Lean tactics?
- Try temperature=0 (greedy) to see the best single tactic per state
- Manually inspect: generate 32 candidates for "⊢ True" — does any say "trivial"?
- Check Lean error messages — are the tactics syntactically valid but semantically wrong?

### Search is very slow (>30s per theorem with budget 50)
- Profile where time is spent: generation vs Lean application
- If generation dominates: expected on CPU. Use a small num_candidates (4-8) for local testing
- If Lean dominates: check worker count, check for workers stuck in recycling loops

### Parquet file is empty
- Check that TrajectoryWriter::finish() is called (not just dropped)
- Check that from_search_result() produces records even for failed proofs
- Verify the Arrow schema column types match what you're writing

### "state_id mismatch" or "unknown state"
- Pantograph state IDs are per-session. If a worker was recycled between starting a proof and applying a tactic, the state IDs are invalidated.
- Fix: run the entire proof search on a single worker (acquire once, search, release).
  Alternatively, ensure start_proof and all subsequent apply_tactic calls use the same worker.
- This is a real design issue — you may need a ProofSession that pins to one worker.

### Priority queue seems broken (wrong order)
- Verify ScoredNode ordering: BinaryHeap is a max-heap in Rust, so highest score should pop first.
- If using OrderedFloat, verify the Ord impl gives the right direction.
- Print the top-5 frontier at each expansion to debug ordering issues.

### Out of memory during long runs
- Worker recycling should handle Lean memory leaks
- For the Rust process itself: check if you're accumulating nodes in the tree without bound.
  With max_nodes=600, the tree is at most ~5000 nodes (600 expansions × 8 children) — fine.
- If reading trajectory files back is OOM: use streaming reads instead of loading all into memory.
