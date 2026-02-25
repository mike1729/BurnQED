# Phase 5: Expert Iteration + Evaluation — Claude Code Instructions

Step-by-step prompts for `claude` CLI. Phases 0-4 are complete.

Phase 5 is different from previous phases. It's not building a new Rust crate — it's:
1. Writing the Python LLM training scripts
2. Implementing the evaluation harness
3. Building the orchestration scripts that tie everything together
4. Actually running 5 expert iterations and benchmarking

This is where you shift from MacBook development to cloud GPU usage. The plan budgets 16 days: ~4 days of coding, ~12 days of waiting for GPU runs.

## Prerequisites

- All Rust code from Phases 0-4 working:
  - `cargo run -p prover-core -- search` produces trajectory Parquet files
  - `cargo run -p prover-core -- train-ebm` trains the EBM from trajectories
  - Search with `--ebm-path` uses EBM scores
- DeepSeek-Prover-V2-7B weights downloaded
- Python environment with: `torch`, `transformers`, `peft`, `datasets`, `safetensors`, `accelerate`
- Cloud GPU access set up (Lambda Labs, RunPod, or Vast.ai)
- Mathlib traced via LeanDojo (or you have a theorem index ready)

## Phase 5 Overview: The Expert Iteration Loop

```
For iteration i = 0, 1, 2, 3, 4:

  ┌──────────────────────────────────────┐
  │ 1. LLM Fine-tuning (Python, GPU)     │
  │    Input: tactic pairs + trajectories │
  │    Output: safetensors checkpoint     │
  ├──────────────────────────────────────┤
  │ 2. EBM Training (Rust/burn, GPU)      │
  │    Input: trajectory Parquet files    │
  │    Output: burn-rs checkpoint         │  ← skip for iter 0
  ├──────────────────────────────────────┤
  │ 3. Search / Trajectory Collection     │
  │    Input: LLM + EBM checkpoints       │  (Rust, multi-GPU)
  │    Output: trajectories/iter_i.parquet│
  ├──────────────────────────────────────┤
  │ 4. Evaluation (Rust, 1 GPU)           │
  │    Input: LLM + EBM + miniF2F        │
  │    Output: solve rates at budgets     │
  └──────────────────────────────────────┘
```

Iteration 0 is special: no EBM yet, base model only, noise injection for diverse negatives.

---

## Prompt 5.1 — Mathlib data preparation (Python)

```
Implement python/data/trace_mathlib.py — extracts theorem statements and tactic pairs from Mathlib for LLM training.

This uses LeanDojo (https://github.com/lean-dojo/LeanDojo) to trace Mathlib and extract
tactic-level training data.

The script should:

1. Install/import LeanDojo:
   pip install lean-dojo

2. Trace Mathlib4 (or load a cached trace):
   from lean_dojo import LeanGitRepo, trace
   repo = LeanGitRepo("https://github.com/leanprover-community/mathlib4", "COMMIT_HASH")
   traced_repo = trace(repo)  # This takes 30-60 minutes the first time

3. Extract theorem statements for the theorem index:
   - For each traced file, for each theorem:
     - name: fully qualified name (e.g., "Nat.add_comm")
     - statement: the type/proposition
     - file_path: source file
   - Write to data/theorem_index.json as TheoremIndex format
   - Filter: only include theorems with tactic proofs (skip term-mode proofs)
   - Target: ~75,000 theorems from Mathlib

4. Extract tactic pairs for LLM training:
   - For each tactic step in each proof:
     - state_before: pretty-printed proof state before the tactic
     - tactic: the tactic that was applied
     - state_after: proof state after
   - Write to data/tactic_pairs/train.jsonl (one JSON object per line)
   - Format: {"state": "...", "tactic": "...", "theorem": "...", "depth": N}
   - Also create a validation split: data/tactic_pairs/val.jsonl (5% of theorems held out)

5. Extract miniF2F test set:
   - LeanDojo provides miniF2F-test and miniF2F-valid
   - Write to data/minif2f_test.json and data/minif2f_valid.json
   - Same TheoremIndex format

6. Print summary:
   - Total theorems traced: N
   - Total tactic pairs: N
   - Train/val split: N/N
   - miniF2F test: N problems
   - Output files and sizes

Handle the common failure modes:
- LeanDojo trace can fail on specific files. Catch and skip those, log them.
- Some theorems have very long proofs (>100 tactics). Include them but note the count.
- Mathlib version pinning: specify the exact commit hash for reproducibility.

If LeanDojo is too slow or broken, provide a fallback approach:
- Download a pre-traced Mathlib dataset from LeanDojo's releases page
- Or use the Lean4 `#check` and `#print` commands to extract theorem statements
  (less data but faster to get started)
```

### Prompt 5.2 — LLM fine-tuning script (Python)

```
Implement python/training/train_llm.py — LoRA fine-tuning of DeepSeek-Prover-V2-7B.

This uses HuggingFace PEFT (Parameter-Efficient Fine-Tuning) with LoRA adapters.

Arguments:
  --model-name: base model (default: "deepseek-ai/DeepSeek-Prover-V2-7B")
  --data: path to tactic pairs JSONL (data/tactic_pairs/train.jsonl)
  --val-data: validation JSONL (data/tactic_pairs/val.jsonl)
  --extra-data: optional glob for trajectory Parquet files (for iterations > 0)
  --output: checkpoint output directory
  --base: optional base checkpoint to continue from (for iterations > 0)
  --epochs: number of training epochs (default 3 for iter 0, 1 for subsequent)
  --lr: learning rate (default 2e-4, halved each iteration)
  --batch-size: per-device batch size (default 4)
  --gradient-accumulation: gradient accumulation steps (default 8, effective batch = 32)
  --lora-r: LoRA rank (default 16)
  --lora-alpha: LoRA alpha (default 32)
  --max-seq-len: max sequence length (default 2048)

Training data format:
  Each example is a (state, tactic) pair formatted as:
    Input:  "[GOAL]\n{state}\n[TACTIC]\n"
    Target: "{tactic}"

  Check DeepSeek-Prover-V2's actual prompt template from its README or paper.
  They may use a specific chat template or marker tokens. Match whatever the
  Phase 2 TacticGenerator.build_prompt() uses for consistency.

For iterations > 0, --extra-data adds successful proof tactics from trajectories:
  - Read Parquet trajectory files
  - Filter to positive (on-proof-path) records
  - Extract (state_pp, tactic_applied) pairs
  - Combine with the base training data
  - This is how the LLM improves: it learns from its own successful proofs

Implementation:

1. Load base model with quantization for training:
   from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
   bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
   model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

2. Apply LoRA:
   from peft import LoraConfig, get_peft_model
   lora_config = LoraConfig(
       r=lora_r, lora_alpha=lora_alpha,
       target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
       lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
   )
   model = get_peft_model(model, lora_config)

3. If --base is provided, load the LoRA weights from the previous iteration

4. Train using HuggingFace Trainer or a manual training loop with accelerate

5. Save:
   - LoRA adapter weights (small, ~100MB)
   - Full merged model (for convenience, but large)

Provide logging: loss curves, learning rate, epoch progress.

The script should work on 1-8 GPUs via accelerate/DeepSpeed:
  accelerate launch python/training/train_llm.py --data ... --output ...
```

### Prompt 5.3 — LLM export for candle (Python)

```
Implement python/training/export_llm.py — merges LoRA weights and exports for candle.

Arguments:
  --checkpoint: path to the LoRA checkpoint from train_llm.py
  --base-model: base model name (default: "deepseek-ai/DeepSeek-Prover-V2-7B")
  --output: output directory for merged safetensors

Steps:

1. Load base model (full precision, CPU is fine for merging)
2. Load LoRA adapter from checkpoint
3. Merge LoRA into base weights: model = model.merge_and_unload()
4. Save merged model in safetensors format:
   model.save_pretrained(output, safe_serialization=True)
5. Copy tokenizer files to output directory
6. Verify: load the merged model and generate one tactic to sanity-check

The output should be a directory that TacticGenerator::load() can consume:
  models/llm/iter_0/
  ├── config.json
  ├── model-00001-of-000002.safetensors
  ├── model-00002-of-000002.safetensors
  ├── model.safetensors.index.json
  ├── tokenizer.json
  └── tokenizer_config.json

Print the output size and verify the safetensors can be loaded:
  from safetensors import safe_open
  with safe_open("model-00001-of-000002.safetensors", framework="pt") as f:
      print(f"Keys: {len(f.keys())}")
```

### Prompt 5.4 — Evaluation harness (Rust)

```
Implement the eval subcommand in prover-core that benchmarks solve rates.

Update crates/prover-core/src/pipeline.rs with run_eval():

The eval command runs search on a benchmark set (miniF2F-test) at multiple node budgets
and reports solve rates.

Arguments (already in main.rs from Phase 3):
  --llm-path, --ebm-path (optional), --theorems, --budgets (comma-separated)

Implementation:

1. Load the model(s) and Lean pool

2. For each budget in --budgets (e.g., 100, 300, 600):
   a. Set config.max_nodes = budget
   b. For each theorem in --theorems:
      - Run search
      - Record: proved (bool), nodes used, time
   c. Compute stats:
      - Solve rate: proved / total
      - Mean nodes for proved theorems
      - Mean time per theorem
      - Median time per theorem

3. Output a results table:

   ┌─────────────────────────────────────────────────────────┐
   │ burn-qed Evaluation Results                              │
   │ Model: models/llm/iter_0                                │
   │ EBM: models/ebm/iter_0                                  │
   │ Benchmark: miniF2F-test (244 problems)                  │
   ├──────────┬───────────┬──────────┬───────────┬───────────┤
   │ Budget   │ Solved    │ Rate     │ Avg Nodes │ Avg Time  │
   ├──────────┼───────────┼──────────┼───────────┼───────────┤
   │ 100      │ 62/244    │ 25.4%    │ 34.2      │ 12.3s     │
   │ 300      │ 89/244    │ 36.5%    │ 102.7     │ 38.1s     │
   │ 600      │ 107/244   │ 43.9%    │ 198.3     │ 72.4s     │
   └──────────┴───────────┴──────────┴───────────┴───────────┘

4. Also output machine-readable JSON results:
   {
     "model": "models/llm/iter_0",
     "ebm": "models/ebm/iter_0",
     "benchmark": "miniF2F-test",
     "results": [
       {"budget": 100, "solved": 62, "total": 244, "rate": 0.254, ...},
       ...
     ],
     "per_theorem": [
       {"name": "...", "proved_at_budget": 100, "nodes_used": 34, ...},
       ...
     ]
   }
   Save to eval_results/iter_0.json

5. Compare mode: if previous iteration results exist, show delta:
   Budget 600: 43.9% (+5.2% vs iter 0)

6. Cumulative solve rate: a theorem is "solved" if it was proved in ANY budget.
   Report this at the bottom:
   "Cumulative (any budget): 112/244 (45.9%)"

7. Pass@N mode: for each theorem, run search N times (e.g., N=8) with different seeds.
   A theorem is "solved" if ANY of the N attempts succeeds.
   This is the standard miniF2F evaluation protocol.
   Add --pass-n flag (default 1 for fast eval, 8 or 32 for final benchmark).
```

### Prompt 5.5 — Orchestration script: run_iteration.sh

```
Implement scripts/run_iteration.sh — the master script that runs one expert iteration.

Read docs/spindle_final_plan.md Section 9 for the reference script.

#!/bin/bash
set -euo pipefail

ITER=${1:?Usage: ./scripts/run_iteration.sh <iteration_number>}
PREV=$((ITER - 1))
THEOREM_INDEX="data/theorem_index.json"
MINIF2F="data/minif2f_test.json"

echo "════════════════════════════════════════"
echo "  burn-qed Expert Iteration ${ITER}"
echo "════════════════════════════════════════"

# Configurable paths
LLM_BASE="deepseek-ai/DeepSeek-Prover-V2-7B"
LLM_DIR="models/llm/iter_${ITER}"
EBM_DIR="models/ebm/iter_${ITER}"
TRAJ_DIR="trajectories"
EVAL_DIR="eval_results"

mkdir -p "$LLM_DIR" "$EBM_DIR" "$TRAJ_DIR" "$EVAL_DIR"

# ── Step 1: LLM Fine-tuning ──

if [ "$ITER" -eq 0 ]; then
    echo "── [1/5] Fine-tuning LLM (iteration 0, from base) ──"
    accelerate launch python/training/train_llm.py \
        --model-name "$LLM_BASE" \
        --data data/tactic_pairs/train.jsonl \
        --val-data data/tactic_pairs/val.jsonl \
        --output "checkpoints/llm/iter_0" \
        --epochs 3 --lr 2e-4
else
    LR=$(python3 -c "print(f'{2e-4 * 0.5 ** $ITER:.2e}')")
    echo "── [1/5] Fine-tuning LLM (iteration ${ITER}, lr=${LR}) ──"
    accelerate launch python/training/train_llm.py \
        --model-name "$LLM_BASE" \
        --data data/tactic_pairs/train.jsonl \
        --extra-data "${TRAJ_DIR}/iter_*.parquet" \
        --output "checkpoints/llm/iter_${ITER}" \
        --base "checkpoints/llm/iter_${PREV}" \
        --epochs 1 --lr "$LR"
fi

echo "── [1b/5] Exporting LLM to safetensors ──"
python python/training/export_llm.py \
    --checkpoint "checkpoints/llm/iter_${ITER}" \
    --output "$LLM_DIR"

# ── Step 2: EBM Training ──

if [ "$ITER" -eq 0 ]; then
    echo "── [2/5] Skipping EBM training (no trajectories yet for iter 0) ──"
else
    echo "── [2/5] Training EBM (burn-rs) ──"
    RESUME_FLAG=""
    if [ "$PREV" -ge 0 ] && [ -d "models/ebm/iter_${PREV}" ]; then
        RESUME_FLAG="--resume-from models/ebm/iter_${PREV}"
    fi
    cargo run --release -p prover-core -- train-ebm \
        --trajectories $(ls ${TRAJ_DIR}/iter_*.parquet | tr '\n' ',') \
        --llm-path "$LLM_DIR" \
        --output "$EBM_DIR" \
        --steps 50000 \
        $RESUME_FLAG
fi

# ── Step 3: Search / Trajectory Collection ──

echo "── [3/5] Running proof search ──"
EBM_FLAG=""
if [ "$ITER" -gt 0 ] && [ -d "$EBM_DIR" ]; then
    EBM_FLAG="--ebm-path $EBM_DIR"
fi

cargo run --release -p prover-core -- search \
    --llm-path "$LLM_DIR" \
    $EBM_FLAG \
    --theorems "$THEOREM_INDEX" \
    --output "${TRAJ_DIR}/iter_${ITER}.parquet"

# Noise injection for iteration 0
if [ "$ITER" -eq 0 ]; then
    echo "── [3b/5] Noise injection (high temperature) ──"
    cargo run --release -p prover-core -- search \
        --llm-path "$LLM_DIR" \
        --no-ebm \
        --temperature 1.2 \
        --theorems "$THEOREM_INDEX" \
        --output "${TRAJ_DIR}/iter_0_noisy.parquet"
fi

# ── Step 4: Evaluation ──

echo "── [4/5] Evaluating on miniF2F ──"
cargo run --release -p prover-core -- eval \
    --llm-path "$LLM_DIR" \
    $EBM_FLAG \
    --theorems "$MINIF2F" \
    --budgets 100,300,600

# ── Step 5: Summary ──

echo "── [5/5] Iteration ${ITER} Summary ──"
echo ""
echo "LLM checkpoint: $LLM_DIR"
echo "EBM checkpoint: $EBM_DIR"
echo "Trajectories: ${TRAJ_DIR}/iter_${ITER}.parquet"
echo ""
ls -lh "${TRAJ_DIR}/iter_${ITER}"*.parquet 2>/dev/null || true
echo ""
cat "${EVAL_DIR}/iter_${ITER}.json" 2>/dev/null | python3 -m json.tool || true

echo ""
echo "════════════════════════════════════════"
echo "  Iteration ${ITER} complete"
echo "════════════════════════════════════════"

Make the script executable: chmod +x scripts/run_iteration.sh

Also create scripts/run_all_iterations.sh:
  #!/bin/bash
  set -euo pipefail
  for i in 0 1 2 3 4; do
    echo "Starting iteration $i at $(date)"
    ./scripts/run_iteration.sh $i 2>&1 | tee "logs/iter_${i}.log"
    echo "Finished iteration $i at $(date)"
  done

And scripts/setup_cloud.sh — environment setup for a fresh cloud GPU instance:
  - Install Rust via rustup
  - Install Lean 4 via elan
  - Clone and build Mathlib
  - Build Pantograph
  - Install Python dependencies (torch, transformers, peft, etc.)
  - Clone the burn-qed repo
  - Build in release mode: cargo build --release -p prover-core
  - Download model weights if not present
  - Run a quick smoke test: search 2 easy theorems
```

### Prompt 5.6 — Spot instance resilience

```
Add checkpointing and resume logic to handle cloud spot instance preemption.

1. Update the search command in prover-core to support resume:

   Add --resume-from flag to the Search subcommand:
   - If provided, read the partial trajectory file
   - Determine which theorems have already been searched
   - Skip those and continue from where we left off
   - Append new results to the same file (or a new shard)

   Implementation:
   - Read existing Parquet file → extract set of theorem_names already searched
   - Filter theorem_index to exclude already-done theorems
   - Proceed with remaining theorems
   - On completion, merge old + new records

2. Add periodic auto-save during search:
   - Every 100 theorems, flush the trajectory writer to disk
   - This means if the instance is killed, we lose at most 100 theorems of work
   - Log: "Checkpoint saved: 2345/75000 theorems searched"

3. Add a signal handler for SIGTERM (what cloud providers send before preemption):
   - Catch SIGTERM via tokio::signal
   - Flush trajectory writer
   - Save partial results
   - Exit cleanly

4. Update train-ebm to support partial trajectory files:
   - If a Parquet file is incomplete (truncated), read as much as possible
   - Log a warning about the partial file

5. Create scripts/resume_search.sh — convenience wrapper:
   #!/bin/bash
   ITER=${1:?}
   PARTIAL="${TRAJ_DIR}/iter_${ITER}.parquet"
   if [ -f "$PARTIAL" ]; then
       echo "Resuming from partial trajectory: $PARTIAL"
       cargo run --release -p prover-core -- search \
           --resume-from "$PARTIAL" \
           --llm-path "models/llm/iter_${ITER}" \
           --theorems data/theorem_index.json \
           --output "$PARTIAL"
   else
       echo "No partial file found, starting fresh"
       ./scripts/run_iteration.sh "$ITER"
   fi
```

### Prompt 5.7 — Lean start: iteration 0 on small data

```
Before running the full 75K theorem pipeline, validate the entire loop on a small dataset.

Create scripts/lean_start.sh — runs iteration 0 on just 500 theorems to verify everything works end-to-end:

  #!/bin/bash
  set -euo pipefail
  echo "=== LEAN START: Iteration 0 on 500 theorems ==="

  # Use a subset of theorems
  python3 -c "
  import json
  with open('data/theorem_index.json') as f:
      data = json.load(f)
  data['theorems'] = data['theorems'][:500]
  with open('data/theorem_index_500.json', 'w') as f:
      json.dump(data, f)
  print(f'Created subset with {len(data[\"theorems\"])} theorems')
  "

  # Step 1: Train LLM (same as full, data is from all of Mathlib)
  # ... (or skip if already done and reuse the checkpoint)

  # Step 2: Search on 500 theorems, LLM only
  cargo run --release -p prover-core -- search \
      --llm-path models/llm/iter_0 \
      --no-ebm \
      --theorems data/theorem_index_500.json \
      --output trajectories/lean_start_iter0.parquet

  # Step 3: Train EBM on the small trajectory
  cargo run --release -p prover-core -- train-ebm \
      --trajectories trajectories/lean_start_iter0.parquet \
      --llm-path models/llm/iter_0 \
      --output models/ebm/lean_start_0 \
      --steps 5000

  # Step 4: Search WITH EBM on same 500 theorems
  cargo run --release -p prover-core -- search \
      --llm-path models/llm/iter_0 \
      --ebm-path models/ebm/lean_start_0 \
      --theorems data/theorem_index_500.json \
      --output trajectories/lean_start_iter0_ebm.parquet

  # Step 5: Compare
  echo ""
  echo "=== Comparison ==="
  echo "Without EBM:"
  # Parse and print solve rate from trajectories/lean_start_iter0.parquet
  echo "With EBM:"
  # Parse and print solve rate from trajectories/lean_start_iter0_ebm.parquet

This is the most important validation: does the EBM improve solve rate on the same theorems?
Even a 1-2% improvement on 500 theorems is a positive signal.

If the EBM HURTS performance, debug:
- Check EBMMetrics from training: is the energy gap positive?
- Check rank_accuracy: is it above 60%?
- Try alpha=0.8, beta=0.2 (less EBM influence) to see if it helps
- Inspect: for a theorem the EBM made worse, print the EBM scores for each child state
  and compare to the LLM log-probs. Is the EBM overriding correct LLM predictions?
```

### Prompt 5.8 — Results tracking and comparison

```
Create a results tracking system that persists across iterations.

1. Create crates/prover-core/src/results.rs:

   IterationResult struct (Serialize, Deserialize):
     - iteration: u32
     - timestamp: String
     - llm_path: String
     - ebm_path: Option<String>
     - benchmark: String
     - budget_results: Vec<BudgetResult>
     - training_metrics: Option<TrainingMetrics>

   BudgetResult:
     - budget: u32
     - solved: u32
     - total: u32
     - rate: f64
     - avg_nodes: f64
     - avg_time_secs: f64
     - per_theorem: Vec<TheoremResult>

   TheoremResult:
     - name: String
     - proved: bool
     - budget_at_proof: Option<u32>  — smallest budget that solved it
     - nodes_used: u32
     - time_secs: f64

   TrainingMetrics:
     - final_loss: f64
     - final_energy_gap: f64
     - final_rank_accuracy: f64
     - training_steps: usize
     - training_time_secs: f64

2. Create eval_results/summary.json — cumulative results across all iterations:
   {
     "iterations": [
       {
         "iteration": 0,
         "miniF2F_600": { "solved": 107, "total": 244, "rate": 0.439 },
         "training_metrics": null
       },
       {
         "iteration": 1,
         "miniF2F_600": { "solved": 118, "total": 244, "rate": 0.484 },
         "delta_vs_prev": "+4.5%",
         "training_metrics": { "energy_gap": 1.23, "rank_accuracy": 0.78 }
       },
       ...
     ]
   }

3. Add a compare subcommand to the CLI:

   Compare {
       #[arg(long)]
       results: Vec<PathBuf>,  // e.g., eval_results/iter_0.json eval_results/iter_1.json
   }

   Output:
     Iteration  │ Budget 100 │ Budget 300 │ Budget 600 │ Cumulative
     ───────────┼────────────┼────────────┼────────────┼───────────
     0 (LLM)    │ 25.4%      │ 36.5%      │ 43.9%      │ 45.9%
     1 (+ EBM)  │ 28.7%      │ 40.1%      │ 48.4%      │ 50.2%
     2           │ 30.2%      │ 42.8%      │ 51.1%      │ 53.3%
     ───────────┼────────────┼────────────┼────────────┼───────────
     Δ (0→2)    │ +4.8%      │ +6.3%      │ +7.2%      │ +7.4%

4. Track newly-solved theorems:
   For each iteration, report which theorems were solved for the first time.
   These are the most interesting to inspect manually.
```

### Prompt 5.9 — Cloud deployment guide

```
Create docs/cloud_deployment.md — a comprehensive guide for running on cloud GPUs.

Cover:

1. Provider recommendations:
   - Lambda Labs: best A100 spot prices (~$1.29/hr), good for search/training
   - RunPod: easy setup, community templates, ~$1.64/hr A100
   - Vast.ai: cheapest, least reliable, ~$1.10/hr A100
   - For LLM fine-tuning: need multi-GPU (4-8× A100). Lambda Labs or RunPod pods.
   - For search: single A100 80GB is enough. Lean workers run on CPU.
   - For EBM training: single A100. Very fast (~6 hours for 50K steps).

2. Instance setup checklist:
   - Ubuntu 22.04 or 24.04
   - CUDA 12.x + cuDNN
   - Rust toolchain
   - Lean 4 + elan
   - Mathlib (built, ~2GB cache)
   - Pantograph
   - Python 3.11+ with torch, transformers, peft
   - burn-qed repo (cargo build --release)
   - Model weights in models/

3. Recommended workflow:
   - Create a Docker image or snapshot with everything pre-installed
   - Use a persistent volume for model weights and checkpoints
   - Start spot instances for compute, attach the volume
   - After each step, push results to persistent storage (S3, GCS, or volume)

4. Cost tracking:
   - Estimate hours for each step
   - Track actual spend
   - Set billing alerts

5. Monitoring:
   - Use tmux/screen for long-running tasks
   - Tail logs: tail -f logs/iter_0.log
   - GPU utilization: nvidia-smi -l 1
   - For search: watch the progress bar output

6. Data transfer:
   - Trajectory files are large (100MB-1GB per iteration)
   - Use rsync or SCP to transfer between instances
   - Or use shared storage (NFS, S3 FUSE mount)
```

### Prompt 5.10 — Run the iterations and update CLAUDE.md

```
This is not a coding prompt — it's a checklist for actually running the experiments.

1. Run the lean start (Prompt 5.7):
   ./scripts/lean_start.sh
   Record: solve rates with/without EBM on 500 theorems.

2. If lean start shows improvement, proceed to full iteration 0:
   ./scripts/run_iteration.sh 0

3. Run iterations 1-4:
   ./scripts/run_iteration.sh 1
   ./scripts/run_iteration.sh 2
   ./scripts/run_iteration.sh 3
   ./scripts/run_iteration.sh 4

4. After each iteration, run:
   cargo run -p prover-core -- compare \
     --results eval_results/iter_0.json eval_results/iter_1.json ...

5. Update CLAUDE.md with final results:
   - Mark Phases 0-5 as complete
   - Set "Current Phase" to Phase 6 (burn-rs PRs)
   - Add "Phase 5 Results" section:
     - Solve rates per iteration per budget
     - Delta improvement per iteration
     - Total newly-solved theorems
     - Training metrics trends (energy gap, rank accuracy)
     - Total GPU cost
     - Wall time
     - Comparison to published baselines

6. Write a brief results document: docs/results.md
   - Table of solve rates across iterations
   - Graph description (or actual graph if you generate one):
     X-axis = iteration, Y-axis = solve rate at budget 600
   - Analysis: did the EBM help? How much? When did improvements plateau?
   - Which kinds of theorems benefited most from EBM guidance?
   - Comparison to DeepSeek-Prover-V2's published results (they report 88.9% at 671B — 
     we won't match that with 7B, but how do we compare to their 7B baseline?)
```

---

## Verification Checklist

```bash
# Python scripts run
python python/data/trace_mathlib.py --help
python python/training/train_llm.py --help
python python/training/export_llm.py --help

# Evaluation harness works on test theorems
MODEL_PATH=./models/deepseek-prover-v2-7b \
LEAN_ENV_PATH=/path/to/mathlib4/.lake/build \
  cargo run -p prover-core -- eval \
    --llm-path ./models/deepseek-prover-v2-7b \
    --theorems data/test_theorems.json \
    --budgets 50,100

# Orchestration script is valid bash
bash -n scripts/run_iteration.sh
bash -n scripts/run_all_iterations.sh

# Lean start completes on 500 theorems (GPU needed)
./scripts/lean_start.sh

# Full iteration 0 completes
./scripts/run_iteration.sh 0
```

### Success Criteria

1. **Data pipeline** works: Mathlib traced, tactic pairs extracted, theorem index created
2. **LLM fine-tuning** completes without errors, validation loss decreases
3. **Exported model** loads in candle and generates reasonable tactics
4. **Evaluation harness** reports solve rates at multiple budgets
5. **Lean start** shows whether EBM improves over LLM-only (even by 1%)
6. **Full iteration** pipeline runs end-to-end: LLM train → search → EBM train → eval
7. **5 iterations** complete with tracked results
8. **Solve rate trend** is upward (each iteration better than or equal to previous)

### Expected Results (Realistic)

Based on published work with similar 7B models:
- Iteration 0 (LLM only, budget 600): ~30-40% on miniF2F-test
- Iteration 1 (+ EBM): ~35-45%
- Iteration 4 (final): ~40-55%
- Published DeepSeek-Prover-V2-7B baseline: ~65% (they use different search)
- Published DeepSeek-Prover-V2-671B: 88.9%

If results are below 30% at iteration 0, the LLM fine-tuning or prompt format may be wrong.
If the EBM doesn't improve results after 2 iterations, check the decision tree in the plan.

---

## Troubleshooting

### LeanDojo trace fails
- Mathlib updates frequently. Pin to a specific commit.
- Some files may fail to trace. Skip and log.
- Fallback: download pre-traced data from LeanDojo's GitHub releases.

### LLM fine-tuning OOM
- Reduce batch_size, increase gradient_accumulation
- Use 4-bit quantization (QLoRA) via BitsAndBytes
- Use DeepSpeed ZeRO stage 2 or 3

### Export produces different results than training
- LoRA merge must happen in float32, not quantized
- Load base model in float32 for merge, then save
- Verify: generate with merged model in Python, compare to training outputs

### EBM doesn't improve solve rate
- Check that training converged: energy_gap > 0, rank_accuracy > 0.6
- Try weaker EBM influence: alpha=0.8, beta=0.2
- Check negative mining: too many easy negatives = useless training
- Inspect EBM predictions on specific theorems manually
- Possible root cause: the encoder (encode_only) doesn't differentiate proof states well enough

### Search is too slow on GPU
- Profile: is it LLM inference or Lean REPL?
- LLM: use int8 quantization in candle if available, or reduce num_candidates to 16
- Lean: increase worker count (64 on a machine with 32+ cores)
- Budget: reduce max_nodes to 300 for faster iterations while debugging

### Spot instance preempted mid-run
- Use --resume-from to continue search from the partial trajectory file
- Checkpoints should be saved frequently enough that you lose < 30 min of work
- For LLM training: HuggingFace Trainer auto-saves checkpoints, resume with --resume_from_checkpoint
