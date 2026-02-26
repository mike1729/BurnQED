# burn-qed v2: Data Format Specification

## The Prompt Format (SETTLED)

Our Rust `policy` crate already has a working format that produced 41.7% miniF2F through 5 iterations. The SFT training data **must match this format exactly.**

### The format: DeepSeek-native with tactic state as Lean comment

```
Complete the following Lean 4 code:

```lean4
/- tactic state:
n : ℕ
h : n > 0
⊢ n * n ≥ n
-/
```
```

Note the code fence is **closed** — the model generates after ` ``` `. 

Model generates: `exact Nat.le_mul_of_pos_left n h`

The `extract_first_tactic()` function in Rust handles output parsing: strips code fences if the model wraps output in `` ```lean4 ... ``` ``, skips comments and declarations, extracts the bare tactic.

**Why this format is correct:**
- Uses DeepSeek-Prover-V2's native instruction prefix ("Complete the following Lean 4 code:") — zero LoRA capacity wasted learning a new protocol
- The `/- ... -/` block is standard Lean 4 comment syntax — the model already understands it from pretraining
- The `` ```lean4 ... ``` `` code fence matches DeepSeek's training distribution
- No special tokens needed (`[GOAL]`, `[PROOFSTEP]` are NOT used) — eliminates Gotcha #14 entirely

### Completion-only loss masking

For SFT training, the loss mask boundary is the closing code fence ` ``` `:

```
Complete the following Lean 4 code:\n\n```lean4\n/- tactic state:\n{state}\n-/\n```\n  ← MASK EVERYTHING UP TO HERE
exact Nat.le_mul_of_pos_left n h  ← TRAIN ON THIS
```

With `DataCollatorForCompletionOnlyLM`, use `response_template="` ``` `\n"` (the closing code fence + newline). Everything before and including this marker is the prompt; everything after is the completion the model learns to generate.

---

## Raw Dataset Formats

### 1. Goedel Workbook Proofs (29.7K proved theorems)
**Source:** `Goedel-LM/Lean-workbook-proofs`
**Format:** Parquet with complete Lean 4 proof code

```python
from datasets import load_dataset
ds = load_dataset("Goedel-LM/Lean-workbook-proofs")

# Each row is a complete theorem + proof
example = ds["train"][0]
# example contains full Lean 4 code like:
"""
import Mathlib
import Aesop

theorem lean_workbook_12345 (n : ℕ) (h : n > 0) : n * n ≥ n := by
  exact Nat.le_mul_of_pos_left n h
"""
```

**What you need to do:** These are WHOLE PROOFS, not (state, tactic) pairs. You must trace them with LeanDojo to extract tactic-level pairs. This is what Task 0.4 does.

**After tracing (LeanDojo output):**
```python
# For each traced theorem, you get:
traced.tactic_pairs = [
    (state_0, tactic_0),  # state_0.pp = "n : ℕ\nh : n > 0\n⊢ n * n ≥ n"
    (state_1, tactic_1),  # if proof has multiple steps
    ...
]
# Each state has: .pp (pretty-printed string), .depth, .goals
# Each tactic has: .text (the tactic string)
```

### 2. NuminaMath-LEAN (104K formalized problems)
**Source:** `AI-MO/NuminaMath-LEAN`
**Format:** Parquet

```python
ds = load_dataset("AI-MO/NuminaMath-LEAN")
row = ds["train"][0]

row.keys()
# uuid, problem, question_type, answer, author, formal_statement,
# formal_ground_truth, ground_truth_type, formal_proof, rl_data,
# source, problem_type, exam

# formal_statement: the Lean 4 theorem statement (always present)
# formal_proof: machine-generated proof (from Kimina-Prover RL, when available)
# formal_ground_truth: human-written proof (only when author == "human")
# ground_truth_type: "complete" or "with_sorry"
```

**CRITICAL FILTERS:**
- `ground_truth_type == "with_sorry"` → SKIP (contains sorry, our filter catches this)
- `author == "autoformalizer"` with no `formal_proof` → statements only, no proof to trace
- `formal_proof` is present → can trace for tactic pairs
- `formal_ground_truth` with `ground_truth_type == "complete"` → highest quality, human-written

**What you need to do:**
1. Filter to rows that have either `formal_proof` or complete `formal_ground_truth`
2. Reconstruct full Lean 4 files (statement + proof)
3. Trace with LeanDojo to get (state, tactic) pairs
4. Lean version must match: compiled against mathlib **v4.15.0**

### 3. Lean Workbook (57K + 83K problems)
**Source:** `internlm/Lean-Workbook` (statements), `internlm/Lean-Workbook-Plus` (more)
**Format:** JSONL or Parquet

```python
# These are mostly STATEMENTS, not proofs
# Example:
{
    "id": "lean_workbook_12345",
    "formal_statement": "theorem lean_workbook_12345 ...",
    "natural_language": "Prove that for all positive integers n, n² ≥ n",
    "status": "open"  # or "proved"
}
```

**What you need to do:** Most of these are unprovable statements (no proof available). The Goedel dataset is the one that actually proved 29.7K of them. Use Goedel Workbook Proofs as the primary source, not raw Lean Workbook.

---

## SFT Training Data Format

After tracing, all datasets are unified into the same format.

### Unified tactic pair record (intermediate format)

```json
{
    "theorem": "lean_workbook_12345",
    "state": "n : ℕ\nh : n > 0\n⊢ n * n ≥ n",
    "tactic": "exact Nat.le_mul_of_pos_left n h",
    "depth": 0,
    "source": "goedel_workbook",
    "num_goals": 1
}
```

All three datasets produce records in this format after tracing. The `source` field tracks provenance.

### SFT training format (what goes to the model)

Matches the Rust `format_tactic_message()` exactly, with tactic appended after the closing fence:

```json
{
    "text": "Complete the following Lean 4 code:\n\n```lean4\n/- tactic state:\nn : ℕ\nh : n > 0\n⊢ n * n ≥ n\n-/\n```\nexact Nat.le_mul_of_pos_left n h"
}
```

The `DataCollatorForCompletionOnlyLM` masks loss on everything up to and including the closing ` ``` \n`. The model only trains on the tactic tokens that follow.

### sft_train.jsonl — one line per tactic pair

```jsonl
{"text": "Complete the following Lean 4 code:\n\n```lean4\n/- tactic state:\nn : ℕ\nh : n > 0\n⊢ n * n ≥ n\n-/\n```\nexact Nat.le_mul_of_pos_left n h", "theorem": "lean_workbook_12345", "source": "goedel_workbook"}
{"text": "Complete the following Lean 4 code:\n\n```lean4\n/- tactic state:\na b : ℝ\n⊢ a + b = b + a\n-/\n```\nring", "theorem": "numina_abc123", "source": "numinamath"}
```

The `theorem` field is kept for the theorem-level train/val split (Gotcha #11) but is NOT part of the model input.

### Formatting script

```python
def format_sft_pair(state: str, tactic: str) -> str:
    """Format a (state, tactic) pair for SFT training.
    
    MUST match the Rust policy crate's format_tactic_message() exactly,
    with the tactic appended as the completion target after the closing fence.
    """
    return (
        f"Complete the following Lean 4 code:\n\n"
        f"```lean4\n"
        f"/- tactic state:\n"
        f"{state}\n"
        f"-/\n"
        f"```\n"
        f"{tactic}"
    )

def format_inference_prompt(state: str) -> str:
    """Format a proof state for inference (no tactic — model generates it).
    
    This is what the Rust policy crate sends to SGLang.
    Identical to format_sft_pair() minus the tactic.
    """
    return (
        f"Complete the following Lean 4 code:\n\n"
        f"```lean4\n"
        f"/- tactic state:\n"
        f"{state}\n"
        f"-/\n"
        f"```"
    )

# For DataCollatorForCompletionOnlyLM:
RESPONSE_TEMPLATE = "```\n"  # Closing code fence = loss mask boundary
```

**CRITICAL GOTCHA:** The closing ` ``` ` appears twice in each example — once opening (`` ```lean4 ``) and once closing (` ``` `). The `response_template` must match the **closing** fence. Since `DataCollatorForCompletionOnlyLM` finds the LAST occurrence of the template, ` ```\n ` will correctly match the closing fence (the opening fence is ` ```lean4\n `, not ` ```\n `). Verify this on Day 1:

```python
from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Prover-V2-7B")
sample = format_sft_pair("n : ℕ\n⊢ n = n", "rfl")
ids = tok.encode(sample)

collator = DataCollatorForCompletionOnlyLM(
    response_template="```\n",
    tokenizer=tok,
)
# Check that the mask boundary falls exactly after the closing fence
```

If the tokenizer fragments ` ``` ` in a way that makes matching unreliable, fall back to using `response_template` as token IDs instead of a string.

**CRITICAL:** The inference prompt format MUST match the training format exactly. If you change the SFT format, you must also update the Rust `policy` crate's `format_tactic_message()`. Since that function already works and produced 41.7% miniF2F, **do not change it** — match the training data to it.

---

## Contrastive Training Data Format (for EBM)

The contrastive data comes from search trajectories, NOT from the raw datasets.

### Data flow

```
Raw datasets → LeanDojo trace → (state, tactic) pairs → SFT training
                                                            ↓
                                              Trained iter_0 model
                                                            ↓
                                    Proof search (2K theorems, 800 nodes)
                                                            ↓
                                    search_iter0.parquet (trajectory data)
                                                            ↓
                                         Contrastive training data for EBM
```

### search_iter0.parquet — trajectory record schema

```
theorem_name:    string    # which theorem this state belongs to
state_pp:        string    # pretty-printed proof state
state_id:        int       # unique ID within the search tree
parent_id:       int       # parent state ID (-1 for root)
depth:           int       # depth from root (root = 0)
tactic:          string    # tactic that produced this state
is_proved:       bool      # does this state lie on a successful proof path?
num_goals:       int       # number of remaining goals
children_ids:    list[int] # IDs of child states
search_priority: float     # LLM log-prob score at expansion time
```

### Contrastive pair construction (for EBM training)

The EBM learns `E(state, goal) → scalar` where lower energy = more provable.

```python
# For each theorem in the trajectory data:
#   - goal = root state (depth 0) — this is the "what we're trying to prove"
#   - positive states = states on a successful proof path (is_proved=True)
#   - negative states = states NOT on any proof path (is_proved=False)

# Negative mining strategy (per positive state):
#   - Hard negatives (60%): siblings (same parent, different tactic choice)
#   - Medium negatives (30%): same theorem, different subtree, similar depth
#   - Easy negatives (10%): random states from different theorems

contrastive_record = {
    "theorem": "lean_workbook_12345",
    "goal_state": "⊢ ∀ n : ℕ, n > 0 → n * n ≥ n",   # root, depth=0
    "positive_state": "n : ℕ\nh : n > 0\n⊢ n * n ≥ n",  # on proof path
    "negative_states": [                                    # not on proof path
        "n : ℕ\nh : n > 0\n⊢ n ≥ 1",       # hard: sibling
        "n : ℕ\n⊢ n * n ≥ 0",              # medium: same theorem
        "x y : ℝ\n⊢ x + y = y + x",       # easy: different theorem
    ],
    "negative_types": ["hard", "medium", "easy"],
    "positive_depth": 1,
}
```

### For decoupled EBM training (Day 6, Config A)

The EBM trains on pre-extracted embeddings:

```python
# Input to GoalConditionedEnergyHead:
#   z_state = frozen embedding of the proof state (from Task 2.1)
#   z_goal = frozen embedding of the root goal (from Task 2.1)
#   input = [z_state; z_goal; z_state ⊙ z_goal]  (concatenated, 12288-dim)
#
# The embeddings come from v2/iter_0/embeddings/:
#   state_embeddings.parquet — one embedding per trajectory state
#   goal_embeddings.parquet — one embedding per theorem (root state)
```

### For joint training (Day 9, Config C)

No pre-extracted embeddings. The model produces embeddings live:

```python
# JointDataset yields per batch:
{
    # SFT stream (same as SFT training):
    "sft_input_ids": ...,      # DeepSeek native prompt + tactic
    "sft_labels": ...,         # masked except tactic tokens
    
    # Contrastive stream (for EBM):
    "positive_states": [...],  # list of state strings
    "negative_states": [...],  # list of negative state strings  
    "goal_states": [...],      # root goal for each positive
    
    # The model forward pass:
    # 1. Run LoRA LM on sft_input_ids → SFT loss (cross-entropy on tactics)
    # 2. Encode positive_states through same LoRA backbone → z_pos
    # 3. Encode negative_states through same backbone → z_neg
    # 4. Encode goal_states through same backbone → z_goal
    # 5. EBM head: E(z_pos, z_goal), E(z_neg, z_goal) → InfoNCE loss
    # 6. Total loss = SFT_loss + λ * InfoNCE_loss (λ=0.1)
    # Gradients from InfoNCE flow back through LoRA → protects embeddings
}
```

---

## Dataset Pipeline Summary

```
┌─────────────────────────────┐
│ Goedel Workbook Proofs      │  29.7K complete proofs
│ (migrated to Lean 4.26)     │  (Phase M migration)
└──────────┬──────────────────┘
           │ LeanDojo trace (Task 0.4)
           │ + sorry filter
           ▼
┌─────────────────────────────┐
│ LEAN-GitHub                 │  218K pre-traced tactics
│ (internlm/Lean-Github)     │  from 28.6K theorems
│                             │  (quality-filtered)
└──────────┬──────────────────┘
           │ Direct use (already state,tactic)
           │ + quality filter + sorry filter
           ▼
┌─────────────────────────────┐
│ NuminaMath-LEAN (Phase 2)   │  ~10-20K with proofs
│ (deferred; after Goedel     │  (filter: has proof,
│  + LEAN-GitHub validated)   │   ground_truth_type ≠ "with_sorry")
└─────────────────────────────┘

         All sources merge into:

┌─────────────────────────────┐
│ Unified tactic pairs        │  ~210-350K pairs
│ {theorem, state, tactic,    │  (source-prefixed dedup)
│  depth, source, num_goals}  │
└──────────┬──────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌──────────┐ ┌──────────────┐
│ SFT data │ │ Contrastive  │
│          │ │ theorem pool │
│ Format:  │ │ (depth ≥ 3)  │
│ DeepSeek │ │              │
│ native   │ │ Used after   │
│ prompt + │ │ iter_0 search│
│ tactic   │ │ to build EBM │
│ {tactic} │ │ training data│
│          │ │              │
│ Split by │ │              │
│ theorem  │ │              │
│ (95/5)   │ │              │
└──────────┘ └──────────────┘
```

---

## Inference-time Prompt (Pantograph search)

When the search engine calls SGLang to generate tactics:

```python
# Rust policy crate constructs this prompt via format_tactic_message():
prompt = (
    f"Complete the following Lean 4 code:\n\n"
    f"```lean4\n"
    f"/- tactic state:\n"
    f"{state_pp}\n"
    f"-/\n"
    f"```"
)

# SGLang generates tactic completions (temperature=0.8, top_p=0.95)
# extract_first_tactic() strips code fences, comments, declarations
# Each candidate tactic is sent to Pantograph for verification
```

The closing `` ``` `` is the generation trigger — the model continues from here.

### Embedding extraction (for EBM scoring)

When the encode server extracts embeddings for the EBM:

```python
# The encode server receives the raw proof state (no code fences or instruction):
state_text = "n : ℕ\nh : n > 0\n⊢ n * n ≥ n"

# It tokenizes with padding_side="right", extracts last-hidden-state
# at the last content token position
```

Note: the embedding input does NOT include the instruction prefix, code fences, or `/- tactic state: ... -/` wrapping. It's the raw proof state text. This is because the EBM cares about the semantic content of the state, not the formatting protocol.
