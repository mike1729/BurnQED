# DeepSeek-Prover-V2 SFT Training — Data & Configuration

Last updated: 2026-03-04

This document consolidates everything known about the v2 SFT training data, prompt format, token geometry, and recommended training configuration. It is the single reference for Task 1.1 (train iter_0 LoRA).

---

## 1. Training Data Summary

### Sources

| Source | Pairs | Theorems | Avg depth | Median depth | Script |
|--------|------:|--------:|----------:|-------------:|--------|
| Goedel Workbook (4.27) | 60,341 | 24,879 | 2.4 | 2 | `python/data/goedel_migration/extract_pairs_pantograph.py` |
| LEAN-GitHub | 196,853 | 19,449 | 10.1 | 4 | `python/data/process_lean_github.py` |
| **Total** | **257,194** | **44,328** | 5.8 | 2 | `python/data/merge_sft_dataset.py` |

### Final SFT Dataset

| Split | Pairs | Theorems | File |
|-------|------:|--------:|------|
| Train | 245,025 | 41,563 | `data/sft/train.jsonl` |
| Val | 12,169 | 2,094 | `data/sft/val.jsonl` |

Split is **by theorem name** (Gotcha 11) using deterministic MD5 hash, 95/5 ratio. No theorem appears in both splits.

### Contrastive Pool

19,213 theorems with depth >= 3 (222,379 pairs). Saved in `data/sft/contrastive_pool.json`. Used for Task 1.3 (search) and Phase 2 (EBM training). Breakdown:

| Source | Depth >= 3 | Depth >= 5 |
|--------|----------:|----------:|
| Goedel | 7,023 | 3,288 |
| LEAN-GitHub | 12,190 | 9,027 |
| **Total** | **19,213** | **12,315** |

---

## 2. Data Characteristics

### Goedel Workbook

- **Origin:** DeepSeek-Prover-V1.5 generated proofs of Lean Workbook competition problems, migrated to Lean 4.27.
- **Extraction:** Pantograph REPL replay of compiled proofs (not LeanDojo ExtractData — see `memory/leandojo_tracing.md`).
- **28,016 proofs compiled** on Lean 4.27 (94.1% survival from 29,759). Extracted 24,879 theorems (2,487 regex parse failures, 650 Pantograph replay errors).
- **Shallow proofs:** 46.3% are single-tactic, 71.8% are 1-2 tactics. Machine-generated proofs favor powerful automation.
- **Dominant tactics:** `nlinarith` (53% of single-tactic proofs), `have` (27% overall), `field_simp`, `ring_nf`.
- **State complexity:** Median 5 variables per state, 6.6% multi-goal states.
- **Integrity:** 0 sorry/admit/cheat in tactic blocks (verified in `data/traced/integrity_report.json`).

### LEAN-GitHub

- **Origin:** Human-written proofs from 147 GitHub repos, pre-traced by InternLM team using LeanDojo.
- **No compilation needed:** Pre-traced `state_before`/`tactic` strings used directly for SFT.
- **Much deeper proofs:** Median 4 tactics, mean 10.1. 64.7% have depth >= 3.
- **Diverse tactics:** 561 unique heads, top-5 cover only 54% (vs Goedel's 62%). Rich in `have`, `cases`, `rcases`, `obtain`, `refine` — structured reasoning the model needs to learn.
- **State complexity:** Median 10 variables per state (2x Goedel), 5.8% multi-goal.
- **Top repos:** girving/ray, pthomas505/FOL, iehality/lean4-logic, AlexKontorovich/PrimeNumberTheoremAnd, flt-regular, leansat, IMOSLLean4.

### Filters Applied

| Filter | Goedel | LEAN-GitHub |
|--------|--------|-------------|
| Sorry/admit/cheat (whole theorem) | 0 removed | 1,364 theorems (10,474 rows) removed |
| State > 4096 chars | N/A (Pantograph limits) | 630 rows removed |
| Trivial tactic subsampling (10%) | N/A | 10,909 rows removed |
| Regex parse failures | 2,487 theorems | N/A |
| Pantograph replay errors | 650 theorems | N/A |

### Lean 4.27 Compatibility

- **Goedel:** Fully compiled and traced on Lean 4.27. All tactics verified via Pantograph replay. 100% compatible.
- **LEAN-GitHub:** Pre-traced strings, version-agnostic for SFT. Compatibility analysis:
  - 71.4% of tactics are pure automation (no lemma name references) — fully safe
  - 0.24% use confirmed-renamed APIs (`Int.coe_nat`, `Array.data`, `ext1`) — negligible
  - 13.9% use rw/simp with qualified Mathlib lemma names — low risk, most are stable across versions
  - **Net assessment: 99.8% safe.** No filtering needed beyond what was applied.

### Zero Overlap

No theorem name overlap between Goedel (`lean_workbook_*`) and LEAN-GitHub (diverse repo namespaces). Dedup is unnecessary but source-prefixed keys are used in the split for safety.

---

## 3. Prompt Format

**SETTLED — must match Rust `policy` crate's `format_tactic_message()` exactly.**

### Training example (`data/sft/train.jsonl`)

```json
{
  "text": "Complete the following Lean 4 code:\n\n```lean4\n/- tactic state:\nn : ℕ\nh : n > 0\n⊢ n * n ≥ n\n-/\n```\nexact Nat.le_mul_of_pos_left n h",
  "theorem": "lean_workbook_12345",
  "source": "goedel_workbook",
  "depth": 0
}
```

### Format function

```python
def format_sft_pair(state: str, tactic: str) -> str:
    return (
        f"Complete the following Lean 4 code:\n\n"
        f"```lean4\n"
        f"/- tactic state:\n"
        f"{state}\n"
        f"-/\n"
        f"```\n"
        f"{tactic}"
    )
```

### Loss masking

`DataCollatorForCompletionOnlyLM` with `response_template="```\n"` (closing code fence + newline). The opening fence is `` ```lean4\n `` which does NOT match the template. Only the closing `` ```\n `` matches, so the mask boundary is correct: everything before is prompt, everything after is the tactic completion the model trains on.

**Critical:** Verify with the tokenizer that `` ``` `` is not fragmented. If it is, use token IDs instead of string template.

---

## 4. Token Geometry

Tokenizer: DeepSeek-Prover-V2-7B (same as Goedel-Prover-V2-8B base). Vocab size: 151,643.

### Full sequence (prompt + tactic)

| Percentile | Tokens |
|----------:|-------:|
| p25 | 103 |
| p50 | 156 |
| p75 | 246 |
| p90 | 384 |
| **p95** | **506** |
| p99 | 793 |
| p99.5 | 975 |
| p99.9 | 1,383 |
| max | 67,837 |
| Mean | 202 |
| Std | 210 |

### State only

| Percentile | Tokens |
|----------:|-------:|
| p50 | 113 |
| p95 | 457 |
| p99 | 725 |

### Tactic only

| Percentile | Tokens |
|----------:|-------:|
| p50 | 10 |
| p95 | 78 |
| p99 | 172 |

### Per-source

| Source | p95 | p99 |
|--------|----:|----:|
| Goedel | 296 | 471 |
| LEAN-GitHub | 545 | 844 |

### Truncation analysis

| max_length | Truncated | % |
|-----------:|----------:|--:|
| 512 | 11,790 | 4.81% |
| 768 | 2,749 | 1.12% |
| **1024** | **1,010** | **0.41%** |
| 1536 | 132 | 0.05% |
| 2048 | 25 | 0.01% |

### Recommendation: `max_length = 1024`

- Captures 99.6% of data (only 0.41% truncated)
- Saves ~50% VRAM vs 2048 (Gotcha 17)
- p95 is only 506 tokens — most batches will be well under the limit
- Mean sequence is 202 tokens — dynamic padding will keep most batches efficient

---

## 5. Tactic Vocabulary

### Combined dataset (257K pairs, 849 unique tactic heads)

| Rank | Tactic | Count | % | Cumulative |
|-----:|--------|------:|--:|-----------:|
| 1 | have | 34,188 | 13.3% | 13.3% |
| 2 | simp | 27,904 | 10.8% | 24.1% |
| 3 | rw | 27,438 | 10.7% | 34.8% |
| 4 | exact | 21,857 | 8.5% | 43.3% |
| 5 | apply | 17,006 | 6.6% | 49.9% |
| 6 | intro | 16,634 | 6.5% | 56.4% |
| 7 | nlinarith | 10,763 | 4.2% | 60.6% |
| 8 | cases | 5,577 | 2.2% | 62.7% |
| 9 | refine | 4,899 | 1.9% | 64.6% |
| 10 | case | 4,851 | 1.9% | 66.5% |

### Diversity metrics

| Metric | Goedel | LEAN-GitHub | Combined |
|--------|-------:|----------:|--------:|
| Unique tactic heads | 352 | 561 | 849 |
| Shannon entropy (bits) | 3.99 | 4.87 | 5.00 |
| Effective vocabulary | 16 | 29 | 32 |
| Top-5 coverage | 62.3% | 53.9% | 49.9% |
| Top-10 coverage | 78.7% | 69.3% | 66.5% |

LEAN-GitHub is essential for diversity. Without it, the model would be heavily biased toward `nlinarith` (Goedel's dominant tactic). Combined top-5 coverage of 50% is healthy — the model must learn a broad vocabulary.

### Tactic distribution by proof depth

| Depth | Top tactics |
|-------|------------|
| 0 (initial) | nlinarith 14%, have 13%, intro 11%, rw 9%, simp 8% |
| 1-2 | have 18%, simp 10%, rw 9%, nlinarith 7%, exact 6% |
| 3-5 | have 17%, exact 10%, simp 10%, rw 10%, intro 6% |
| 6+ | simp 13%, rw 12%, exact 12%, have 10%, apply 8% |

Shallow proofs are dominated by automation (`nlinarith`, `intro`). Deeper proofs shift to structured reasoning (`exact`, `apply`, `rw`). This is the pattern the model should learn.

---

## 6. Depth Distribution

### Per-source

| Depth bucket | Goedel | LEAN-GitHub |
|-------------|-------:|----------:|
| 1 tactic | 11,512 (46.3%) | 2,816 (15.0%) |
| 2 tactics | 6,344 (25.5%) | 3,772 (20.1%) |
| 3-5 | 4,725 (19.0%) | 4,005 (21.3%) |
| 6-10 | 1,975 (7.9%) | 3,540 (18.9%) |
| 11-20 | 310 (1.2%) | 2,430 (12.9%) |
| 21-50 | 13 (0.1%) | 1,625 (8.7%) |
| 51+ | 0 | 590 (3.1%) |

### Single-tactic proof breakdown (Goedel)

11,512 single-tactic proofs (46.3%). Dominated by:

| Tactic | Count | % of single-tactic |
|--------|------:|-------------------:|
| nlinarith | 6,058 | 52.6% |
| have | 1,539 | 13.4% |
| norm_num | 672 | 5.8% |
| cases | 525 | 4.6% |
| induction | 457 | 4.0% |
| rw | 400 | 3.5% |

These are still valuable — the model learns *when* to deploy power tactics. A competition algebra problem where `nlinarith` suffices in one step is useful training signal.

---

## 7. Recommended Training Configuration

Based on the data analysis above and Gotchas 1-17 from CLAUDE.md.

### Model

- **Base:** DeepSeek-Prover-V2-7B (or Goedel-Prover-V2-8B, same architecture)
- **LoRA:** rank 32 for iter_0 (rank 64 for iter_1 joint training)
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_length | **1024** | p99.5 = 975, only 0.41% truncated |
| batch_size | 8-16 per GPU | Adjust to fit VRAM with max_length=1024 |
| gradient_accumulation | 4-8 | Effective batch ~64-128 |
| learning_rate | 2e-5 | Standard for LoRA SFT |
| lr_scheduler | cosine | With 3% warmup |
| epochs | 2-3 | 245K samples × 2-3 epochs = ~500-750K steps / effective_batch |
| weight_decay | 0.01 | Standard |
| fp16/bf16 | bf16 | DeepSeek-V2 supports bf16 |
| padding_side | right | Gotcha 4 — verify last-token indexing |

### Loss masking

```python
from trl import DataCollatorForCompletionOnlyLM

collator = DataCollatorForCompletionOnlyLM(
    response_template="```\n",  # Closing code fence = mask boundary
    tokenizer=tokenizer,
)
```

Verify on Day 1 that the mask falls correctly (Gotcha 13). If tokenizer fragments the backticks, switch to token ID template.

### Data loading

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "data/sft/train.jsonl",
    "validation": "data/sft/val.jsonl",
})
# The "text" field contains the full formatted example
```

### VRAM estimate

- DeepSeek-Prover-V2-7B in bf16: ~14 GB
- LoRA r=32 adapters: ~0.2 GB
- Optimizer states (AdamW, bf16): ~0.4 GB
- Activations (max_length=1024, batch=8): ~8-12 GB
- **Total: ~24-28 GB** — fits on a single A100 40GB or similar

### Monitoring

- **Training loss:** Should drop below 1.0 within first 1K steps, converge around 0.3-0.5
- **Val loss:** Track for overfitting. Expect slight gap from training loss
- **Completion-only sanity check:** Log a few decoded completions every 500 steps to verify the model generates tactics (not echoing states)

---

## 8. Data Files Reference

```
data/
├── lean/
│   └── lean_github/
│       └── lean-github.parquet          # Raw LEAN-GitHub (218K rows, 42 MB)
├── traced/
│   ├── pantograph_pairs/
│   │   ├── chunk_000.jsonl ... chunk_015.jsonl  # Per-chunk Pantograph output
│   │   └── goedel_427_pairs.jsonl       # Merged Goedel pairs (60K)
│   ├── lean_github_pairs.jsonl          # Filtered LEAN-GitHub pairs (197K)
│   └── integrity_report.json            # Goedel sorry/admit sweep
├── sft/
│   ├── train.jsonl                      # SFT training data (245K pairs)
│   ├── val.jsonl                        # SFT validation data (12K pairs)
│   └── contrastive_pool.json            # Theorems with depth>=3 (19K)
└── models/
    └── base/
        └── goedel-prover-v2-8b/         # Base model + tokenizer
```

### Scripts

| Script | Purpose |
|--------|---------|
| `python/data/goedel_migration/extract_pairs_pantograph.py` | Pantograph-based tactic extraction from compiled Goedel proofs |
| `python/data/process_lean_github.py` | LEAN-GitHub quality filtering + format conversion |
| `python/data/merge_sft_dataset.py` | Merge sources, format to DeepSeek-native, train/val split |
