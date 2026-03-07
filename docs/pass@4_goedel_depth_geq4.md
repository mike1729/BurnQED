# pass@4 Base Model Search on Goedel Depth>=4 Theorems

**Date:** 2026-03-06
**Model:** DeepSeek-Prover-V2-7B (base, fp8)
**Config:** `configs/search_goedel_pass4.toml`
**Dataset:** Goedel Workbook (Lean 4.27), depth >= 4 only

## Configuration

| Parameter | Value |
|-----------|-------|
| Search mode | pass@4 (single round, 4 whole-proof attempts) |
| Max tokens | 2048 |
| Temperature | 1.0 |
| Timeout/theorem | 600s |
| Concurrency | 8 |
| Lean workers | 8 |
| EBM | None (base model, LLM-only) |

## Overall Results

| Metric | Value |
|--------|-------|
| Theorems in benchmark | 3,288 |
| Theorems searched | 3,285 |
| Errors (skipped) | 3 (0.09%) |
| **Proved** | **616 (18.8%)** |
| Failed | 2,669 (81.2%) |
| Total trajectory records | 28,427 |
| Avg nodes/theorem | 8.7 |
| Wall time | ~112 min |
| Throughput | 29.2 theorems/min |

## Depth-Stratified Results

| Depth | Attempted | Proved | Rate |
|-------|-----------|--------|------|
| 4 | 989 | 302 | 30.5% |
| 5 | 640 | 92 | 14.4% |
| 6 | 546 | 104 | 19.0% |
| 7-10 | 931 | 94 | 10.1% |
| 11+ | 179 | 24 | 13.4% |

Depth 4 has the highest success rate (30.5%). Deeper proofs are harder but the model still solves 10-13% at depth 7+. The depth 6 bump (19.0% > depth 5's 14.4%) suggests some structured proofs at that depth are amenable to whole-proof generation.

## Node Statistics

| Metric | Proved | Failed |
|--------|--------|--------|
| Avg nodes/theorem | 7.3 | 9.1 |

Failed theorems use slightly more nodes (more proof attempts exhausted without success).

## Timing

| Phase | Time (s) | % of wall |
|-------|----------|-----------|
| LLM generation | 12,417 | 186% (overlapped across concurrency) |
| Lean verification | 4,605 | 69% |
| Total wall time | 6,694 | 100% |

## Errors

Only 3 theorems (0.09%) had statement parsing errors:
- `lean_workbook_1642`: hygienic name `h✝` in tactic state (not valid in source expressions)
- `lean_workbook_plus_32839`: universe polymorphism (`u_1`, `u_2`) in type signature
- `lean_workbook_plus_80141`: similar edge case

Statements are extracted from Pantograph tactic states (root goal at depth 0) by converting hypotheses to `∀`-binders. These 3 edge cases would require special handling (stripping `✝`, instantiating universe variables).

## Latency Percentiles

| Metric | p50 | p95 | p99 |
|--------|-----|-----|-----|
| Lean/tactic | 22ms | 299ms | 1,978ms |
| LLM gen/batch | 3.3s | 7.9s | 12.9s |

## Key Observations

1. **18.8% pass@4 on depth>=4** — decent base model performance on hard theorems with only 4 attempts and 2048 tokens
2. **Steep depth gradient:** 30.5% at depth 4 drops to ~10-13% at depth 7+
3. **Only 3 errors** (0.09%) — fixed from initial 253 errors by extracting full multi-line goals from tactic states instead of truncating at first line
4. **Whole-proof generation** works surprisingly well for these competition problems — avg 7.3 nodes for proved theorems suggests most are solved in few attempts
5. **Depth 6 anomaly:** 19.0% vs depth 5's 14.4% — likely due to structured case-split proofs that are formulaic despite depth
