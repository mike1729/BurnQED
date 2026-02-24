# Performance Tuning Notes

Findings from parameter sweep on miniF2F test set (2026-02-24).
Hardware: 2× GPU (GPU 0: SGLang inference, GPU 1: encode server bf16).
Model: DeepSeek-Prover-V2-7B iter_4 LLM + iter_4 EBM unless noted.

## Architecture Change: Remove Rust-Side Batchers

Removed `GlobalBatcher` and `GlobalEncodeBatcher` (Nagle-style request coalescing in Rust).
Each search thread now sends `/generate` and `/encode` HTTP requests directly.
SGLang's continuous batching + RadixAttention handles batching server-side.
Encode server gained asyncio queue-based dynamic batching (5ms linger).

**Result:** Simpler code, no throughput regression. SGLang's internal scheduler
is more effective than our client-side coalescing.

## Time Breakdown: Where Time Goes

Across all configurations tested, **LLM generation dominates wall time**:

| Phase    | % of wall time | Avg latency | Notes                          |
|----------|---------------|-------------|--------------------------------|
| Generate | 80–93%        | 1.5–9.7s    | Scales with batch size         |
| Lean     | 4–11%         | 150–500ms   | Tactic verification            |
| EBM      | 2–9%          | 260–470ms   | Encode + MLP scoring           |

Lean and EBM are not bottlenecks. All optimization effort should focus on
reducing generate latency or making better use of each generation call.

## batch_expansion_size: Biggest Latency Lever

`batch_expansion_size` (bex) controls how many nodes are popped from the
priority queue per expansion step. Each expansion sends
`bex × num_candidates` prompts to SGLang in one batch.

| bex | States/generate | Avg gen_ms | Notes                           |
|-----|----------------|------------|---------------------------------|
| 2   | ~2             | 9,057      | 32 prompts per batch at nc=16   |
| 1   | ~1             | 3,294      | 16 prompts per batch at nc=16   |
| 1   | ~1             | 1,819      | 8 prompts per batch at nc=8     |

**bex=1 is 3× faster** than bex=2 for generation. The tradeoff is less
batching amortization, but SGLang handles the smaller batches efficiently.
**Recommendation: bex=1** for all configurations.

## Concurrency: Minimal Impact on Gen Latency

| Concurrency | Avg gen_ms | Notes                    |
|-------------|-----------|--------------------------|
| 16          | 9,057     | bex=2, nc=16             |
| 8           | 9,678     | bex=2, nc=16             |
| 16          | 3,294     | bex=1, nc=16             |
| 10          | 1,531     | bex=1, nc=10             |

Halving concurrency (16→8) did NOT reduce gen latency — SGLang's scheduler
handles concurrent requests well. Concurrency mainly affects how many
theorems run in parallel, not per-request latency.

## Temperature and num_candidates Sweep

All runs: bex=1, max_nodes=600, conc=10, max_theorems=10, same first-10 theorems.

| nc  | temp | Unique tactics | Children/exp | Gen (ms) | Proved |
|-----|------|---------------|-------------|----------|--------|
| 8   | 0.8  | 3.5/8 (44%)   | 1.5         | 1,466    | —      |
| 10  | 1.0  | 5.4/10 (54%)  | 1.7         | 1,470    | —      |
| 10  | 1.3  | 7.5/10 (75%)  | 1.6         | 1,531    | —      |
| 16  | 1.3  | 10.3/16 (64%) | 2.3         | 1,817    | 9/10   |
| 8   | 1.2  | 5.2/8 (65%)   | 1.3         | 2,052*   | 8/16*  |
| 16  | 1.2  | 13.6/16 (85%) | 3.9         | 9,057*   | —      |

*Different concurrency/bex settings, not directly comparable.

### Key observations:

1. **Lean pass rate is ~25–30%** regardless of temperature/nc. Of unique
   tactics generated, only about 1 in 4 passes Lean verification.

2. **Children/expansion ≈ unique_tactics × 0.25**. To get 2–4 children/exp
   you need 8–16 unique tactics.

3. **Gen latency is flat at ~1.5s for nc≤10** with bex=1. SGLang batches
   small requests efficiently. nc=16 adds ~0.3s.

4. **Higher temp increases diversity** but with diminishing returns on
   children/exp (many diverse tactics still fail Lean).

5. **Best config for prove rate: nc=16, t=1.3** — found 9/10 proofs.
   The extra unique tactics (10.3) give enough Lean-passing children
   to maintain broad search.

## EBM Value Function: Currently Noise

Analyzed EBM accuracy on proof paths from the nc=16/t=1.3 run (9 proved theorems):

- **EBM picked the correct child 13% of the time** (9/67 decisions).
  With ~10 siblings, random would be ~10%. Barely above chance.

- **Proof-completing nodes always get ebm_score=0.0** (fallback) and rank
  last among siblings. The EBM actively penalizes proof completion.

- **Directionally correct on averages**: proof-path nodes mean=-0.50,
  dead-end nodes mean=-0.22. But not discriminative at the sibling level.

- **Per-theorem accuracy**: 0–29%, worst on deep proofs (12% for depth-17).

**Conclusion:** Proofs are found by LLM policy + search breadth, not EBM
guidance. The EBM needs significantly better training data or architecture
changes to be useful for search guidance.

### Known bug: terminal node score = 0.0

Proof-completing states are never sent to the encode server (they have no
remaining goals to embed), so they receive the fallback `ebm_score = 0.0`.
Since proof-path nodes average `-0.50` and dead-ends average `-0.22`, a
score of `0.0` makes the actual winning state look like the **worst**
possible state in the priority queue.

**Fix:** Hardcode a large negative (= "very provable") score for terminal
states in the search engine, e.g. `ebm_score = -100.0`, so they are
popped immediately if they ever enter the queue. This doesn't affect
correctness (terminal states are detected on creation), but fixes the
score used in trajectory labels and any future backpropagation of values.

### Known bug: `sorry` tactic leak

Two proofs ended with `sorry` (aime_1990_p4, aime_1999_p11). The LLM
learned from Mathlib training data that `sorry` closes goals without error.
Lean accepts `sorry` as a valid tactic — it just marks the proof as
incomplete.

**Fix:** Add a lexical filter to reject `sorry` and `admit` tactics before
sending them to Lean. This should go in the candidate loop in
`engine.rs`, before `apply_tactic`.

### Why the EBM doesn't discriminate siblings

With a branching factor of 3.6 and 99.1% negative labels, the EBM's
training data is overwhelmingly "easy" negatives. The model learns to
predict a flat, safe average (`-0.22`) rather than the subtle differences
between a good step and a bad step at the same parent.

**Fix path:** Use the fast generation pipeline (`nc=16, t=1.3, bex=1`) to
generate thousands of deep trajectories. Feed them into `train-ebm` with
the contrastive sampler (`hard_ratio`) to train on hard negatives
(siblings of proof-path nodes). This teaches the EBM to rank siblings
rather than just separate "any positive" from "any negative".

## Trajectory Statistics

From nc=16/t=1.3 run (16 theorems, 8 proved):

- **Total nodes:** 7,726
- **Label balance:** 99.1% negative, 0.9% positive (71 nodes on proof paths)
- **Branching factor:** mean 3.6 children/node
- **Depth distribution:** bell-shaped, peaks at depth 9, max 19
- **Proof depths:** 3–16 (mean 7.9, median 7.0)
- **Nodes/theorem:** mean 483, range 23–1,837

## Recommended Default Configuration

```toml
[search]
max_nodes = 600
batch_expansion_size = 1
num_candidates = 16
timeout_per_theorem = 600
```

CLI: `--concurrency 10 --temperature 1.3`

This gives: ~1.8s gen latency, ~10 unique tactics, ~2.3 children/exp,
~50% prove rate on miniF2F test (iter_4 model).
