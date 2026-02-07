# Week 2 Pipeline Summary — Phase 2: LLM Policy Crate

## Overview

Phase 2 delivers **LLM-based tactic generation and embedding extraction** via the `policy` crate. This is the core inference component that the search engine (Phase 3) will use to:

1. **Generate candidate tactics** for a given proof state (autoregressive decoding with top-p sampling)
2. **Extract mean-pooled hidden-state embeddings** for the Energy-Based Model to score proof states

The crate wraps DeepSeek-Prover-V2-7B (a Llama-architecture model) loaded via [candle](https://github.com/huggingface/candle), Hugging Face's Rust ML framework. All inference runs on CPU (f32) or CUDA (bf16) — no Python dependency at inference time.

## Architecture

```
Input: proof state string ("n : Nat\n⊢ n + 0 = n")
                │
                ▼
      ┌─────────────────┐
      │  LeanTokenizer   │   tokenizer.rs
      │  (HuggingFace)   │   BPE encode → token IDs
      └────────┬─────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Prompt Formatting    │   model.rs: format_prompt()
    │  [GOAL]{state}        │   "[GOAL]n : Nat\n⊢ n + 0 = n[PROOFSTEP]"
    │  [PROOFSTEP]          │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Llama Transformer    │   llama.rs (forked from candle-transformers)
    │  (7B params, frozen)  │
    │                       │
    │  ┌─────────────────┐  │
    │  │ forward()        │──┼──▶ logits → sample_top_p() → next token
    │  │ (autoregressive) │  │     ↓ loop until EOS or max_tokens
    │  └─────────────────┘  │     ↓
    │                       │   GeneratedTactic { text, log_prob, tokens }
    │  ┌─────────────────┐  │
    │  │ forward_hidden   │──┼──▶ (batch, seq_len, hidden_size) tensor
    │  │ _states()        │  │     ↓ mean-pool over seq_len
    │  │ (encoder path)   │  │     ↓
    │  └─────────────────┘  │   Embedding { data: Vec<f32>, dim: 4096 }
    └──────────────────────┘
```

The shared backbone serves dual purposes — the LM head for tactic generation and the hidden states for EBM scoring — following the AlphaZero-style "one network, two heads" design. The `encode_only()` path uses a **fresh KV cache** per call (no caching) to ensure deterministic embeddings.

## Module Walkthrough

### `types.rs` — Configuration and Data Types

| Type | Purpose |
|------|---------|
| `PolicyConfig` | Model path, sampling params (temperature, top_p), max sequence length, device selection. Deserializable from TOML. |
| `DeviceConfig` | Enum: `Cpu` or `Cuda { ordinal }`. Converts to candle `Device`. |
| `GeneratedTactic` | Output of generation: tactic text, sum of log-probabilities, raw token IDs. |
| `Embedding` | Mean-pooled hidden-state vector: `Vec<f32>` + dimensionality. |

**8 unit tests**: default construction, serde deserialization (CPU, CUDA, full config), data type construction.

### `tokenizer.rs` — LeanTokenizer

Wraps a HuggingFace `tokenizers::Tokenizer` loaded from `tokenizer.json` in the model directory.

Key design: **BOS/EOS token discovery with fallback**. DeepSeek-Prover-V2 uses fullwidth Unicode special tokens (`<｜begin▁of▁sentence｜>`), while standard Llama models use `<s>`/`</s>`. The tokenizer tries DeepSeek names first, then falls back to Llama names, so the same code works with both DeepSeek-7B and TinyLlama-1.1B.

Methods:
- `load(model_path)` — load tokenizer + discover BOS/EOS IDs
- `encode(text)` / `encode_with_bos(text)` — tokenize without/with BOS prefix
- `decode(ids)` — detokenize, skipping special tokens
- `truncate(ids, max_len)` — static helper for smart truncation
- `vocab_size()`, `bos_token_id()`, `eos_token_id()` — accessors

**4 unit tests**: truncate short/exact/long/empty sequences.

### `llama.rs` — Forked Llama Implementation

Forked from `candle-transformers` 0.8.4 (`models/llama.rs`) because the upstream version has:
- **Private fields** — can't access hidden states from outside the struct
- **No encoder mode** — only returns logits for the last token, not full sequence hidden states

Changes from upstream:
1. **`forward_hidden_states()`** — returns `(batch, seq_len, hidden_size)` tensor after the final RmsNorm but *before* the `lm_head` projection. This is what we mean-pool for embeddings.
2. **`DeepSeekConfig`** — custom deserializer for the model's `config.json` that accepts YaRN `rope_scaling` fields (which candle's `LlamaConfig` can't parse). Converts to the runtime `Config` with `rope_scaling: None` since YaRN is backwards-compatible within the original 4096 context window and our sequences are ≤2048 tokens.
3. **`EosToks` enum** — handles both single (`u32`) and multiple (`Vec<u32>`) EOS token representations in config.json.
4. **Inlined `with_tracing` wrappers** — `Linear` and `RmsNorm` wrappers with tracing spans, inlined from candle-transformers to avoid depending on its internals.
5. **Removed flash-attn** — not needed for CPU/standard CUDA inference.

Internal structure: `Cache` (KV cache + precomputed RoPE cos/sin tables) → `CausalSelfAttention` (GQA with RoPE) → `Mlp` (SwiGLU) → `Block` (pre-norm transformer block) → `Llama` (full model with embed + blocks + norm + lm_head).

**4 unit tests**: DeepSeekConfig deserialization (with/without YaRN), EosToks single/multiple.

### `model.rs` — TacticGenerator

The main entry point for all inference. Owns the `Llama` model, `LeanTokenizer`, `Cache`, and `PolicyConfig`.

**Loading** (`TacticGenerator::load`):
1. Parse `config.json` → `DeepSeekConfig` → `Config`
2. Find `*.safetensors` files (supports multi-shard models)
3. Memory-map weights via `VarBuilder::from_mmaped_safetensors`
4. Construct `Llama`, `LeanTokenizer`, and KV `Cache`
5. Select dtype: f32 on CPU, bf16 on CUDA

**Tactic generation** (`generate_one`, `generate_candidates`):
1. Format prompt: `[GOAL]{proof_state}[PROOFSTEP]`
2. Tokenize with BOS, truncate to `max_seq_len`
3. Prefill: process all prompt tokens in one forward pass
4. Decode autoregressively: sample next token via `sample_top_p()`, accumulate log-probability, repeat until EOS or `max_tactic_tokens`
5. `generate_candidates(n)` calls `generate_one()` n times, sorts by log-prob descending

**Top-p (nucleus) sampling** (`sample_top_p`):
- Temperature=0 → greedy (argmax)
- Temperature>0 → sort probabilities descending, find smallest set whose cumulative probability ≥ top_p, re-normalize within that nucleus, sample uniformly

**Embedding extraction** (`encode_only`, `encode_batch`):
1. Tokenize (no BOS for encoding), truncate
2. Create a **fresh cache** (no KV caching) — ensures deterministic output
3. Forward through all transformer layers via `forward_hidden_states()`
4. Mean-pool across sequence dimension: `(1, seq_len, hidden) → (1, hidden) → Vec<f32>`
5. `encode_batch` processes texts sequentially (batch GPU processing deferred to Phase 4)

**5 unit tests**: prompt formatting (normal + empty), top-p greedy/temperature/narrow-nucleus.

## Integration Test Infrastructure

### TinyLlama-1.1B as Fast Local Proxy

Integration tests default to **TinyLlama-1.1B** (`models/tinyllama-1.1b/` relative to workspace root) instead of the full DeepSeek-Prover-V2-7B. This provides:

- **Fast CI**: ~1.1B params vs 7B, loads in seconds on CPU
- **Same architecture**: Llama-compatible, exercises the same code paths
- **No special tokens needed**: Fallback BOS/EOS discovery handles standard `<s>`/`</s>`
- **Dynamic dimension assertions**: Tests use `gen.hidden_size()` instead of hardcoded `4096`, so they pass with any Llama-variant model

Override with `MODEL_PATH=path/to/deepseek-7b` for full-model testing.

### Test Helper Functions

```rust
// model_path() — resolve MODEL_PATH env var or default to models/tinyllama-1.1b
// test_config() — PolicyConfig with conservative settings (3 candidates, 32 max tokens)
```

## Test Coverage

### Unit Tests (21 total, no model needed)

| Module | Count | What's tested |
|--------|-------|---------------|
| `types` | 8 | DeviceConfig default/CPU/CUDA deserialization, PolicyConfig defaults + deserialization, GeneratedTactic + Embedding construction |
| `tokenizer` | 4 | Truncate: short, exact, long, empty |
| `llama` | 4 | DeepSeekConfig deserialization (with YaRN, without YaRN), EosToks single, EosToks multiple |
| `model` | 5 | format_prompt normal + empty, sample_top_p greedy + temperature + narrow nucleus |

```bash
cargo test -p policy    # runs all 21 unit tests
```

### Integration Tests (10 total, require model weights)

| Test | What's verified |
|------|----------------|
| `test_model_loads` | Model + tokenizer load successfully, hidden_size > 0 |
| `test_tokenizer_roundtrip` | encode → decode preserves text |
| `test_tokenizer_special_tokens` | BOS/EOS IDs exist, encode_with_bos prepends BOS |
| `test_forward_logits_shape` | Forward pass returns `(1, vocab_size)` logits |
| `test_generate_one_tactic` | Generates non-empty tactic with finite log-prob |
| `test_generate_candidates_sorted` | Multiple candidates sorted by log-prob descending |
| `test_encode_only_shape` | Embedding dim == model hidden_size |
| `test_encode_only_distinct` | Different states → cosine similarity < 0.999 |
| `test_encode_only_deterministic` | Same state twice → max element diff < 1e-5 |
| `test_encode_batch` | Batch of 3 texts → 3 embeddings with correct dim |

```bash
# Default: uses models/tinyllama-1.1b
cargo test -p policy -- --ignored --nocapture --test-threads=1

# Override for full 7B model:
MODEL_PATH=models/deepseek-prover-v2-7b cargo test -p policy -- --ignored --nocapture --test-threads=1
```

## Design Decisions

### Why fork llama.rs?

candle-transformers' `Llama` has private fields and only exposes `forward()` returning last-token logits. We need `forward_hidden_states()` for mean-pooled embeddings. Forking ~540 lines is cleaner than patching upstream or using unsafe access. The fork is self-contained and only depends on `candle-core` and `candle-nn`.

### Why f32 on CPU?

candle's CPU backend has limited bf16 support — many operations fall back to f32 anyway, causing overhead. Using f32 directly is faster and avoids silent precision issues. CUDA uses bf16 for memory efficiency.

### Why fresh cache for encode_only?

KV caching across calls would make embeddings depend on what was encoded previously (the cache accumulates state). A fresh cache per call guarantees deterministic output: same input always produces the same embedding, regardless of prior calls.

### Why [GOAL]...[PROOFSTEP] prompt format?

This matches the training format used by DeepSeek-Prover-V2. The model was trained to complete after `[PROOFSTEP]` with a tactic. Simple and unambiguous — no complex chat template needed.

### Why TinyLlama for integration tests?

DeepSeek-Prover-V2-7B is 14GB and takes minutes to load on CPU. TinyLlama-1.1B is ~2GB, loads in seconds, and has the same Llama architecture. All tests use dynamic assertions (`gen.hidden_size()` instead of `== 4096`) so they pass with either model. This enables fast local iteration without compromising test quality.

## Phase 2 → Phase 3 Handoff

The search engine (Phase 3) will consume:

### From `TacticGenerator`
- `generate_candidates(proof_state, n)` → `Vec<GeneratedTactic>` — candidate tactics to expand in the search tree
- `encode_only(proof_state)` → `Embedding` — hidden-state vectors for EBM scoring

### Key Types
- `GeneratedTactic { text, log_prob, tokens }` — the search engine uses `text` for Lean verification and `log_prob` for policy-based node scoring
- `Embedding { data: Vec<f32>, dim }` — the EBM (Phase 4) will take these as input to produce scalar energy scores

### Integration Points
```rust
// In the search loop:
let candidates = generator.generate_candidates(&goal_string, 32)?;
for tactic in &candidates {
    // 1. Verify with Lean REPL (Phase 1)
    let result = proof_handle.run_tactic(state_id, None, &tactic.text).await?;

    // 2. Score new state with EBM (Phase 4)
    let embedding = generator.encode_only(&new_goal_string)?;
    let energy = ebm.score(&embedding);  // lower = more promising

    // 3. Add to priority queue
    search_tree.insert(ScoredNode::new(new_state, -energy + tactic.log_prob));
}
```

## Files Changed in Phase 2

| File | Lines | Description |
|------|-------|-------------|
| `crates/policy/Cargo.toml` | 16 | Dependencies: candle-core/nn/transformers, tokenizers, rand |
| `crates/policy/src/lib.rs` | 22 | Module declarations + public API re-exports |
| `crates/policy/src/types.rs` | 189 | PolicyConfig, DeviceConfig, GeneratedTactic, Embedding + 8 tests |
| `crates/policy/src/tokenizer.rs` | 125 | LeanTokenizer: load, encode, decode, truncate + 4 tests |
| `crates/policy/src/llama.rs` | 619 | Forked Llama: Config, DeepSeekConfig, Cache, attention, model + 4 tests |
| `crates/policy/src/model.rs` | 371 | TacticGenerator: load, generate, encode, sample_top_p + 5 tests |
| `crates/policy/tests/integration.rs` | 227 | 10 integration tests with TinyLlama setup |
| `docs/phase3_instructions.md` | — | Phase 3 implementation instructions |
| `docs/phase4_instructions.md` | — | Phase 4 implementation instructions |
| `docs/phase5_instructions.md` | — | Phase 5 implementation instructions |
