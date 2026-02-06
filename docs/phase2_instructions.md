# Phase 2: LLM Inference in candle — Claude Code Instructions

Step-by-step prompts for `claude` CLI. Phase 1 (lean-repl) is complete and working.

Phase 2 builds the `policy` crate: loading DeepSeek-Prover-V2-7B in candle, generating tactics via autoregressive decoding, and extracting hidden-state embeddings via `encode_only()` for the EBM value function.

## Prerequisites

Before starting:
- Phase 1 complete (`cargo test -p lean-repl` passes, integration tests pass)
- DeepSeek-Prover-V2-7B weights downloaded locally:
  ```bash
  # ~14GB download. Uses HuggingFace hub.
  pip install huggingface_hub
  huggingface-cli download deepseek-ai/DeepSeek-Prover-V2-7B --local-dir models/deepseek-prover-v2-7b
  # Or if gated: huggingface-cli login first
  ```
- Verify the download contains: `config.json`, `tokenizer.json`, `model*.safetensors`
- For local Mac testing: candle's Metal backend works on Apple Silicon. Inference will be slow (~10-30 tokens/sec) but functional for testing with short inputs.

## Background: DeepSeek-Prover-V2-7B Architecture

DeepSeek-Prover-V2-7B is a Llama-family model. Key specs from `config.json`:
- `model_type`: likely `"llama"` or `"deepseek"` (check the actual file)
- `hidden_size`: 4096
- `num_hidden_layers`: 28-32 (check actual config)
- `num_attention_heads`: 32
- `num_key_value_heads`: possibly fewer (GQA — grouped query attention)
- `intermediate_size`: 11008 or similar
- `vocab_size`: 32000 or similar
- `max_position_embeddings`: 4096+
- Uses RoPE positional embeddings
- Uses SiLU activation in FFN

candle-transformers has a Llama implementation at `candle_transformers::models::llama`. If the model config says `"model_type": "llama"`, we can use it directly. If it says something DeepSeek-specific, we may need to check if candle has a DeepSeek model or adapt the Llama loader.

**First thing to do in Phase 2: inspect the actual `config.json` to determine which candle model class to use.**

---

## Prompt 2.1 — Inspect model files and plan approach

```
We're starting Phase 2: loading DeepSeek-Prover-V2-7B in candle for tactic generation.

First, inspect the downloaded model to understand what we're working with:

1. Read models/deepseek-prover-v2-7b/config.json and report:
   - model_type
   - hidden_size
   - num_hidden_layers
   - num_attention_heads
   - num_key_value_heads (GQA?)
   - intermediate_size
   - vocab_size
   - max_position_embeddings
   - rope_theta (if present)
   - torch_dtype
   - Any DeepSeek-specific fields (like MoE config, expert counts, etc.)

2. List all .safetensors files and their sizes.

3. Read models/deepseek-prover-v2-7b/tokenizer.json — report the tokenizer type (BPE? SentencePiece?) and vocab size.

4. Check candle-transformers source code (in our Cargo dependencies or on GitHub) to see:
   - Does it have a Llama model? What's the module path?
   - Does it have a DeepSeek-specific model?
   - What model types does candle-transformers support for text generation?
   - How does the existing Llama model handle weight loading from safetensors?

5. Based on the above, determine:
   - Can we use candle_transformers::models::llama directly?
   - Or do we need a different model, or a custom wrapper?
   - Are there any GQA/MoE/RoPE compatibility concerns?

Report your findings. Do NOT write any code yet — just investigate and plan.
```

### Prompt 2.2 — PolicyConfig and types

```
Based on the model inspection from 2.1, implement crates/policy/src/types.rs and update the crate's Cargo.toml.

Cargo.toml for the policy crate needs:
- candle-core (workspace)
- candle-nn (workspace)
- candle-transformers (workspace)
- safetensors (workspace)
- tokenizers (workspace)
- serde, serde_json (workspace)
- anyhow (workspace)
- tracing (workspace)

Create src/types.rs with:

1. PolicyConfig struct (Deserialize, Debug, Clone):
   - model_path: PathBuf — path to model directory (contains config.json + safetensors)
   - max_seq_len: usize — maximum sequence length (default 2048)
   - num_candidates: usize — number of tactic candidates to generate per state (default 32)
   - temperature: f64 — sampling temperature (default 0.8)
   - top_p: f64 — nucleus sampling threshold (default 0.95)
   - max_tactic_tokens: usize — max tokens per generated tactic (default 128)
   - device: DeviceConfig — CPU, Metal, or Cuda(id)

2. DeviceConfig enum (Deserialize):
   - Cpu
   - Metal
   - Cuda(usize)
   With a method to_candle_device() -> candle_core::Device

3. GeneratedTactic struct:
   - text: String — the generated tactic string
   - log_prob: f64 — sum of log-probabilities of the generated tokens
   - tokens: Vec<u32> — the token IDs generated

4. Embedding struct:
   - data: Vec<f32> — the pooled hidden state vector
   - dim: usize — dimensionality (should be hidden_size, e.g. 4096)

Make sure `cargo check -p policy` passes.
```

### Prompt 2.3 — Tokenizer wrapper

```
Implement crates/policy/src/tokenizer.rs — a wrapper around the HuggingFace tokenizers crate.

This wraps tokenizers::Tokenizer with convenience methods for our use case.

LeanTokenizer struct:
  - inner: tokenizers::Tokenizer

Methods:

1. fn load(model_path: &Path) -> Result<Self>
   - Load from model_path/tokenizer.json
   - If that doesn't exist, try model_path/tokenizer.model (SentencePiece)
   - Return an error with a helpful message if neither exists

2. fn encode(&self, text: &str) -> Result<Vec<u32>>
   - Encode text, return token IDs
   - Do NOT add special tokens here (the caller decides)

3. fn encode_with_special(&self, text: &str) -> Result<Vec<u32>>
   - Encode with special tokens (BOS, etc.) — for generation prompts

4. fn decode(&self, ids: &[u32]) -> Result<String>
   - Decode token IDs back to string
   - Skip special tokens in output

5. fn vocab_size(&self) -> usize

6. fn bos_token_id(&self) -> Option<u32>
   - Return the BOS token ID if one is configured

7. fn eos_token_id(&self) -> Option<u32>
   - Return the EOS token ID. IMPORTANT: we need this to know when to stop generation.

8. fn truncate(&self, ids: &[u32], max_len: usize) -> Vec<u32>
   - Truncate to max_len tokens
   - For proof states: try to keep the goal (after ⊢) and truncate hypotheses.
   - Simple approach for now: just truncate from the end. We can add smart truncation later.

Add unit tests:
- Load the tokenizer from the model directory (use an env var MODEL_PATH, skip if not set)
- Encode a simple Lean proof state like "n : Nat\n⊢ n + 0 = n" and verify it produces tokens
- Round-trip: encode then decode should recover approximately the original text
- Verify BOS and EOS token IDs are found
- Verify truncation works correctly

Mark tests that need the actual model files as #[ignore].
```

### Prompt 2.4 — Model loading

```
Implement the core model loading in crates/policy/src/model.rs.

This is the most complex part of Phase 2. Read CLAUDE.md and docs/spindle_final_plan.md Section 4.4 before starting.

Create a TacticGenerator struct that wraps the candle Llama model.

IMPORTANT: Study how candle-transformers loads Llama models. Look at:
- candle-transformers/src/models/llama.rs (or the specific DeepSeek variant)
- The candle examples for text generation (examples/llama/ in the candle repo)
- How weights are loaded from safetensors files
- How the model config (from config.json) maps to the candle model config

The general candle pattern for loading a Llama model is:
1. Parse config.json → model config struct
2. Load safetensors files as a VarBuilder
3. Construct the model struct from config + VarBuilder
4. The model has a forward() method that takes input_ids and returns logits

TacticGenerator struct fields:
- model: candle_transformers::models::llama::Llama (or equivalent)
- tokenizer: LeanTokenizer
- config: PolicyConfig
- device: candle_core::Device
- hidden_size: usize — cached from model config (4096 for 7B)

Methods to implement in this prompt (just loading and basic forward, NOT generation yet):

1. fn load(config: &PolicyConfig) -> Result<Self>
   - Determine candle Device from config.device
   - Load tokenizer from config.model_path
   - Parse config.json from config.model_path to get model hyperparameters
   - Load safetensors weights: use candle_nn::VarBuilder::from_safetensors(...)
     or the standard candle pattern for multi-file safetensors
   - Construct the Llama model
   - Log model info: tracing::info!(hidden_size, num_layers, vocab_size, "Model loaded")
   - Return TacticGenerator

2. fn forward_logits(&self, input_ids: &candle_core::Tensor) -> Result<candle_core::Tensor>
   - Run a single forward pass through the full model (including LM head)
   - Input: (batch, seq_len) tensor of token IDs
   - Output: (batch, seq_len, vocab_size) logits tensor
   - This is a thin wrapper around the candle model's forward method

Handle potential issues:
- Multi-shard safetensors: DeepSeek-7B may split across multiple files.
  Use candle's unsafe_load_multi pattern or iterate over all .safetensors files.
- config.json field naming: candle may expect specific field names. Check if there's a
  conversion needed between HF config format and candle's expected format.
- dtype: Load in f16 (half precision) for efficiency. candle supports this natively.

Add a test (marked #[ignore]) that:
- Loads the model from MODEL_PATH env var
- Creates a dummy input tensor of 5 tokens
- Runs forward_logits and checks the output shape is (1, 5, vocab_size)
- Prints timing info

If the model loading fails due to config format or weight naming issues, document what went wrong clearly so we can debug.
```

### Prompt 2.5 — Autoregressive tactic generation

```
Add autoregressive tactic generation to TacticGenerator in crates/policy/src/model.rs.

Read the candle examples for text generation to understand the sampling pattern.

Implement these methods:

1. fn generate_one(&self, prompt: &str) -> Result<GeneratedTactic>
   - Tokenize the prompt (with special tokens)
   - Truncate to max_seq_len if needed
   - Run autoregressive decoding:
     a. Forward pass to get logits for the last token position
     b. Apply temperature: logits = logits / temperature
     c. Apply top-p (nucleus) sampling:
        - Sort logits descending
        - Compute cumulative softmax probabilities
        - Zero out tokens beyond the top_p cumulative threshold
        - Sample from the remaining distribution
     d. Append sampled token to the sequence
     e. Record log_prob of the sampled token
     f. Stop if EOS token is generated or max_tactic_tokens reached
   - Decode generated tokens (excluding the prompt) back to text
   - Return GeneratedTactic { text, log_prob, tokens }

   IMPORTANT for efficiency: candle Llama models use a KV cache internally.
   After the initial prompt forward pass, subsequent tokens should use the cache
   so we only process one token at a time, not the full sequence. Check how the
   candle Llama model handles this — there's likely a cache mechanism or a
   `forward_one` / `forward_with_cache` method. If so, use it:
   - First forward: process entire prompt, get logits for last position
   - Subsequent forwards: process single new token, get next logits

2. fn generate_candidates(&self, proof_state: &str, n: usize) -> Result<Vec<GeneratedTactic>>
   - Build the prompt for tactic generation. The format should be:
     ```
     [GOAL]
     {proof_state}
     [TACTIC]
     ```
     Check DeepSeek-Prover-V2's actual prompt format — it may use a different template.
     If unsure, use the simple format above.
   - Generate n tactic candidates independently
   - Sort by log_prob descending (best first)
   - PERF NOTE: Each candidate requires a separate forward pass through the model.
     For n=32, this means 32 sequential generation passes. On GPU this is ~500ms total.
     On CPU it will be very slow — acceptable for testing, not for production.
   - Consider: we could batch the initial prompt encoding and then branch into
     separate generation streams, but this is complex. Skip batching for now.

3. fn build_prompt(&self, proof_state: &str) -> String
   - Format the proof state into the model's expected prompt format
   - Handle multi-goal states (multiple goals separated by newlines)
   - Truncate if too long: keep the first goal's target, truncate hypotheses

Add tests (#[ignore], need model files):
- Generate 1 tactic for a simple proof state "⊢ ∀ (n : Nat), n = n"
  Verify it returns a non-empty string
- Generate 5 candidates, verify they're sorted by log_prob
- Generate for a state with hypotheses "n : Nat\n⊢ n + 0 = n"
- Print all generated tactics and their log-probs for manual inspection
- Time the generation and report tokens/sec
```

### Prompt 2.6 — encode_only() for EBM embeddings

```
Add the encode_only() method to TacticGenerator. This is CRITICAL for the EBM integration.

Read docs/spindle_final_plan.md Section 4.4 carefully.

The idea: run the 7B model as an encoder only. Forward through all transformer blocks and the final layer norm, but STOP before the LM head. Mean-pool the hidden states to get a single vector per input.

This requires accessing the model's internal layers, which candle-transformers may or may not expose directly.

APPROACH — Investigate and implement one of these:

Option A (preferred): candle's Llama model exposes its components.
  If the Llama struct has public fields or methods to access:
  - embed_tokens (embedding layer)
  - layers (transformer blocks)  
  - norm (final RMSNorm/LayerNorm)
  Then we can call them directly and skip the lm_head.

Option B: The model has a method that returns hidden states.
  Some candle model implementations have a `forward_hidden()` or similar.
  Check the actual API.

Option C: We must fork/wrap the forward method.
  If the model only exposes a monolithic `forward()` that goes straight to logits,
  we need to either:
  - Copy and modify the forward method to return hidden states
  - Or add a hook/wrapper

Option D (fallback): Apply the LM head and then project back.
  This is wasteful but works as a last resort. Not recommended.

Implement:

1. fn encode_only(&self, text: &str) -> Result<Embedding>
   - Tokenize the text (without generation prompt wrapper)
   - Create input tensor
   - Forward through transformer blocks + final norm, get hidden states
     Shape: (1, seq_len, hidden_size)
   - Mean pooling: average across the seq_len dimension
     Shape: (1, hidden_size) → squeeze to (hidden_size,)
   - Convert to Vec<f32>
   - Return Embedding { data, dim: self.hidden_size }

2. fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>>
   - Tokenize all texts
   - Pad to same length within the batch
   - Create attention mask (1 for real tokens, 0 for padding)
   - Forward through transformer blocks + norm
   - Mean pooling with attention mask:
     masked_hidden = hidden_states * mask.unsqueeze(-1)
     pooled = masked_hidden.sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)
   - Return Vec<Embedding>
   - If batched forward is too complex to implement initially, fall back to
     sequential calls to encode_only()

DOCUMENT clearly in code comments which option (A/B/C/D) you used and why.
If you had to modify or wrap candle-transformers internals, explain what you did.

Add tests (#[ignore]):
- encode_only("⊢ True") → verify embedding.dim == 4096 (or whatever hidden_size is)
- encode_only on two different states → verify embeddings are different
- encode_only on the same state twice → verify embeddings are identical (deterministic)
- Verify embedding values are not all zeros or NaN
- Time encode_only for a typical proof state (~200 tokens)
- If encode_batch works: test batch of 8, verify shapes match
```

### Prompt 2.7 — Integration with Lean REPL (the core loop)

```
Create crates/policy/tests/integration.rs — the CRITICAL integration test that validates the core search loop.

This test combines the policy crate (Phase 2) with lean-repl (Phase 1) into the end-to-end loop described in the plan:

  State → candle (generate tactics) → Lean (apply tactic) → new State → repeat

This is the most important milestone of the project. If this works, everything else is incremental.

Test: test_search_loop_simple_theorem

1. Setup:
   - Load TacticGenerator from MODEL_PATH
   - Create LeanPool with 2 workers (from LEAN_ENV_PATH)
   - Choose a simple theorem: "∀ (n : Nat), n + 0 = n"

2. Start proof:
   - pool.start_proof("∀ (n : Nat), n + 0 = n")
   - Get initial ProofState with goals

3. Loop (max 10 iterations):
   a. Take the first unsolved goal
   b. Build the proof state string from goal.raw
   c. Generate 8 tactic candidates via generator.generate_candidates(state_str, 8)
   d. For each candidate (sorted by log_prob, best first):
      - Apply tactic via pool.run_tactic(state_id, goal_id, &tactic.text)
      - If TacticResult::ProofComplete → print "PROVED!" and return success
      - If TacticResult::Success → record the new state, break to try this branch
      - If TacticResult::Failed → try next candidate
   e. If no candidate succeeded → print "stuck" and break

4. Print summary:
   - Tactics tried per step
   - Time per tactic generation
   - Time per Lean application
   - Total wall time
   - Whether the proof was found

Mark this test #[ignore] (requires model + Lean).

Test: test_encode_only_produces_distinct_embeddings

1. Load TacticGenerator
2. Encode three different proof states:
   - "⊢ ∀ (n : Nat), n + 0 = n"
   - "n : Nat\n⊢ n + 0 = n"
   - "⊢ True"
3. Verify all three embeddings have dim 4096
4. Compute cosine similarity between pairs
5. Verify that the first two are more similar to each other than either is to the third
   (they're about Nat arithmetic; the third is trivially different)

Test: test_generate_multiple_distinct_tactics

1. Load TacticGenerator
2. Generate 16 candidates for "n : Nat\n⊢ n + 0 = n"
3. Verify at least 3 distinct tactic strings (not all identical)
4. Print all 16 for manual inspection
5. Verify log_probs are monotonically non-increasing (sorted correctly)

Run with:
  MODEL_PATH=./models/deepseek-prover-v2-7b \
  LEAN_ENV_PATH=/path/to/mathlib4/.lake/build \
  cargo test -p policy -- --ignored --nocapture
```

### Prompt 2.8 — Public API and lib.rs

```
Wire up crates/policy/src/lib.rs:

pub mod model;
pub mod tokenizer;
pub mod types;

// Re-export main types at crate root
pub use model::TacticGenerator;
pub use tokenizer::LeanTokenizer;
pub use types::{PolicyConfig, DeviceConfig, GeneratedTactic, Embedding};

Then do a full review pass:

1. cargo check -p policy — fix any warnings
2. cargo clippy -p policy — fix all clippy lints
3. cargo doc -p policy --no-deps — verify docs generate cleanly
4. Review all public API surface:
   - Is every public method documented with /// comments?
   - Do doc comments explain what the method does, its inputs, and outputs?
   - Are error conditions documented?
5. Review error handling:
   - Are candle errors properly wrapped with context? Use .context("loading model weights")?
   - Are there any unwrap() calls that should be ? instead?
6. Review logging:
   - Model loading should log at info level (model path, hidden_size, num_layers, device)
   - Generation should log at debug level (prompt length, num candidates, time elapsed)
   - Errors should log at warn level before returning
```

### Prompt 2.9 — Performance baseline and device handling

```
Add device-aware model loading and a benchmark utility to the policy crate.

1. Update TacticGenerator::load() to handle all device types:
   - DeviceConfig::Cpu → candle_core::Device::Cpu
   - DeviceConfig::Metal → candle_core::Device::new_metal(0)?
     Handle the case where Metal is not available (not on Mac, or feature not compiled)
   - DeviceConfig::Cuda(id) → candle_core::Device::new_cuda(id)?
     Handle the case where CUDA is not available

   On load, log which device is being used and the dtype.
   Load weights in f16 when on GPU (Metal/CUDA). Use f32 on CPU if f16 is too slow
   or not supported.

2. Add dtype handling:
   - Metal and CUDA: load in f16 (DType::F16) for speed + memory
   - CPU: load in f32 (DType::F32) for compatibility
   - Make this configurable via PolicyConfig if needed

3. Create a simple benchmark in crates/policy/tests/benchmark.rs (#[ignore]):

   fn bench_generation_throughput:
   - Load model
   - Generate 1 tactic for a warmup pass
   - Time: generate 8 tactics for "n : Nat\n⊢ n + 0 = n"
   - Report: total time, time per tactic, tokens generated per second
   - Print device info

   fn bench_encode_only_throughput:
   - Load model
   - Warmup: encode_only once
   - Time: encode_only for 10 different proof states
   - Report: avg time per encode, embeddings per second
   - This number matters: during search, we call encode_only ~8 times per expansion.
     At 600 expansions, that's 4800 calls. If each takes 50ms → 4 minutes total (acceptable).
     If each takes 500ms → 40 minutes (too slow, need optimization).

4. Print a summary like:
   ```
   === burn-qed Policy Benchmark ===
   Device: Metal (Apple M2 Pro)
   Dtype: f16
   Model: DeepSeek-Prover-V2-7B (4096 hidden, 32 layers)

   Generation (8 candidates):
     Total: 12.3s
     Per tactic: 1.54s
     Tokens/sec: 23.4

   Encode-only (10 states):
     Total: 2.1s
     Per state: 210ms
     Embeddings/sec: 4.8
   ```

This gives us the baseline numbers to plan around. On cloud A100, expect 10-50x faster than Mac CPU.
```

### Prompt 2.10 — Update CLAUDE.md

```
Update the CLAUDE.md file:

1. Mark Phase 0 and Phase 1 as complete: [x]
2. Mark Phase 2 as complete: [x]
3. Update "Current Phase" to Phase 3

4. Add a "Phase 2 Results" section after "Phase Status" with the benchmark numbers from 2.9 (leave as TODO placeholders if benchmarks haven't been run yet):
   - Device tested
   - Generation throughput (tactics/sec)
   - Encode-only throughput (embeddings/sec)
   - Any model loading issues encountered and workarounds applied
   - Which approach was used for encode_only (Option A/B/C/D)

5. Add to the "Settled Architecture Decisions" section:
   - Note which candle model class is used (llama? deepseek? custom wrapper?)
   - Note the actual hidden_size from the loaded model
   - Note the prompt format used for tactic generation

6. Add a section "Cross-Crate Integration Notes":
   - How to get embeddings from policy for the EBM: `generator.encode_only(state) -> Embedding`
   - The Embedding.data is Vec<f32> of length 4096 — this becomes input to the burn-rs energy head
   - For search: `generator.generate_candidates(state, 32)` returns sorted Vec<GeneratedTactic>
   - The GeneratedTactic.log_prob is used for beam filtering (top-8) before Lean application
```

---

## Verification Checklist

After all Phase 2 prompts, verify:

```bash
# Policy crate compiles
cargo check -p policy

# Whole workspace still compiles
cargo check --workspace

# Unit tests pass (no model needed)
cargo test -p policy

# No clippy warnings
cargo clippy -p policy

# Integration tests pass (need MODEL_PATH + LEAN_ENV_PATH)
MODEL_PATH=./models/deepseek-prover-v2-7b \
LEAN_ENV_PATH=/path/to/mathlib4/.lake/build \
  cargo test -p policy -- --ignored --nocapture

# THE CRITICAL TEST — core search loop works:
MODEL_PATH=./models/deepseek-prover-v2-7b \
LEAN_ENV_PATH=/path/to/mathlib4/.lake/build \
  cargo test -p policy -- test_search_loop_simple_theorem --ignored --nocapture

# Benchmark numbers are reasonable:
MODEL_PATH=./models/deepseek-prover-v2-7b \
  cargo test -p policy -- bench_ --ignored --nocapture
```

### Success Criteria

1. **Model loads** without errors from safetensors on at least one device (CPU or Metal)
2. **Tactic generation** produces syntactically plausible Lean tactics (not gibberish)
3. **encode_only()** returns 4096-dim embeddings that differ between different proof states
4. **Core search loop** test completes — ideally finds a proof, but even if it doesn't, the loop runs without crashes
5. **Throughput baseline** documented — we know how fast generation and encoding are on local hardware

---

## Troubleshooting

### "model type not supported" or "unknown architecture"
DeepSeek-Prover-V2 may not be a pure Llama model. Check `config.json` for the `model_type` field. If it's `"deepseek"` or `"deepseek_v2"`, candle may not support it natively. Options:
- Check if candle-transformers has a DeepSeek model in recent versions
- If DeepSeek-Prover-V2-7B is actually just a fine-tuned Llama (common), the config might just need `model_type` changed to `"llama"`
- Worst case: look at the model architecture in Python (`from transformers import AutoModel; model = AutoModel.from_pretrained(...); print(model)`) and map it to the closest candle model

### Weight name mismatches
candle-transformers expects specific weight names in the safetensors files. If HuggingFace uses different naming:
- Compare `safetensors.metadata()` weight names with what candle expects
- May need a name-mapping layer in VarBuilder
- candle has `VarBuilder::rename` or similar for this

### "out of memory" on MacBook
7B model in f16 needs ~14GB. With f32 it's ~28GB. If your Mac has 32GB RAM:
- Use f16 on Metal
- If still OOM, try CPU with f32 but smaller batch size
- Or use a quantized version (GGUF) — but this requires candle's GGUF loader, which is a different code path. Skip for now, use full precision.

### Generation produces gibberish
- Check that the prompt format matches what the model was trained on. DeepSeek-Prover-V2 may expect a specific format like `<｜begin▁of▁sentence｜>` tokens.
- Check that BOS token is prepended correctly
- Check temperature — too high (>1.5) produces noise, too low (<0.1) produces repetition
- Try temperature=0 (greedy) first to verify the model works at all

### encode_only() returns all zeros
- The model might have a different internal structure than expected
- Check that you're accessing the right layer — after transformer blocks and norm, before lm_head
- Print intermediate shapes: the hidden states should be (1, seq_len, 4096)
- Verify the mean pooling math: sum over seq_len dimension, divide by seq_len

### KV cache issues during generation
- candle Llama models may require explicit cache management
- Check if there's a `Cache` struct that needs to be passed to forward()
- Some implementations need cache.reset() between different generation runs
- If cache is causing issues, start without it (reprocess full sequence each step — slow but correct), then optimize

### candle version incompatibility
If the candle version in Cargo.toml doesn't support the features we need:
- Check the candle GitHub for the latest version and any DeepSeek-specific PRs
- Pin to a specific commit if the latest release is too old
- Consider using the candle git dependency instead of crates.io version:
  `candle-core = { git = "https://github.com/huggingface/candle", rev = "..." }`

### The model is actually a MoE (Mixture of Experts)
DeepSeek-Prover-V2 has both a 7B dense and a 236B MoE variant. Verify you downloaded the 7B dense model, not the MoE. The 7B should have a simple config without `num_experts` or `expert_*` fields. If it IS an MoE even at 7B, candle may not support MoE — escalate this as a blocker.
