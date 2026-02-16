# SGLang Inference Backend

## Context

Baseline runs show candle in-process inference is the bottleneck: ~50 tok/s, ~40s per 4-candidate batch. SGLang achieves 2000-5000 tok/s with continuous batching + PagedAttention. Research findings:

- **vLLM**: No hidden states from generation models. Two RFCs open, neither merged. **Ruled out.**
- **SGLang**: `return_hidden_states=True` merged (PR #3897). Returns per-token last-layer hidden states for mean-pooling into EBM embeddings. Known limitation: async engine (HTTP server) may not support it yet (Issue #6528, PR #6605 pending). Offline `Engine.generate()` works reliably.

Requirement: **encoding for EBM must work immediately** — cannot defer hidden states support.

## Goal

Add SGLang HTTP client as an alternative inference backend, enabling 50-100x faster generation while preserving hidden states for EBM encoding. Keep candle as a local fallback.

## Architecture

```
                         ┌─────────────────────────┐
                         │   SGLang Server (GPU)    │
                         │  DeepSeek-Prover-V2-7B   │
                         │                         │
                         │  /generate              │
                         │    - tactic generation   │
                         │    - hidden states       │
                         └────────┬────────────────┘
                                  │ HTTP
                         ┌────────┴────────────────┐
                         │   InferenceHandle::      │
                         │   Remote(SglangClient)   │
                         └────────┬────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
   InferencePolicyProvider   encode closures        health_check
   (PolicyProvider trait)    (single + batch)       (startup probe)
            │                     │
      SearchEngine           EBMScorer → EBMValueFn
      (deferred batch          (score_batch via
       EBM scoring)            batch_encode_fn)
```

Alternatively, use `InferenceHandle::Local(GenerationServiceHandle)` for candle (existing path, unchanged).

## Server-Side Batch Generation

With `batch_expansion_size=8`, the search engine pops 8 frontier nodes per iteration. Each needs `n` tactic candidates (default 8). Instead of firing `8×8=64` separate HTTP requests, `SglangClient::generate_candidates_batch()` sends a **single** `BatchGenerateRequest` containing all `states×n` prompts replicated into a flat array.

**How it works:**

1. For each proof state, the prompt is formatted via `format_prompt()` and replicated `n` times
2. The flat prompt array is sent as one POST to `/generate` (SGLang's `BatchGenerateRequest` format — `text: Vec<String>`)
3. SGLang's RadixAttention automatically caches shared prompt prefixes, so identical prompts within a state share KV cache
4. The response (a JSON array of `states×n` items) is unflatten back into per-state groups, deduplicated, and sorted by log_prob

**Call chain:**

```
SearchEngine::search_one()
  → policy.generate_candidates_batch(&prompts, n)    // PolicyProvider trait
    → InferencePolicyProvider delegates to:
      → InferenceHandle::generate_candidates_batch()
        → SglangClient::generate_candidates_batch()  // Single HTTP POST
```

**Timeout:** 300 seconds (same as encode batch), handling large batches like 8×8=64 sequences. Response length is validated against `states.len() * n`.

## Approach

### Step 1: Add dependencies + feature gate

**Files:** `Cargo.toml` (workspace), `crates/policy/Cargo.toml`, `crates/prover-core/Cargo.toml`

Add workspace deps:
```toml
reqwest = { version = "0.12", features = ["json"] }
url = "2"
```

In policy crate, add feature-gated deps:
```toml
[features]
sglang = ["dep:reqwest", "dep:url"]

[dependencies]
reqwest = { version = "0.12", features = ["json"], optional = true }
url = { version = "2", optional = true }
```

In prover-core, forward the feature:
```toml
[features]
sglang = ["policy/sglang"]
```

### Step 2: Make `format_tactic_message` and `extract_first_tactic` public

**File:** `crates/policy/src/model.rs`

Change both from `fn` to `pub fn`. Re-export from `lib.rs`. The SGLang client reuses these for identical prompt formatting and tactic extraction.

### Step 3: Create `SglangClient` HTTP client (core new code)

**New file:** `crates/policy/src/sglang.rs` (~250 lines)

```rust
pub struct SglangConfig {
    pub server_url: String,
    pub temperature: f64,
    pub top_p: f64,
    pub max_tactic_tokens: usize,
    pub num_candidates: usize,
    pub hidden_size: usize,  // 4096 for DeepSeek-Prover-V2-7B
}

pub struct SglangClient {
    client: reqwest::Client,
    base_url: url::Url,
    config: SglangConfig,
}
```

**`generate_candidates(&self, proof_state: &str, n: usize) -> Result<Vec<GeneratedTactic>>`:**
1. Build prompt: `format_tactic_message(proof_state)` wrapped in DeepSeek chat template
   ```
   <｜begin▁of▁sentence｜><｜User｜>{message}<｜Assistant｜>
   ```
2. POST to `{base_url}/generate`:
   ```json
   {
     "text": "<formatted_prompt>",
     "sampling_params": {
       "temperature": 0.6, "top_p": 0.95,
       "max_new_tokens": 128, "n": 4,
     }
     "return_logprob": true
   }
   ```
   **Note:** `return_logprob` is a **top-level field**, not inside `sampling_params`.
3. Parse response: each candidate has `meta_info.output_token_logprobs` — a list of
   `(logprob, token_id, None)` tuples. Compute `log_prob = sum(entry[0] for entry)`.
4. Apply `extract_first_tactic()` on each, deduplicate, sort by log_prob descending

**`encode(&self, text: &str) -> Result<Embedding>`:**
1. Same prompt formatting as generation
2. POST to `{base_url}/generate`:
   ```json
   {
     "text": "<formatted_prompt>",
     "sampling_params": { "max_new_tokens": 1, "temperature": 0.0 },
     "return_hidden_states": true
   }
   ```
   **Note:** `return_hidden_states` is a **top-level field**, not inside `sampling_params`.
   Server must be launched with `--enable-return-hidden-states`.
3. Parse `meta_info.hidden_states` — shape `(1, num_tokens, 4096)`.
   Index `[0]` to remove batch dim → `(num_tokens, 4096)`.
4. Mean-pool across axis 0 → `Embedding { data: Vec<f32>, dim: 4096 }`

**`health_check(&self) -> Result<()>`:**
- Called at startup when `--server-url` is provided
- Sends minimal generation request to verify server is reachable
- Fails fast with clear error if unreachable

**`test_hidden_states_support(&self) -> Result<bool>`:**
- Probe: send short request with `return_hidden_states: true`
- Returns false if server omits hidden states (Issue #6528)
- Used by pipeline to decide whether to enable EBM

**Error handling:** `SglangError` enum (HttpError, ServerError, HiddenStatesUnsupported, InvalidResponse). Retry up to 3x with exponential backoff on HTTP 5xx. Connection timeout 5s, request timeout 120s.

### Step 4: Create `InferenceHandle` enum

**New file:** `crates/policy/src/handle.rs` (~80 lines)

```rust
#[derive(Clone)]
pub enum InferenceHandle {
    Local(GenerationServiceHandle),
    #[cfg(feature = "sglang")]
    Remote(Arc<SglangClient>),
}
```

Mirrors `GenerationServiceHandle`'s interface: `generate_candidates()`, `generate_candidates_blocking()`, `encode()`, `encode_blocking()`. Each method dispatches via match. Blocking variants use `block_in_place` (same pattern as existing code).

### Step 5: Add `InferencePolicyProvider` adapter

**File:** `crates/search/src/adapters.rs`

New adapter alongside existing ones (they stay untouched):
```rust
pub struct InferencePolicyProvider {
    handle: InferenceHandle,
}
impl PolicyProvider for InferencePolicyProvider { ... }
```
Calls `handle.generate_candidates_blocking()`. Also exposes `handle()` for EBM encode closures.

### Step 6: Update CLI — add `--server-url`

**File:** `crates/prover-core/src/main.rs`

Add to both `Search` and `Eval` variants:
```rust
#[arg(long)]
server_url: Option<String>,

#[arg(long, required_unless_present = "server_url")]
model_path: Option<PathBuf>,
```

Wire through to `SearchArgs` / `EvalArgs`.

### Step 7: Update pipeline — branch on server_url

**File:** `crates/prover-core/src/pipeline.rs`

In `load_policy_and_ebm()`, add `server_url: Option<&str>` parameter. Branch:

**Remote path** (when `server_url` is Some):
1. Create `SglangClient::new(config).await`
2. `health_check()` — fail fast if unreachable
3. If `ebm_path` is set, call `test_hidden_states_support()`
4. Wrap in `InferenceHandle::Remote(Arc::new(client))`

**Local path** (existing, when `model_path` is Some):
1. Load `PolicyConfig` + `TacticGenerator`
2. `spawn_generation_service(generator)`
3. Wrap in `InferenceHandle::Local(service_handle)`

Both paths produce `InferenceHandle` → `InferencePolicyProvider` → same EBM encode closure pattern.

### Step 8: SGLang server launch script

**New file:** `scripts/start_sglang.sh`

```bash
#!/bin/bash
MODEL_PATH="${1:-deepseek-ai/DeepSeek-Prover-V2-7B}"
PORT="${PORT:-30000}"
TP="${TENSOR_PARALLEL:-1}"
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" --port "$PORT" --tp "$TP" \
    --trust-remote-code --mem-fraction-static 0.85 \
    --enable-return-hidden-states
```

### Step 9: Update shell scripts for `SGLANG_URL`

**Files:** `scripts/run_baseline.sh`, `scripts/lean_start.sh`

```bash
if [ -n "$SGLANG_URL" ]; then
    INFERENCE_FLAGS="--server-url $SGLANG_URL"
else
    INFERENCE_FLAGS="--model-path $MODEL_PATH"
fi
```

## Files Affected

| File | Action | Description |
|------|--------|-------------|
| `Cargo.toml` (workspace) | Modify | Add reqwest, url workspace deps |
| `crates/policy/Cargo.toml` | Modify | Add `sglang` feature with reqwest, url |
| `crates/policy/src/model.rs` | Modify | Make `format_tactic_message`, `extract_first_tactic` pub |
| `crates/policy/src/sglang.rs` | **New** | SglangClient, SglangConfig, HTTP methods |
| `crates/policy/src/handle.rs` | **New** | InferenceHandle enum (Local/Remote) |
| `crates/policy/src/lib.rs` | Modify | Add modules + re-exports |
| `crates/search/src/adapters.rs` | Modify | Add InferencePolicyProvider |
| `crates/search/src/lib.rs` | Modify | Re-export InferencePolicyProvider |
| `crates/prover-core/src/main.rs` | Modify | Add --server-url, make --model-path optional |
| `crates/prover-core/src/pipeline.rs` | Modify | Branch on server_url in load_policy_and_ebm() |
| `crates/prover-core/Cargo.toml` | Modify | Forward sglang feature |
| `scripts/start_sglang.sh` | **New** | Server launch script |
| `scripts/run_baseline.sh` | Modify | SGLANG_URL env var support |
| `scripts/lean_start.sh` | Modify | SGLANG_URL env var support |

## Verified API (from python/verify_sglang_hidden_states.py)

Tested with SGLang 0.5.8 + DeepSeek-Prover-V2-7B on GPU.

| Feature | Status | Details |
|---------|--------|---------|
| Hidden states (offline Engine) | **Works** | Shape `(1, num_tokens, 4096)`. Squeeze `[0]`, mean-pool axis=0 → `(4096,)` |
| Mean-pooled embedding | **Works** | Norm ~18, matches expected range |
| Multi-candidate generation | **Works** | `n=4` in sampling_params, returns 4 completions |
| Log probabilities | **Works** | `return_logprob=True` top-level kwarg. `output_token_logprobs`: list of `(logprob, token_id, None)` tuples |
| HTTP server hidden states | **Not tested** | Requires separate server launch with `--enable-return-hidden-states` |

**Key API notes:**
- `return_logprob` and `return_hidden_states` are **top-level kwargs** on `engine.generate()`, NOT inside `sampling_params`
- Server requires `--enable-return-hidden-states` flag at launch (or `enable_return_hidden_states=True` for offline Engine)
- Hidden states include a batch dimension: always index `[0]` before processing

## Edge Cases & Risks

1. **Hidden states not available over HTTP (HIGH).** SGLang Issue #6528 — async engine may not support `return_hidden_states`. Mitigation: `test_hidden_states_support()` probe at startup.

2. **Prompt format drift.** SGLang tokenizes differently from candle. Mitigation: send raw text with special tokens to native `/generate` (bypass server chat template).

3. **Log-prob format differences.** SGLang's `output_token_logprobs` may differ. Mitigation: careful parsing + integration tests.

4. **Server unavailability.** HTTP server could die mid-search. Mitigation: retry with backoff + existing per-theorem error recovery.

5. **Double model load impossible.** Only one process can hold the GPU model. Mitigation: `InferenceHandle` enum — choose one at startup.

6. **Batched hidden states bugs (Issue #8066).** Mitigation: encode requests are always batch_size=1.

## Verification Script

Before implementing, run this on a GPU machine with SGLang installed to verify the hidden states output format:

```python
"""
Verify SGLang hidden states format for BurnQED EBM integration.

Run on a GPU machine:
    pip install "sglang[all]"
    python python/verify_sglang_hidden_states.py
"""
import sglang as sgl
import numpy as np

MODEL = "deepseek-ai/DeepSeek-Prover-V2-7B"

print(f"Loading {MODEL}...")
engine = sgl.Engine(model_path=MODEL)

# --- Test 1: Hidden states shape ---
print("\n=== Test 1: Hidden states from offline Engine ===")
out = engine.generate(
    ["test input"],
    sampling_params={"max_new_tokens": 1},
    return_hidden_states=True,
)
hs = out[0]["meta_info"]["hidden_states"]
print(f"type: {type(hs)}")
if isinstance(hs, list):
    print(f"len: {len(hs)}")
    if len(hs) > 0:
        first = hs[0]
        print(f"  element type: {type(first)}")
        if isinstance(first, list):
            print(f"  element len: {len(first)} (expected 4096)")
        elif hasattr(first, 'shape'):
            print(f"  element shape: {first.shape}")
elif hasattr(hs, 'shape'):
    print(f"shape: {hs.shape}")

# --- Test 2: Mean-pooling ---
print("\n=== Test 2: Mean-pooling ===")
arr = np.array(hs, dtype=np.float32)
print(f"numpy shape: {arr.shape}")  # Expected: (num_tokens, 4096)
embedding = arr.mean(axis=0)
print(f"embedding shape: {embedding.shape}")  # Expected: (4096,)
print(f"embedding norm: {np.linalg.norm(embedding):.4f}")
print(f"embedding[:5]: {embedding[:5]}")

# --- Test 3: Proof state prompt ---
print("\n=== Test 3: Proof state hidden states ===")
proof_state = "n : Nat\n⊢ n + 0 = n"
prompt = f"Complete the following Lean 4 code:\n\n```lean4\n/- tactic state:\n{proof_state}\n-/\n```"
out2 = engine.generate(
    [prompt],
    sampling_params={"max_new_tokens": 1},
    return_hidden_states=True,
)
hs2 = out2[0]["meta_info"]["hidden_states"]
arr2 = np.array(hs2, dtype=np.float32)
print(f"proof state hidden states shape: {arr2.shape}")
emb2 = arr2.mean(axis=0)
print(f"proof state embedding shape: {emb2.shape}")
print(f"proof state embedding norm: {np.linalg.norm(emb2):.4f}")

# --- Test 4: Generation with logprobs ---
print("\n=== Test 4: Generation with logprobs ===")
out3 = engine.generate(
    [prompt],
    sampling_params={
        "max_new_tokens": 64,
        "temperature": 0.6,
        "top_p": 0.95,
        "n": 4,
        "return_logprob": True,
        "logprob_start_len": -1,
    },
)
for i, item in enumerate(out3):
    text = item["text"]
    meta = item["meta_info"]
    logprobs = meta.get("output_token_logprobs", [])
    print(f"\n  Candidate {i}:")
    print(f"    text: {repr(text[:80])}")
    print(f"    logprobs type: {type(logprobs)}, len: {len(logprobs) if isinstance(logprobs, list) else 'N/A'}")
    if logprobs and len(logprobs) > 0:
        print(f"    first logprob entry: {logprobs[0]}")
        total_lp = sum(lp[0] if isinstance(lp, (list, tuple)) else lp for lp in logprobs)
        print(f"    total log_prob: {total_lp:.4f}")

# --- Test 5: HTTP server hidden states (if running) ---
print("\n=== Test 5: HTTP server check ===")
try:
    import requests
    resp = requests.post(
        "http://localhost:30000/generate",
        json={
            "text": "test",
            "sampling_params": {"max_new_tokens": 1},
            "return_hidden_states": True,
        },
        timeout=10,
    )
    data = resp.json()
    hs_http = data.get("meta_info", {}).get("hidden_states")
    if hs_http is not None:
        arr_http = np.array(hs_http, dtype=np.float32)
        print(f"HTTP hidden states shape: {arr_http.shape}")
        print("HTTP server SUPPORTS return_hidden_states!")
    else:
        print("HTTP hidden_states is None — server does NOT support it (Issue #6528)")
        print(f"meta_info keys: {list(data.get('meta_info', {}).keys())}")
except Exception as e:
    print(f"HTTP test skipped (server not running or error): {e}")

engine.shutdown()
print("\nDone. Use shapes above to adjust /encode indexing in SglangClient.")
```

## Testing Strategy

**Unit tests (no server):**
- Prompt formatting matches candle's detokenized output
- Response parsing for generation (mock JSON)
- Response parsing for hidden states (mock JSON)
- extract_first_tactic applied correctly
- Dedup and log_prob accumulation
- Error handling (HTTP errors, missing fields)
- InferenceHandle is Clone + Send + Sync
- InferencePolicyProvider satisfies PolicyProvider bounds
- Batch generate request serialization (prompt replication → flat array)
- Batch generate response parsing/unflatten (grouping, dedup, sort)
- Batch generate empty input returns empty output

**Integration tests (#[ignore], require SGLang server, ~5 tests):**
- Generate candidates, verify GeneratedTactic structure
- Encode proof state, verify embedding dim = 4096
- test_hidden_states_support() probe
- Full pipeline: SGLang + Lean pool, search simple theorem
- EBM scoring with SGLang embeddings

**Manual verification:**
1. `./scripts/start_sglang.sh` on GPU host
2. `cargo run -p prover-core --features sglang -- search --server-url http://localhost:30000 --theorems data/test_theorems.json --output test.parquet`
3. Compare prove rate with candle baseline
4. Verify wall-clock speedup (target: 50-100x per generation batch)
