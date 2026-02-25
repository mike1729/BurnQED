# SGLang Batch Hidden States Bug & Encoding Workaround

## The Bug

SGLang's `return_hidden_states=True` is broken in batch mode. When multiple
prompts are sent in a single `Engine.async_generate()` call, only the first
prompt gets valid hidden states — the rest return all-zero vectors.

This is SGLang Issue #8066 / #4997. The hidden states get "randomly stacked
to one or few outputs" rather than correctly assigned to each prompt.

### Impact

Our embedding cache for EBM training had **14.6% zero vectors** (31,815 / 217,499
states). The EBM was training on garbage — positive and negative states alike
got energy ~0, making the model a random guesser with an energy gap of only 0.113.

### What We Tried

| Approach | Result |
|----------|--------|
| `chunked_prefill_size=-1` | No effect — zeros persist |
| `asyncio.gather` with individual calls | SGLang merges them internally — same bug |
| Sequential (one at a time) | Works correctly, ~35 states/sec |
| vLLM alternative | Can't serve generation + encoding from same instance |

## The Solution

**Direct PyTorch encoding** via `python/encode_embeddings.py`:

- Loads the model directly with HuggingFace `AutoModelForCausalLM`
- Runs batch `model.forward()` with `output_hidden_states=True`
- Mean-pools last hidden layer over non-padding tokens
- Writes Parquet cache compatible with Rust `EmbeddingCache`

This is used for:
1. **Training cache** (pre-computed before EBM training in `run_iteration_train.sh`)
2. **Search-time encoding** still uses SGLang sequential (one prompt at a time),
   which is correct. Search latency is dominated by LLM generation (538% of wall
   time) vs EBM encoding (72%), so sequential encoding is acceptable.

### Usage

```bash
# Kill SGLang first to free VRAM
tmux kill-session -t sglang

# Encode embeddings (~15-30 min for 217K states)
python python/encode_embeddings.py \
    --model-path models/llm/iter_3_new \
    --trajectories trajectories/iter_0.parquet trajectories/iter_1.parquet \
    --output checkpoints/ebm/iter_4/embeddings.parquet \
    --batch-size 32

# Restart SGLang for search
./scripts/start_inference_server.sh models/llm/iter_3_new
```

### Pipeline Integration

Encoding is integrated into `scripts/run_iteration_train.sh` as Step 4
(after LLM export, before EBM training). The server is stopped, embeddings
are encoded directly, then the server is restarted for post-training eval.

EBM training (previously Step 2 of `run_iteration_search.sh`) is now Step 5
of `run_iteration_train.sh`, immediately after encoding. This keeps the
training pipeline self-contained: LLM train → export → encode → EBM train.

## Search-Time Encoding

During proof search, the SGLang server handles encoding sequentially (one
prompt per `async_generate` call). This avoids the batch bug.

From iter_2 search logs:
- LLM generation: p50 = 3.7s per batch (538% of wall time)
- EBM scoring: p50 = 291ms per batch (72% of wall time)

Even if sequential encoding doubles EBM latency, generation still dominates.
