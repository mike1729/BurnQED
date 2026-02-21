"""
Custom inference server wrapping sgl.Engine for BurnQED.

Replaces SGLang's HTTP server to avoid the ~10MB JSON serialization bottleneck
for hidden states. Mean-pools hidden states in-process before any JSON
serialization, reducing /encode response from ~10MB to ~16KB.

Two endpoints:
  POST /generate — Drop-in replacement for SGLang's /generate
  POST /encode   — New: returns pre-pooled (hidden_size,) embedding
  GET  /health   — Returns 200

Usage:
  python python/inference_server.py --model-path models/deepseek-prover-v2-7b
  python python/inference_server.py --model-path deepseek-ai/DeepSeek-Prover-V2-7B --tp 2

Environment variables (override CLI args):
  PORT           — Server port (default: 30000)
  TP             — Tensor parallelism (default: 1)
  MEM_FRACTION   — Static memory fraction (default: auto-detected)
"""

import argparse
import asyncio
import logging
import sys
import time
from typing import Union

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger("inference_server")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Matches SGLang's /generate request format."""
    text: Union[str, list[str]]
    sampling_params: dict = Field(default_factory=dict)
    return_logprob: bool = False
    return_hidden_states: bool = False


class EncodeRequest(BaseModel):
    """Custom /encode endpoint for pre-pooled embeddings."""
    text: Union[str, list[str]]
    hidden_size: int = 4096


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

app = FastAPI(title="BurnQED Inference Server")

# Global state set during startup
_engine = None
# True if startup self-test confirms batch hidden states are correct
_batch_encode_ok: bool = False
# Semaphore(1) fallback when batch encode is broken (SGLang Issue #8066)
_encode_semaphore: asyncio.Semaphore = None


def _build_engine(model_path: str, tp: int = 1, mem_fraction: float = 0.85):
    """Build sgl.Engine with hidden states always enabled."""
    import sglang as sgl

    logger.info(
        "Loading model: %s (tp=%d, mem_fraction=%.2f)",
        model_path, tp, mem_fraction,
    )
    engine = sgl.Engine(
        model_path=model_path,
        enable_return_hidden_states=True,
        tp_size=tp,
        mem_fraction_static=mem_fraction,
        trust_remote_code=True,
    )
    logger.info("Engine ready")
    return engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean_pool_hidden_states(hidden_states, hidden_size: int) -> list[float]:
    """Mean-pool per-token hidden states to a single (hidden_size,) vector.

    hidden_states from Engine.generate() varies in shape:
      - (1, num_tokens, hidden_size) for single prompts
      - (num_tokens, hidden_size) sometimes
      - list of per-token arrays
    """
    arr = np.array(hidden_states, dtype=np.float32)

    # Squeeze batch dimension if present: (1, T, H) -> (T, H)
    while arr.ndim > 2:
        arr = arr[0]

    # Handle 1D case (single token): (H,) -> (1, H)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # arr is now (num_tokens, hidden_size)
    if arr.shape[-1] != hidden_size:
        raise ValueError(
            f"Hidden size mismatch: expected {hidden_size}, got {arr.shape[-1]}"
        )

    embedding = arr.mean(axis=0)  # (hidden_size,)
    return embedding.tolist()


def _format_generate_response(output: dict) -> dict:
    """Format Engine.generate() output to match SGLang HTTP server format."""
    meta = output.get("meta_info", {})

    resp = {
        "text": output.get("text", ""),
        "meta_info": {},
    }

    # Copy relevant meta_info fields
    for key in [
        "output_token_logprobs",
        "prompt_token_logprobs",
        "completion_tokens",
        "prompt_tokens",
        "finish_reason",
    ]:
        if key in meta:
            resp["meta_info"][key] = meta[key]

    return resp


# ---------------------------------------------------------------------------
# Encode helpers
# ---------------------------------------------------------------------------


async def _encode_batch(prompts: list[str], hidden_size: int) -> list[list[float]]:
    """Encode all prompts in a single async_generate call (true GPU batching).

    Returns zero vectors for individual prompts that fail (missing hidden_states
    or shape mismatch) instead of failing the whole batch.
    """
    zero = [0.0] * hidden_size
    outputs = await _engine.async_generate(
        prompts,
        sampling_params={"max_new_tokens": 1, "temperature": 0},
        return_hidden_states=True,
    )
    embeddings = []
    for i, output in enumerate(outputs):
        hs = output.get("meta_info", {}).get("hidden_states")
        if hs is None:
            logger.warning("Batch encode prompt %d/%d: no hidden_states (len=%d chars, first80=%s)",
                           i + 1, len(prompts), len(prompts[i]), prompts[i][:80])
            embeddings.append(list(zero))
            continue
        try:
            emb = _mean_pool_hidden_states(hs, hidden_size)
            # Detect zero embeddings from valid hidden_states
            emb_norm = sum(x * x for x in emb) ** 0.5
            if emb_norm < 1e-6:
                logger.warning("Batch encode prompt %d/%d: zero-norm embedding (hs_type=%s, hs_len=%s, len=%d chars, first80=%s)",
                               i + 1, len(prompts), type(hs).__name__,
                               getattr(hs, 'shape', None) or len(hs) if hasattr(hs, '__len__') else '?',
                               len(prompts[i]), prompts[i][:80])
            embeddings.append(emb)
        except Exception as e:
            logger.warning("Batch encode prompt %d/%d: %s", i + 1, len(prompts), e)
            embeddings.append(list(zero))
    return embeddings


async def _encode_sequential(prompts: list[str], hidden_size: int) -> list[list[float]]:
    """Encode prompts as individual async_generate calls, fired concurrently.

    Each prompt is a separate single-prompt request (hidden_states work correctly
    for individual requests). All requests are fired concurrently via asyncio.gather
    so SGLang's continuous batching can still group them on the GPU.

    Returns per-prompt results: successful embeddings or zero vectors on failure.
    """
    zero = [0.0] * hidden_size

    async def _encode_one(prompt: str, idx: int) -> list[float]:
        try:
            output = await asyncio.wait_for(
                _engine.async_generate(
                    [prompt],
                    sampling_params={"max_new_tokens": 1, "temperature": 0},
                    return_hidden_states=True,
                ),
                timeout=10.0,
            )
            hs = output[0].get("meta_info", {}).get("hidden_states")
            if hs is None:
                logger.warning("Encode prompt %d/%d: no hidden_states (len=%d chars)", idx + 1, len(prompts), len(prompt))
                return list(zero)
            emb = _mean_pool_hidden_states(hs, hidden_size)
            emb_norm = sum(x * x for x in emb) ** 0.5
            if emb_norm < 1e-6:
                logger.warning("Encode prompt %d/%d: zero-norm embedding (len=%d chars, first80=%s)",
                               idx + 1, len(prompts), len(prompt), prompt[:80])
            return emb
        except asyncio.TimeoutError:
            logger.warning("Encode prompt %d/%d: timeout (len=%d chars)", idx + 1, len(prompts), len(prompt))
            return list(zero)
        except Exception as e:
            logger.warning("Encode prompt %d/%d: error %s (len=%d chars)", idx + 1, len(prompts), e, len(prompt))
            return list(zero)

    # Must serialize: SGLang's continuous batching merges concurrent requests
    # internally, and return_hidden_states is broken for all but the first
    # request in an internal batch. asyncio.gather would trigger this.
    results = []
    for i, p in enumerate(prompts):
        results.append(await _encode_one(p, i))
    return results


def _run_batch_selftest(engine, hidden_size: int = 4096) -> bool:
    """Test whether batch hidden states match individual results.

    Runs synchronously at startup (before the event loop is serving requests).
    Returns True if batch encoding is safe to use.
    """
    import asyncio as _asyncio

    test_prompts = [
        "n : Nat\n⊢ n + 0 = n",
        "p q : Prop\nhp : p\n⊢ p ∧ q",
    ]

    async def _test():
        # Individual encodes
        individual = []
        for prompt in test_prompts:
            outputs = await engine.async_generate(
                [prompt],
                sampling_params={"max_new_tokens": 1, "temperature": 0},
                return_hidden_states=True,
            )
            hs = outputs[0].get("meta_info", {}).get("hidden_states")
            if hs is None:
                logger.warning("Self-test: no hidden_states returned for individual encode")
                return False
            individual.append(np.array(_mean_pool_hidden_states(hs, hidden_size)))

        # Batch encode
        outputs = await engine.async_generate(
            test_prompts,
            sampling_params={"max_new_tokens": 1, "temperature": 0},
            return_hidden_states=True,
        )
        batch = []
        for i, output in enumerate(outputs):
            hs = output.get("meta_info", {}).get("hidden_states")
            if hs is None:
                logger.warning("Self-test: no hidden_states in batch output[%d]", i)
                return False
            batch.append(np.array(_mean_pool_hidden_states(hs, hidden_size)))

        # Compare cosine similarity
        for i in range(len(test_prompts)):
            dot = np.dot(individual[i], batch[i])
            norm_a = np.linalg.norm(individual[i])
            norm_b = np.linalg.norm(batch[i])
            if norm_a < 1e-8 or norm_b < 1e-8:
                logger.warning("Self-test: near-zero norm for prompt %d", i)
                return False
            cos_sim = dot / (norm_a * norm_b)
            logger.info("Self-test prompt %d: cosine_sim=%.6f", i, cos_sim)
            if cos_sim < 0.99:
                logger.warning(
                    "Self-test FAILED: prompt %d cosine_sim=%.6f < 0.99", i, cos_sim
                )
                return False

        return True

    try:
        loop = _asyncio.new_event_loop()
        result = loop.run_until_complete(_test())
        loop.close()
        return result
    except Exception:
        logger.exception("Self-test raised an exception")
        return False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Drop-in replacement for SGLang's /generate endpoint.

    Always passes return_hidden_states=True to Engine.generate() to avoid
    CUDA graph recapture. Hidden states are discarded before responding
    unless the client explicitly requested them.

    Uses Engine.async_generate() to avoid the event-loop nesting issue
    that occurs when Engine.generate() calls loop.run_until_complete()
    inside an already-running async server.
    """
    is_batch = isinstance(request.text, list)
    prompts = request.text if is_batch else [request.text]

    outputs = await _engine.async_generate(
        prompts,
        sampling_params=request.sampling_params,
        return_hidden_states=True,
        return_logprob=request.return_logprob,
    )

    responses = []
    for output in outputs:
        resp = _format_generate_response(output)

        # Include hidden states only if client explicitly requested them
        # (for backward compatibility with legacy clients)
        if request.return_hidden_states:
            meta = output.get("meta_info", {})
            if "hidden_states" in meta:
                resp["meta_info"]["hidden_states"] = meta["hidden_states"]

        responses.append(resp)

    if is_batch:
        return responses
    else:
        return responses[0]


@app.post("/encode")
async def encode(request: EncodeRequest):
    """Encode text to a mean-pooled embedding.

    Hidden states are mean-pooled in-process (as PyTorch/numpy tensors)
    before JSON serialization, reducing response from ~10MB to ~16KB.

    Uses true batch encoding if the startup self-test passed, otherwise
    falls back to sequential Semaphore(1) processing (SGLang Issue #8066).
    """
    is_batch = isinstance(request.text, list)
    prompts = request.text if is_batch else [request.text]

    if _batch_encode_ok:
        embeddings = await _encode_batch(prompts, request.hidden_size)
    else:
        embeddings = await _encode_sequential(prompts, request.hidden_size)

    if is_batch:
        return {"embeddings": embeddings}
    else:
        return {"embedding": embeddings[0]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="BurnQED Inference Server")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path or HuggingFace model ID",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Server port (default: 30000, overridden by PORT env var)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism (default: 1, overridden by TP env var)",
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.85,
        help="Static memory fraction (default: 0.85, overridden by MEM_FRACTION env var)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    return parser.parse_args()


def main():
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    # Environment variables override CLI args
    port = int(os.environ.get("PORT", args.port))
    tp = int(os.environ.get("TP", args.tp))
    mem_fraction = float(os.environ.get("MEM_FRACTION", args.mem_fraction))

    global _engine, _batch_encode_ok, _encode_semaphore
    _engine = _build_engine(args.model_path, tp=tp, mem_fraction=mem_fraction)
    _encode_semaphore = asyncio.Semaphore(1)

    # Batch return_hidden_states is broken in SGLang — only the first item in
    # a batch gets valid hidden states, rest are zeros (SGLang #8066).
    # chunked_prefill_size=-1 does NOT fix this. Use sequential encoding
    # (individual async_generate calls, one at a time). At ~35 states/sec
    # this gives ~1h45m for full 217K cache and ~460ms per search expansion.
    _batch_encode_ok = False
    logger.info(
        "Batch encoding DISABLED (SGLang #8066) — sequential at ~35 states/sec"
    )

    logger.info("Starting server on %s:%d", args.host, port)
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
