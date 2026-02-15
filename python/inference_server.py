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
from concurrent.futures import ThreadPoolExecutor
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
_executor: ThreadPoolExecutor = None
# Semaphore to serialize encode requests (SGLang Issue #8066: BS>1 corrupts hidden states)
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
    """
    is_batch = isinstance(request.text, list)
    prompts = request.text if is_batch else [request.text]

    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        _executor,
        lambda: _engine.generate(
            prompts,
            sampling_params=request.sampling_params,
            return_hidden_states=True,
            return_logprob=request.return_logprob,
        ),
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

    Serialized with asyncio.Semaphore(1) to work around SGLang Issue #8066
    where BS>1 produces incorrect hidden states.
    """
    is_batch = isinstance(request.text, list)
    prompts = request.text if is_batch else [request.text]

    # Sequential processing: SGLang Issue #8066 reports incorrect hidden states
    # when BS>1. When the upstream fix lands, replace this loop with a single
    # engine.generate(prompts, ...) call for true batch encoding (~3-5x faster).
    embeddings = []
    for prompt in prompts:
        async with _encode_semaphore:
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                _executor,
                lambda p=prompt: _engine.generate(
                    [p],
                    sampling_params={"max_new_tokens": 1, "temperature": 0},
                    return_hidden_states=True,
                ),
            )

        output = outputs[0]
        hs = output.get("meta_info", {}).get("hidden_states")
        if hs is None:
            raise ValueError(
                "Engine did not return hidden_states. "
                "Ensure enable_return_hidden_states=True in Engine constructor."
            )

        embedding = _mean_pool_hidden_states(hs, request.hidden_size)
        embeddings.append(embedding)

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
        "--workers",
        type=int,
        default=4,
        help="ThreadPoolExecutor workers for Engine.generate() calls (default: 4)",
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

    global _engine, _executor, _encode_semaphore
    _engine = _build_engine(args.model_path, tp=tp, mem_fraction=mem_fraction)
    _executor = ThreadPoolExecutor(max_workers=args.workers)
    _encode_semaphore = asyncio.Semaphore(1)

    logger.info("Starting server on %s:%d", args.host, port)
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
