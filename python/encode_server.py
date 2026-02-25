#!/usr/bin/env python3
"""Standalone encode server for BurnQED.

Loads the model directly via HuggingFace transformers (no SGLang) and serves
true batch encoding over HTTP. Concurrent requests are coalesced server-side
via an asyncio queue + background batch worker for GPU-efficient batching.

Two endpoints:
  POST /encode  — Returns mean-pooled (hidden_size,) embeddings
  GET  /health  — Returns 200

Usage:
  python python/encode_server.py --model-path models/deepseek-prover-v2-7b
  python python/encode_server.py --model-path models/llm/iter_3 --port 30001

Environment variables (override CLI args):
  ENCODE_PORT   — Server port (default: 30001)
  ENCODE_DTYPE  — Model dtype: float16, bfloat16, float32, nf4 (default: bfloat16)
"""

import argparse
import asyncio
import logging
import os
import time
from typing import Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("encode_server")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class EncodeRequest(BaseModel):
    """Encode request: single text or batch of texts."""
    text: Union[str, list[str]]
    hidden_size: int = 4096


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

app = FastAPI(title="BurnQED Encode Server")

# Global state set during startup
_model = None
_tokenizer = None
_device = None
_max_length = 2048
_model_path = "unknown"

# Dynamic batching state (initialized in startup event)
_request_queue: asyncio.Queue = None
_linger_ms: int = 5


# ---------------------------------------------------------------------------
# Core encode logic (duplicated from encode_embeddings.py to avoid coupling)
# ---------------------------------------------------------------------------

_max_batch_size = 8  # Default; overridden by --max-batch-size CLI arg


@torch.no_grad()
def _encode_chunk(prompts: list[str]) -> np.ndarray:
    """Encode a single chunk of prompts, return (batch, hidden_size) numpy array."""
    try:
        inputs = _tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_max_length,
        ).to(_device)

        # Use base model (model.model) to get last_hidden_state directly.
        # Avoids storing all 30+ layers of hidden states (saves ~15GB VRAM).
        outputs = _model.model(**inputs)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Mean pool over non-padding tokens
        attention_mask = inputs["attention_mask"]  # (batch, seq_len)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # (batch, hidden_size)
        count = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
        pooled = sum_hidden / count  # (batch, hidden_size)
        result = pooled.cpu().float().numpy()
    finally:
        # Release activation cache so large batches don't permanently inflate VRAM.
        # Also recovers from OOM — frees partial allocations so next request can succeed.
        torch.cuda.empty_cache()

    return result


@torch.no_grad()
def encode_batch(prompts: list[str]) -> np.ndarray:
    """Encode prompts in sub-batches to cap peak VRAM usage.

    On OOM, halves the batch size and retries down to batch=1.
    """
    batch_size = min(_max_batch_size, len(prompts))
    while batch_size >= 1:
        try:
            if len(prompts) <= batch_size:
                return _encode_chunk(prompts)
            chunks = []
            for i in range(0, len(prompts), batch_size):
                chunk = prompts[i : i + batch_size]
                chunks.append(_encode_chunk(chunk))
            return np.concatenate(chunks, axis=0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            old = batch_size
            batch_size = max(1, batch_size // 2)
            if old == 1:
                raise  # Already at batch=1, can't recover
            logger.warning(
                "OOM at batch_size=%d, retrying with batch_size=%d", old, batch_size
            )


# ---------------------------------------------------------------------------
# Dynamic batching: queue + background worker
# ---------------------------------------------------------------------------

async def _batch_worker():
    """Background task that coalesces concurrent encode requests.

    1. Wait for the first request
    2. Linger briefly to accumulate concurrent requests
    3. Merge all prompts into a flat batch
    4. Run encode_batch() via run_in_executor() (non-blocking)
    5. Distribute result slices back via asyncio.Future
    """
    loop = asyncio.get_running_loop()

    while True:
        # 1. Wait for first request
        first_prompts, first_future = await _request_queue.get()
        pending = [(first_prompts, first_future)]

        # 2. Linger to accumulate more requests
        deadline = loop.time() + _linger_ms / 1000.0
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(
                    _request_queue.get(), timeout=remaining
                )
                pending.append(item)
            except asyncio.TimeoutError:
                break

        # 3. Merge all prompts into flat batch
        all_prompts = []
        offsets = []  # (start, length) for each request
        for prompts, _ in pending:
            start = len(all_prompts)
            all_prompts.extend(prompts)
            offsets.append((start, len(prompts)))

        logger.info(
            "Batch worker: coalesced %d request(s), %d total prompts",
            len(pending), len(all_prompts),
        )

        # 4. Run encode on thread pool (non-blocking for event loop)
        try:
            embeddings = await loop.run_in_executor(None, encode_batch, all_prompts)
            embeddings_list = embeddings.tolist()

            # 5. Distribute result slices
            for (start, length), (_, future) in zip(offsets, pending):
                if not future.cancelled():
                    future.set_result(embeddings_list[start : start + length])
        except Exception as e:
            # Propagate error to all waiters
            for _, future in pending:
                if not future.cancelled():
                    future.set_exception(e)


@app.on_event("startup")
async def startup_event():
    """Start the background batch worker."""
    global _request_queue
    _request_queue = asyncio.Queue()
    asyncio.create_task(_batch_worker())
    logger.info("Batch worker started (linger_ms=%d)", _linger_ms)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.get("/model_info")
async def model_info():
    """Return model path and config for reproducibility."""
    return {"model_path": _model_path}


@app.post("/encode")
async def encode(request: EncodeRequest):
    """Encode text to mean-pooled embeddings.

    Requests are enqueued and coalesced by the background batch worker
    for GPU-efficient batching. Each caller awaits its own Future.

    Returns:
      - Single text: {"embedding": [floats...]}
      - Batch text:  {"embeddings": [[floats...], ...]}
    """
    is_batch = isinstance(request.text, list)
    prompts = request.text if is_batch else [request.text]

    t0 = time.time()

    # Enqueue and wait for batch worker to process
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await _request_queue.put((prompts, future))
    embeddings_list = await future

    elapsed = time.time() - t0
    logger.info(
        "Encoded %d prompt(s) in %.3fs (%.1f states/sec)",
        len(prompts), elapsed, len(prompts) / elapsed if elapsed > 0 else 0,
    )

    # Validate hidden_size
    if embeddings_list and len(embeddings_list[0]) != request.hidden_size:
        logger.warning(
            "Hidden size mismatch: model=%d, requested=%d",
            len(embeddings_list[0]), request.hidden_size,
        )

    if is_batch:
        return {"embeddings": embeddings_list}
    else:
        return {"embedding": embeddings_list[0]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="BurnQED Encode Server")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30001,
        help="Server port (default: 30001, overridden by ENCODE_PORT env var)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32", "nf4"],
        default="bfloat16",
        help="Model dtype (default: bfloat16, overridden by ENCODE_DTYPE env var). "
             "nf4 uses 4-bit NormalFloat quantization via bitsandbytes (~4GB VRAM).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max token length for truncation (default: 2048)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Max prompts per forward pass to cap VRAM usage (default: 8). "
             "Larger batches are split into sub-batches automatically.",
    )
    parser.add_argument(
        "--linger-ms",
        type=int,
        default=5,
        help="Milliseconds to wait for additional requests before flushing a batch (default: 5).",
    )
    parser.add_argument(
        "--save-quantized",
        type=str,
        default=None,
        help="Save quantized model to this directory (e.g. models/llm/iter_4_nf4). "
             "Subsequent loads from this path are faster (~10s vs ~70s).",
    )
    return parser.parse_args()


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    args = parse_args()

    # Environment variables override CLI args
    port = int(os.environ.get("ENCODE_PORT", args.port))
    dtype_str = os.environ.get("ENCODE_DTYPE", args.dtype)

    global _model, _tokenizer, _device, _max_length, _max_batch_size, _model_path, _linger_ms
    _model_path = str(os.path.realpath(args.model_path))
    _max_length = args.max_length
    _max_batch_size = args.max_batch_size
    _linger_ms = args.linger_ms
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cap PyTorch VRAM to prevent leaked activations from filling the GPU.
    # Default 0.90 leaves ~2.4 GB headroom on a 24 GB card for the OS / SGLang.
    vram_fraction = float(os.environ.get("ENCODE_VRAM_FRACTION", "0.90"))
    if _device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(vram_fraction)
        logger.info("CUDA memory fraction capped at %.0f%%", vram_fraction * 100)

    logger.info(
        "Loading model: %s (dtype=%s, device=%s)",
        args.model_path, dtype_str, _device,
    )
    _tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Optional VRAM cap (in GB) via environment variable
    max_memory_gb = os.environ.get("ENCODE_MAX_MEMORY_GB")
    max_memory = {0: f"{max_memory_gb}GiB"} if max_memory_gb else None

    if dtype_str == "nf4":
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            attn_implementation="sdpa",
            device_map={"": 0},
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        _model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype_map[dtype_str],
            attn_implementation="sdpa",
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    _model.eval()

    # Save quantized model if requested (before stripping LM head)
    if args.save_quantized:
        logger.info("Saving quantized model to %s", args.save_quantized)
        _model.save_pretrained(args.save_quantized)
        _tokenizer.save_pretrained(args.save_quantized)
        logger.info("Quantized model saved. Re-run with --model-path %s for faster loading.", args.save_quantized)

    # Strip the LM head — we only use the base model for embeddings.
    # Frees ~1.2GB VRAM (vocab_size × hidden_size in bf16).
    if hasattr(_model, "lm_head"):
        del _model.lm_head
        torch.cuda.empty_cache()

    hidden_size = _model.config.hidden_size
    num_params = sum(p.numel() for p in _model.parameters()) / 1e9
    logger.info(
        "Model loaded: hidden_size=%d, params=%.1fB", hidden_size, num_params,
    )

    logger.info("Starting encode server on %s:%d (linger_ms=%d)", args.host, port, _linger_ms)
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
