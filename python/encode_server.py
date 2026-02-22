#!/usr/bin/env python3
"""Standalone encode server for BurnQED.

Loads the model directly via HuggingFace transformers (no SGLang) and serves
true batch encoding over HTTP. This bypasses SGLang's broken batch
return_hidden_states (Issue #8066) and provides much higher throughput than
the serialized fallback in inference_server.py.

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
    """Matches inference_server.py /encode request format."""
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


# ---------------------------------------------------------------------------
# Core encode logic (duplicated from encode_embeddings.py to avoid coupling)
# ---------------------------------------------------------------------------

_max_batch_size = 16  # Default; overridden by --max-batch-size CLI arg


@torch.no_grad()
def _encode_chunk(prompts: list[str]) -> np.ndarray:
    """Encode a single chunk of prompts, return (batch, hidden_size) numpy array."""
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

    return pooled.cpu().float().numpy()


@torch.no_grad()
def encode_batch(prompts: list[str]) -> np.ndarray:
    """Encode prompts in sub-batches to cap peak VRAM usage."""
    if len(prompts) <= _max_batch_size:
        return _encode_chunk(prompts)
    # Process in chunks to avoid VRAM OOM from activation memory
    chunks = []
    for i in range(0, len(prompts), _max_batch_size):
        chunk = prompts[i : i + _max_batch_size]
        chunks.append(_encode_chunk(chunk))
    return np.concatenate(chunks, axis=0)


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

    Returns same format as inference_server.py:
      - Single text: {"embedding": [floats...]}
      - Batch text:  {"embeddings": [[floats...], ...]}
    """
    is_batch = isinstance(request.text, list)
    prompts = request.text if is_batch else [request.text]

    t0 = time.time()
    embeddings = encode_batch(prompts)
    elapsed = time.time() - t0

    logger.info(
        "Encoded %d prompt(s) in %.3fs (%.1f states/sec)",
        len(prompts), elapsed, len(prompts) / elapsed if elapsed > 0 else 0,
    )

    # Validate hidden_size
    if embeddings.shape[-1] != request.hidden_size:
        logger.warning(
            "Hidden size mismatch: model=%d, requested=%d",
            embeddings.shape[-1], request.hidden_size,
        )

    embeddings_list = embeddings.tolist()

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
        default=16,
        help="Max prompts per forward pass to cap VRAM usage (default: 16). "
             "Larger batches are split into sub-batches automatically.",
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

    global _model, _tokenizer, _device, _max_length, _max_batch_size, _model_path
    _model_path = str(os.path.realpath(args.model_path))
    _max_length = args.max_length
    _max_batch_size = args.max_batch_size
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    logger.info("Starting encode server on %s:%d", args.host, port)
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
