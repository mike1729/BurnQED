#!/usr/bin/env python3
"""Encode proof state embeddings via encode server or direct PyTorch forward pass.

Preferred mode: connect to the encode server (--encode-url) which runs nf4
quantized (~5GB VRAM) and can coexist with the inference server.

Fallback mode: load the model directly (--model-path without --encode-url),
requiring full VRAM and SGLang to be stopped.

Usage:
    # Via encode server (preferred — no VRAM needed, coexists with inference)
    python python/encode_embeddings.py \
        --encode-url http://localhost:30001 \
        --trajectories trajectories/iter_0.parquet trajectories/iter_1.parquet \
        --output checkpoints/ebm/iter_4/embeddings.parquet \
        --batch-size 32

    # Direct PyTorch (fallback — needs full VRAM, kill SGLang first)
    python python/encode_embeddings.py \
        --model-path models/llm/iter_3_new \
        --trajectories trajectories/iter_0.parquet trajectories/iter_1.parquet \
        --output checkpoints/ebm/iter_4/embeddings.parquet \
        --batch-size 32

    # Resume from partial cache (only encodes missing states)
    python python/encode_embeddings.py \
        --encode-url http://localhost:30001 \
        --trajectories trajectories/*.parquet \
        --output checkpoints/ebm/iter_4/embeddings.parquet \
        --cache checkpoints/ebm/old/embeddings.parquet \
        --batch-size 32

Environment:
    CUDA_VISIBLE_DEVICES  Control GPU selection (default: 0, direct mode only)

Notes:
    - Writes checkpoint every 10K states for crash recovery
    - Output schema: {state_pp: string, embedding: list<float32>}
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# DeepSeek-Prover-V2 prompt template (must match Rust format_prompt)
PROMPT_TEMPLATE = (
    "\uff5cbegin\u2581of\u2581sentence\uff5c"
    "\uff5cUser\uff5c"
    "Complete the following Lean 4 code:\n\n"
    "```lean4\n"
    "/- tactic state:\n"
    "{state}\n"
    "-/\n"
    "```"
    "\uff5cAssistant\uff5c"
)


def format_prompt(state: str) -> str:
    return PROMPT_TEMPLATE.format(state=state)


def collect_states(trajectory_files: list[str]) -> set[str]:
    """Extract unique state_pp values from trajectory parquet files."""
    states = set()
    for f in trajectory_files:
        try:
            table = pq.read_table(f, columns=["state_pp"])
            states.update(table.column("state_pp").to_pylist())
        except Exception as e:
            logger.warning("Failed to read %s: %s", f, e)
    logger.info("Collected %d unique states from %d files", len(states), len(trajectory_files))
    return states


def load_cache(cache_path: str) -> dict[str, np.ndarray]:
    """Load existing embedding cache from Parquet, skipping zero-norm entries.

    Returns numpy arrays directly to avoid Python list[float] overhead
    (36 bytes/float vs 4 bytes/float — 9x memory difference).
    """
    if not os.path.exists(cache_path):
        return {}
    pf = pq.ParquetFile(cache_path)
    cache = {}
    skipped_zeros = 0
    for batch in pf.iter_batches(batch_size=5000, columns=["state_pp", "embedding"]):
        states_col = batch.column("state_pp").to_pylist()
        embs_col = batch.column("embedding").to_pylist()
        for s, e in zip(states_col, embs_col):
            arr = np.array(e, dtype=np.float32)
            if np.linalg.norm(arr) < 1e-6:
                skipped_zeros += 1
                continue
            cache[s] = arr
    logger.info("Loaded %d cached embeddings from %s (skipped %d zero-norm)",
                len(cache), cache_path, skipped_zeros)
    return cache


def save_parquet(embeddings: dict[str, np.ndarray], output_path: str):
    """Save embeddings to Parquet in the format expected by Rust EmbeddingCache."""
    states = list(embeddings.keys())
    emb_arrays = [embeddings[s] for s in states]

    state_array = pa.array(states, type=pa.string())
    emb_list_array = pa.array(
        [e.tolist() for e in emb_arrays],
        type=pa.list_(pa.float32()),
    )

    table = pa.table({
        "state_pp": state_array,
        "embedding": emb_list_array,
    })

    # Write with tmp file to avoid partial writes
    tmp_path = output_path + ".tmp"
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, output_path)
    logger.info("Saved %d embeddings to %s", len(embeddings), output_path)


@torch.no_grad()
def encode_batch(
    model,
    tokenizer,
    states: list[str],
    device: torch.device,
    max_length: int = 2048,
) -> np.ndarray:
    """Encode a batch of proof states, return (batch, hidden_size) numpy array."""
    prompts = [format_prompt(s) for s in states]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    # Use base model (model.model) to get last_hidden_state directly.
    # Avoids storing all 30+ layers of hidden states (saves ~15GB VRAM).
    outputs = model.model(**inputs)
    hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

    # Mean pool over non-padding tokens
    attention_mask = inputs["attention_mask"]  # (batch, seq_len)
    mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
    sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # (batch, hidden_size)
    count = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
    pooled = sum_hidden / count  # (batch, hidden_size)

    return pooled.cpu().float().numpy()


def encode_batch_http(
    encode_url: str,
    states: list[str],
    hidden_size: int = 4096,
    timeout: float = 120.0,
) -> np.ndarray:
    """Encode a batch of proof states via the encode server HTTP endpoint."""
    prompts = [format_prompt(s) for s in states]
    resp = requests.post(
        f"{encode_url}/encode",
        json={"text": prompts, "hidden_size": hidden_size},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Encode proof state embeddings via encode server or direct PyTorch")
    parser.add_argument("--model-path", help="Path to HuggingFace model directory (direct mode)")
    parser.add_argument("--encode-url", help="URL of encode server, e.g. http://localhost:30001 (preferred)")
    parser.add_argument("--trajectories", nargs="+", required=True, help="Trajectory parquet files")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    parser.add_argument("--cache", help="Existing cache to resume from (skip already-encoded states)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding (default: 32)")
    parser.add_argument("--max-length", type=int, default=2048, help="Max token length (default: 2048)")
    parser.add_argument("--checkpoint-interval", type=int, default=10000, help="Save checkpoint every N states")
    parser.add_argument("--hidden-size", type=int, default=4096, help="Model hidden size (default: 4096)")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16",
                        help="Model dtype for direct mode (default: bfloat16)")
    args = parser.parse_args()

    if not args.encode_url and not args.model_path:
        parser.error("Either --encode-url or --model-path is required")

    use_server = bool(args.encode_url)

    # Collect states
    all_states = collect_states(args.trajectories)

    # Load existing cache
    cached = {}
    if args.cache:
        cached = load_cache(args.cache)
    elif os.path.exists(args.output):
        cached = load_cache(args.output)

    # Filter to uncached states
    to_encode = sorted(s for s in all_states if s not in cached)
    logger.info("States: %d total, %d cached, %d to encode", len(all_states), len(cached), len(to_encode))

    if not to_encode:
        logger.info("All states already cached, nothing to do")
        # Still save merged output if cache came from a different file
        if cached:
            embeddings = {s: e for s, e in cached.items() if s in all_states}
            save_parquet(embeddings, args.output)
        return

    # Setup encoder
    model = None
    tokenizer = None
    device = None

    if use_server:
        logger.info("Using encode server at %s", args.encode_url)
        # Verify server is reachable
        try:
            resp = requests.get(f"{args.encode_url}/health", timeout=5)
            resp.raise_for_status()
            logger.info("Encode server healthy")
        except Exception as e:
            logger.error("Encode server unreachable at %s: %s", args.encode_url, e)
            sys.exit(1)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Load model directly
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        model_dtype = dtype_map[args.dtype]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading model: %s (dtype=%s, device=%s)", args.model_path, args.dtype, device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=model_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()

        hidden_size = model.config.hidden_size
        logger.info("Model loaded: hidden_size=%d, params=%.1fB", hidden_size, sum(p.numel() for p in model.parameters()) / 1e9)

    # Encode — start from cached (already numpy arrays), add new encodings
    embeddings = {s: e for s, e in cached.items() if s in all_states}
    del cached  # free ~3GB
    num_batches = (len(to_encode) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    encoded = 0
    errors = 0

    for batch_idx in range(num_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(to_encode))
        batch_states = to_encode[start:end]

        try:
            if use_server:
                embs = encode_batch_http(args.encode_url, batch_states, args.hidden_size)
            else:
                embs = encode_batch(model, tokenizer, batch_states, device, args.max_length)

            for i, state in enumerate(batch_states):
                norm = np.linalg.norm(embs[i])
                if norm < 1e-6:
                    logger.warning("Zero-norm embedding for state (len=%d): %s", len(state), state[:80])
                    errors += 1
                    continue
                embeddings[state] = embs[i]
                encoded += 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Retry one-by-one to handle variable-length sequences
            logger.warning("Batch %d OOM (size=%d) — retrying individually", batch_idx, len(batch_states))
            for i, state in enumerate(batch_states):
                try:
                    emb = encode_batch(model, tokenizer, [state], device, args.max_length)
                    norm = np.linalg.norm(emb[0])
                    if norm < 1e-6:
                        errors += 1
                        continue
                    embeddings[state] = emb[0]
                    encoded += 1
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    logger.warning("OOM even for single state (len=%d): %s", len(state), state[:80])
                    errors += 1
                except Exception as e2:
                    logger.error("Single encode failed: %s", e2)
                    errors += 1

        except requests.exceptions.RequestException as e:
            logger.error("Batch %d HTTP error: %s", batch_idx, e)
            errors += len(batch_states)

        except Exception as e:
            logger.error("Batch %d failed: %s", batch_idx, e)
            errors += len(batch_states)

        # Progress
        done = start + len(batch_states)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(to_encode) - done) / rate if rate > 0 else 0
        if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
            logger.info(
                "Progress: %d/%d (%.1f%%) | %.1f states/s | ETA: %.0fs | errors: %d",
                done, len(to_encode), 100 * done / len(to_encode), rate, eta, errors,
            )

        # Checkpoint
        if done % args.checkpoint_interval < args.batch_size and done > 0:
            save_parquet(embeddings, args.output)
            logger.info("Checkpoint: %d embeddings saved", len(embeddings))

    # Final save
    elapsed = time.time() - t0
    logger.info(
        "Encoding complete: %d encoded, %d errors, %.1f states/s, %.0fs total",
        encoded, errors, encoded / elapsed if elapsed > 0 else 0, elapsed,
    )
    save_parquet(embeddings, args.output)

    # Verify output
    total_expected = len(all_states)
    total_saved = len(embeddings)
    missing = total_expected - total_saved
    error_rate = errors / max(len(to_encode), 1)

    logger.info(
        "Verification: %d/%d states saved (%.1f%%), %d missing, error rate %.2f%%",
        total_saved, total_expected, 100 * total_saved / max(total_expected, 1),
        missing, 100 * error_rate,
    )

    if error_rate > 0.01:
        logger.error(
            "ERROR: %.1f%% error rate (%d/%d) exceeds 1%% threshold — embeddings may be unreliable",
            100 * error_rate, errors, len(to_encode),
        )
        sys.exit(1)

    if missing > 0:
        logger.warning(
            "WARNING: %d states missing from output (%.1f%% of total)",
            missing, 100 * missing / total_expected,
        )


if __name__ == "__main__":
    main()
