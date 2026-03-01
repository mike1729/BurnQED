#!/usr/bin/env python3
"""Benchmark SGLang /generate latency with different batch sizes and parallelism."""

import asyncio
import time
import json
import aiohttp
import sys

SERVER = "http://localhost:30000"

# Sample proof state (miniF2F-style)
PROOF_STATE = "⊢ ∀ (x y : ℤ) (h₀ : 0 < y) (h₁ : y < x) (h₂ : x + y + x * y = 80), x = 26"

def format_prompt(state: str) -> str:
    msg = f"Complete the following Lean 4 proof. Give only the proof steps.\n\n```lean4\n{state}\n```"
    return (
        f"<\uff5c begin\u2581of\u2581sentence\uff5c>"
        f"<\uff5c User\uff5c>{msg}"
        f"<\uff5c Assistant\uff5c>"
        f"```lean4\n/- tactic state:\n{state}\n-/\nexample := by\n  "
    )

PROMPT = format_prompt(PROOF_STATE)

async def generate_batch(session: aiohttp.ClientSession, n: int, max_tokens: int = 1024) -> float:
    """Send a single /generate request with n prompts, return wall time in seconds."""
    payload = {
        "text": [PROMPT] * n,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 1.4,
            "top_p": 0.95,
            "stop": ["```"],
        },
        "return_logprob": True,
        "return_hidden_states": False,
    }
    t0 = time.monotonic()
    async with session.post(f"{SERVER}/generate", json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
        body = await resp.json()
    elapsed = time.monotonic() - t0
    count = len(body) if isinstance(body, list) else 0
    return elapsed, count


async def bench_sequential(session, total_n: int, chunk_size: int, max_tokens: int = 1024):
    """Sequential chunks (old behavior)."""
    t0 = time.monotonic()
    total_returned = 0
    for start in range(0, total_n, chunk_size):
        cn = min(chunk_size, total_n - start)
        _, count = await generate_batch(session, cn, max_tokens)
        total_returned += count
    return time.monotonic() - t0, total_returned


async def bench_parallel(session, total_n: int, chunk_size: int, max_tokens: int = 1024):
    """Parallel chunks (new behavior)."""
    chunks = []
    for start in range(0, total_n, chunk_size):
        cn = min(chunk_size, total_n - start)
        chunks.append(cn)

    t0 = time.monotonic()
    results = await asyncio.gather(*[generate_batch(session, cn, max_tokens) for cn in chunks])
    elapsed = time.monotonic() - t0
    total_returned = sum(count for _, count in results)
    return elapsed, total_returned


async def main():
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 1024

    print(f"SGLang batch latency benchmark (max_tokens={max_tokens})")
    print(f"Prompt length: ~{len(PROMPT)} chars")
    print(f"Server: {SERVER}")
    print()

    # Warmup
    async with aiohttp.ClientSession() as session:
        print("Warming up...", flush=True)
        await generate_batch(session, 4, max_tokens)
        print()

        configs = [
            # (total_n, chunk_size, mode)
            (4,  4,  "single"),
            (8,  8,  "single"),
            (16, 16, "single"),
            (32, 32, "single"),
            # Sequential chunking (old behavior)
            (16, 8,  "seq"),
            (32, 8,  "seq"),
            (32, 16, "seq"),
            # Parallel chunking (new behavior)
            (16, 8,  "par"),
            (32, 8,  "par"),
            (32, 16, "par"),
        ]

        print(f"{'Config':<28} {'Time (s)':>8} {'Returned':>8} {'per-seq (ms)':>12}")
        print("-" * 60)

        for total_n, chunk_size, mode in configs:
            if mode == "single":
                elapsed, returned = await generate_batch(session, total_n, max_tokens)
                label = f"n={total_n} (1 req)"
            elif mode == "seq":
                elapsed, returned = await bench_sequential(session, total_n, chunk_size, max_tokens)
                n_chunks = (total_n + chunk_size - 1) // chunk_size
                label = f"n={total_n} seq {n_chunks}x{chunk_size}"
            else:
                elapsed, returned = await bench_parallel(session, total_n, chunk_size, max_tokens)
                n_chunks = (total_n + chunk_size - 1) // chunk_size
                label = f"n={total_n} par {n_chunks}x{chunk_size}"

            per_seq = elapsed / total_n * 1000
            print(f"{label:<28} {elapsed:>8.2f} {returned:>8} {per_seq:>12.1f}")

        print()
        print("single = one HTTP request with all n prompts")
        print("seq    = sequential chunks (old code)")
        print("par    = parallel chunks (new code)")


if __name__ == "__main__":
    asyncio.run(main())
