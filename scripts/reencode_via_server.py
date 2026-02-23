#!/usr/bin/env python3
"""Re-encode embedding cache via inference server /encode endpoint.

Batch encoding with per-item zero detection. When a zero is found in a batch,
saves progress, restarts server, and continues. Saves checkpoint frequently.

Usage:
    python scripts/reencode_via_server.py \
        --input checkpoints/ebm/iter_4/embeddings.parquet \
        --output checkpoints/ebm/iter_4/embeddings_sglang.parquet \
        --server-url http://localhost:30000 --batch-size 16
"""
import argparse
import os
import subprocess
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests


def encode_batch(url: str, states: list[str], hidden_size: int = 4096) -> list[np.ndarray]:
    resp = requests.post(
        f"{url}/encode",
        json={"text": states, "hidden_size": hidden_size},
        timeout=120,
    )
    resp.raise_for_status()
    return [np.array(e, dtype=np.float32) for e in resp.json()["embeddings"]]


def restart_server(model: str, port: int, mem_fraction: float):
    """Kill and restart inference server, wait for health."""
    subprocess.run(["pkill", "-9", "-f", f"inference_server.py.*--port {port}"],
                   capture_output=True)
    time.sleep(3)
    subprocess.Popen(
        ["python3", "python/inference_server.py",
         "--model-path", model, "--port", str(port),
         "--mem-fraction", str(mem_fraction)],
        stdout=open("/tmp/inference_server.log", "w"),
        stderr=subprocess.STDOUT,
    )
    url = f"http://localhost:{port}"
    for _ in range(60):
        try:
            requests.get(f"{url}/health", timeout=2).raise_for_status()
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("Server failed to start after restart")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-url", default="http://localhost:30000")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--model", default="models/llm/iter_4")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument("--max-restarts", type=int, default=200)
    args = parser.parse_args()

    # Load source states
    src = pq.read_table(args.input).to_pandas()
    all_states = src["state_pp"].tolist()
    print(f"Source: {len(all_states)} states from {args.input}")

    # Resume from checkpoint
    done = {}
    if os.path.exists(args.output):
        ckpt = pq.read_table(args.output).to_pandas()
        for _, row in ckpt.iterrows():
            emb = np.array(row["embedding"], dtype=np.float32)
            if np.linalg.norm(emb) > 1e-6:
                done[row["state_pp"]] = emb.tolist()
        print(f"Resumed: {len(done)} valid embeddings from checkpoint")

    # Verify server
    requests.get(f"{args.server_url}/health", timeout=5).raise_for_status()
    print(f"Server healthy at {args.server_url}")

    restarts = 0
    t0 = time.time()
    last_save = 0

    while True:
        todo = [(i, s) for i, s in enumerate(all_states) if s not in done]
        if not todo:
            break

        print(f"\n--- Pass (restarts={restarts}, remaining={len(todo)}) ---")
        hit_zero = False

        for batch_start in range(0, len(todo), args.batch_size):
            batch = todo[batch_start : batch_start + args.batch_size]
            batch_states = [s for _, s in batch]

            try:
                embeddings = encode_batch(args.server_url, batch_states, args.hidden_size)
            except Exception as e:
                print(f"  Batch request failed: {e}")
                hit_zero = True
                break

            for j, emb in enumerate(embeddings):
                if np.linalg.norm(emb) < 1e-6:
                    # Save good ones from this batch before the zero
                    for k in range(j):
                        done[batch_states[k]] = embeddings[k].tolist()
                    print(f"  Zero at batch item {j+1}/{len(batch)}, "
                          f"total={len(done)}/{len(all_states)}")
                    hit_zero = True
                    break
                done[batch_states[j]] = emb.tolist()

            if hit_zero:
                break

            # Progress + periodic save
            encoded = len(done)
            if encoded - last_save >= args.save_every:
                elapsed = time.time() - t0
                rate = encoded / elapsed if elapsed > 0 else 0
                remaining = len(all_states) - encoded
                eta = remaining / rate if rate > 0 else 0
                print(f"  [{encoded}/{len(all_states)}] {rate:.1f}/s, "
                      f"restarts={restarts}, ETA {eta/60:.0f}m")
                _save(args.output, all_states, done)
                last_save = encoded

        # Save after each pass
        _save(args.output, all_states, done)

        if not hit_zero:
            break  # All done

        restarts += 1
        if restarts >= args.max_restarts:
            print(f"Hit max restarts ({args.max_restarts})")
            break

        print(f"Restarting server (#{restarts})...")
        restart_server(args.model, args.port, args.mem_fraction)
        print("Server restarted.")

    elapsed = time.time() - t0
    print(f"\nDone: {len(done)}/{len(all_states)} in {elapsed/60:.1f}m, "
          f"{restarts} restarts")
    _save(args.output, all_states, done)


def _save(path, all_states, done):
    states_out, embs_out = [], []
    for s in all_states:
        if s in done:
            states_out.append(s)
            embs_out.append(done[s])
    pq.write_table(pa.table({"state_pp": states_out, "embedding": embs_out}), path)
    print(f"  Saved {len(states_out)} embeddings to {path}")


if __name__ == "__main__":
    main()
