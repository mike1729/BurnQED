"""
Test batch /encode vs individual /encode.

Usage:
  1. Start inference server: ./scripts/start_inference_server.sh
  2. python scripts/test_batch_encode.py [--url http://localhost:30000]
"""
import argparse
import json
import requests
import numpy as np
import time

def encode_single(url, text, hidden_size=4096):
    resp = requests.post(f"{url}/encode", json={"text": text, "hidden_size": hidden_size}, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if "embedding" in body:
        return np.array(body["embedding"], dtype=np.float32)
    raise ValueError(f"Unexpected response: {str(body)[:200]}")

def encode_batch(url, texts, hidden_size=4096):
    resp = requests.post(f"{url}/encode", json={"text": texts, "hidden_size": hidden_size}, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    # Custom server: {"embeddings": [[f32...], ...]}
    if isinstance(body, dict) and "embeddings" in body:
        return [np.array(e, dtype=np.float32) for e in body["embeddings"]]
    # SGLang fallback: top-level array [{"embedding": [...]}, ...]
    if isinstance(body, list):
        results = []
        for i, item in enumerate(body):
            if "embedding" in item:
                results.append(np.array(item["embedding"], dtype=np.float32))
            else:
                results.append(None)
                print(f"  [FAIL] Item {i}: no 'embedding' key. Keys: {list(item.keys())}")
        return results
    raise ValueError(f"Unexpected response format: {str(body)[:200]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:30000")
    parser.add_argument("--hidden-size", type=int, default=4096)
    args = parser.parse_args()

    # Test prompts of varying lengths
    prompts = [
        "n : Nat\n⊢ n + 0 = n",
        "α : Type\nl : List α\n⊢ l.reverse.reverse = l",
        "x y : ℝ\nhx : x > 0\nhy : y > 0\n⊢ x * y > 0",
        "p q : Prop\nhp : p\nhq : q\n⊢ p ∧ q",
        "n m : Nat\n⊢ n + m = m + n",
        "a b c : Nat\n⊢ a + (b + c) = (a + b) + c",
        "f : Nat → Nat\nhf : ∀ n, f n = n * 2\n⊢ f 5 = 10",
        "xs ys : List Nat\n⊢ (xs ++ ys).length = xs.length + ys.length",
    ]

    print(f"Server: {args.url}")
    print(f"Testing {len(prompts)} prompts\n")

    # --- Test 1: Individual requests ---
    print("=== Individual /encode requests ===")
    individual = []
    for i, p in enumerate(prompts):
        try:
            emb = encode_single(args.url, p, args.hidden_size)
            individual.append(emb)
            print(f"  [{i}] dim={len(emb)}, norm={np.linalg.norm(emb):.4f}")
        except Exception as e:
            individual.append(None)
            print(f"  [{i}] FAILED: {e}")

    # --- Test 2: Batch request (all at once) ---
    print(f"\n=== Batch /encode (batch_size={len(prompts)}) ===")
    try:
        batch = encode_batch(args.url, prompts, args.hidden_size)
        for i, emb in enumerate(batch):
            if emb is not None:
                print(f"  [{i}] dim={len(emb)}, norm={np.linalg.norm(emb):.4f}")
            else:
                print(f"  [{i}] EMPTY/MISSING")
    except Exception as e:
        print(f"  BATCH FAILED: {e}")
        batch = [None] * len(prompts)

    # --- Test 3: Compare ---
    print("\n=== Comparison (individual vs batch) ===")
    all_match = True
    for i in range(len(prompts)):
        if individual[i] is None or batch[i] is None:
            print(f"  [{i}] SKIP (missing)")
            all_match = False
            continue
        cos_sim = np.dot(individual[i], batch[i]) / (np.linalg.norm(individual[i]) * np.linalg.norm(batch[i]) + 1e-8)
        l2_dist = np.linalg.norm(individual[i] - batch[i])
        dim_match = len(individual[i]) == len(batch[i]) == args.hidden_size
        ok = cos_sim > 0.99 and dim_match
        status = "OK" if ok else "MISMATCH"
        print(f"  [{i}] {status}: cos_sim={cos_sim:.6f}, l2={l2_dist:.4f}, dims={len(individual[i])}/{len(batch[i])}")
        if not ok:
            all_match = False

    # --- Test 4: Increasing batch sizes ---
    print("\n=== Batch size sweep ===")
    for bs in [2, 4, 8]:
        subset = prompts[:bs]
        try:
            t0 = time.time()
            batch_res = encode_batch(args.url, subset, args.hidden_size)
            elapsed = time.time() - t0
            ok_count = sum(1 for e in batch_res if e is not None and len(e) == args.hidden_size)
            print(f"  batch_size={bs}: {ok_count}/{bs} OK, {elapsed:.2f}s")
        except Exception as e:
            print(f"  batch_size={bs}: FAILED — {e}")

    print(f"\n{'ALL PASSED' if all_match else 'FAILURES DETECTED'}")

if __name__ == "__main__":
    main()
