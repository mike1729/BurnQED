"""
Verify SGLang hidden states format for BurnQED EBM integration.

Run on a GPU machine:
    python python/verify_sglang_hidden_states.py ~/BurnQED/models/DeepSeek-Prover-V2-7B

What to check:
    1. hidden_states shape: (1, num_prompt_tokens, 4096)
    2. Mean-pooling produces a (4096,) vector
    3. Log-prob format for tactic ranking
    4. Whether HTTP server supports return_hidden_states (Issue #6528)
"""
import sys
import numpy as np


DEFAULT_MODEL = "deepseek-ai/DeepSeek-Prover-V2-7B"


def main():
    import sglang as sgl

    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"Loading {model_path}...")
    engine = sgl.Engine(
        model_path=model_path,
        enable_return_hidden_states=True,
    )

    # --- Test 1: Hidden states shape ---
    print("\n=== Test 1: Hidden states from offline Engine ===")
    out = engine.generate(
        ["test input"],
        sampling_params={"max_new_tokens": 1},
        return_hidden_states=True,
    )
    hs = out[0]["meta_info"]["hidden_states"]
    arr = np.array(hs, dtype=np.float32)
    print(f"raw shape: {arr.shape}")  # Observed: (1, num_tokens, 4096)

    # --- Test 2: Mean-pooling ---
    print("\n=== Test 2: Mean-pooling ===")
    # Shape is (1, num_tokens, 4096) — squeeze batch dim, then mean over tokens
    if arr.ndim == 3:
        arr = arr[0]  # (num_tokens, 4096)
    print(f"after squeeze: {arr.shape}")
    embedding = arr.mean(axis=0)
    print(f"embedding shape: {embedding.shape}")  # Expected: (4096,)
    print(f"embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"embedding[:5]: {embedding[:5]}")

    # --- Test 3: Proof state prompt ---
    print("\n=== Test 3: Proof state hidden states ===")
    proof_state = "n : Nat\n\u22a2 n + 0 = n"
    prompt = (
        "Complete the following Lean 4 code:\n\n"
        "```lean4\n"
        "/- tactic state:\n"
        f"{proof_state}\n"
        "-/\n"
        "```"
    )
    out2 = engine.generate(
        [prompt],
        sampling_params={"max_new_tokens": 1},
        return_hidden_states=True,
    )
    hs2 = out2[0]["meta_info"]["hidden_states"]
    arr2 = np.array(hs2, dtype=np.float32)
    if arr2.ndim == 3:
        arr2 = arr2[0]
    print(f"proof state shape: {arr2.shape}")  # Expected: (num_tokens, 4096)
    emb2 = arr2.mean(axis=0)
    print(f"embedding shape: {emb2.shape}")  # Expected: (4096,)
    print(f"embedding norm: {np.linalg.norm(emb2):.4f}")

    # --- Test 4: Generation with logprobs ---
    # SGLang 0.5.x uses 'return_logprob' as a top-level arg, not in sampling_params.
    # Try multiple approaches to find the correct API.
    print("\n=== Test 4: Generation with logprobs ===")

    # Approach A: return_logprob as top-level kwarg (like return_hidden_states)
    try:
        out3 = engine.generate(
            [prompt],
            sampling_params={
                "max_new_tokens": 64,
                "temperature": 0.6,
                "top_p": 0.95,
                "n": 4,
            },
            return_logprob=True,
        )
        print("  (used return_logprob as top-level kwarg)")
    except TypeError:
        # Approach B: logprobs in sampling_params
        try:
            out3 = engine.generate(
                [prompt],
                sampling_params={
                    "max_new_tokens": 64,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "n": 4,
                    "logprobs": True,
                },
            )
            print("  (used logprobs in sampling_params)")
        except TypeError:
            # Approach C: no logprobs, just generate
            out3 = engine.generate(
                [prompt],
                sampling_params={
                    "max_new_tokens": 64,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "n": 4,
                },
            )
            print("  (no logprob support found, generated without)")

    for i, item in enumerate(out3):
        text = item["text"]
        meta = item["meta_info"]
        print(f"\n  Candidate {i}:")
        print(f"    text: {repr(text[:100])}")
        print(f"    meta_info keys: {list(meta.keys())}")
        # Try various logprob key names
        for key in ["output_token_logprobs", "token_logprobs", "logprobs",
                     "output_logprobs", "completion_token_logprobs"]:
            if key in meta:
                logprobs = meta[key]
                print(f"    {key} type: {type(logprobs)}, len: {len(logprobs) if isinstance(logprobs, list) else 'N/A'}")
                if logprobs and len(logprobs) > 0:
                    print(f"    first entry: {logprobs[0]}")
                    try:
                        total_lp = sum(
                            lp[0] if isinstance(lp, (list, tuple)) else float(lp)
                            for lp in logprobs
                        )
                        print(f"    total log_prob: {total_lp:.4f}")
                    except (TypeError, ValueError) as e:
                        print(f"    (couldn't sum logprobs: {e})")
                break
        else:
            print("    No logprob keys found in meta_info")

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
            print(
                "HTTP hidden_states is None - server does NOT support it (Issue #6528)"
            )
            print(f"meta_info keys: {list(data.get('meta_info', {}).keys())}")
    except Exception as e:
        print(f"HTTP test skipped (server not running or error): {e}")

    engine.shutdown()
    print("\nDone. Use shapes above to adjust /encode indexing in SglangClient.")


if __name__ == "__main__":
    main()
