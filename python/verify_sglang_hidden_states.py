"""
Verify SGLang hidden states format for BurnQED EBM integration.

Run on a GPU machine:
    pip install "sglang[all]"
    python python/verify_sglang_hidden_states.py

What to check:
    1. hidden_states shape: expected (num_prompt_tokens, 4096)
    2. Mean-pooling produces a (4096,) vector
    3. Log-prob format for tactic ranking
    4. Whether HTTP server supports return_hidden_states (Issue #6528)
"""
import sglang as sgl
import numpy as np

MODEL = "deepseek-ai/DeepSeek-Prover-V2-7B"

print(f"Loading {MODEL}...")
engine = sgl.Engine(model_path=MODEL)

# --- Test 1: Hidden states shape ---
print("\n=== Test 1: Hidden states from offline Engine ===")
out = engine.generate(
    ["test input"],
    sampling_params={"max_new_tokens": 1},
    return_hidden_states=True,
)
hs = out[0]["meta_info"]["hidden_states"]
print(f"type: {type(hs)}")
if isinstance(hs, list):
    print(f"len: {len(hs)}")
    if len(hs) > 0:
        first = hs[0]
        print(f"  element type: {type(first)}")
        if isinstance(first, list):
            print(f"  element len: {len(first)} (expected 4096)")
        elif hasattr(first, "shape"):
            print(f"  element shape: {first.shape}")
elif hasattr(hs, "shape"):
    print(f"shape: {hs.shape}")

# --- Test 2: Mean-pooling ---
print("\n=== Test 2: Mean-pooling ===")
arr = np.array(hs, dtype=np.float32)
print(f"numpy shape: {arr.shape}")  # Expected: (num_tokens, 4096)
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
print(f"proof state hidden states shape: {arr2.shape}")
emb2 = arr2.mean(axis=0)
print(f"proof state embedding shape: {emb2.shape}")
print(f"proof state embedding norm: {np.linalg.norm(emb2):.4f}")

# --- Test 4: Generation with logprobs ---
print("\n=== Test 4: Generation with logprobs ===")
out3 = engine.generate(
    [prompt],
    sampling_params={
        "max_new_tokens": 64,
        "temperature": 0.6,
        "top_p": 0.95,
        "n": 4,
        "return_logprob": True,
        "logprob_start_len": -1,
    },
)
for i, item in enumerate(out3):
    text = item["text"]
    meta = item["meta_info"]
    logprobs = meta.get("output_token_logprobs", [])
    print(f"\n  Candidate {i}:")
    print(f"    text: {repr(text[:80])}")
    print(
        f"    logprobs type: {type(logprobs)}, "
        f"len: {len(logprobs) if isinstance(logprobs, list) else 'N/A'}"
    )
    if logprobs and len(logprobs) > 0:
        print(f"    first logprob entry: {logprobs[0]}")
        total_lp = sum(
            lp[0] if isinstance(lp, (list, tuple)) else lp for lp in logprobs
        )
        print(f"    total log_prob: {total_lp:.4f}")

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
