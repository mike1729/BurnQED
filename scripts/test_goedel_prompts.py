#!/usr/bin/env python3
"""Experiment with Goedel-Prover-V2-8B whole-proof generation via SGLang.

Focus: direct_sorry prompt format (no CoT), temperature sweep, extraction validation.

Usage:
    # Start SGLang first:
    ./scripts/start_sglang.sh goedel

    # Run all experiments:
    python scripts/test_goedel_prompts.py

    # Specific theorem / temperature:
    python scripts/test_goedel_prompts.py --theorem square_equation --temperature 1.4
    python scripts/test_goedel_prompts.py --sweep   # full temperature sweep
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

SGLANG_URL = "http://localhost:30000"

# ── Test theorems (increasing difficulty) ────────────────────────────────────

THEOREMS = {
    # --- Easy: single-tactic ---
    "comm_add": {
        "statement": """\
import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat

theorem comm_add {a b : ℝ} : a + b = b + a := by
  sorry""",
        "difficulty": "easy",
    },
    "nat_pos": {
        "statement": """\
import Mathlib
import Aesop
set_option maxHeartbeats 0

theorem nat_pos {n : ℕ} (h : n > 0) : n ≥ 1 := by
  sorry""",
        "difficulty": "easy",
    },
    # --- Medium: multi-step ---
    "square_equation": {
        "statement": """\
import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat

theorem square_equation_solution {x y : ℝ} (h : x^2 + y^2 = 2*x - 4*y - 5) : x + y = -1 := by
  sorry""",
        "difficulty": "medium",
    },
    "div_iff": {
        "statement": """\
import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat

theorem div_iff_example {a b : ℝ} (hb : b ≠ 0) (h : a / b = 3) : a = 3 * b := by
  sorry""",
        "difficulty": "medium",
    },
    "abs_ineq": {
        "statement": """\
import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat

theorem abs_ineq {x : ℝ} (h : |x - 2| ≤ 3) : -1 ≤ x ∧ x ≤ 5 := by
  sorry""",
        "difficulty": "medium",
    },
    # --- Hard: competition-style (miniF2F level) ---
    "mathd_algebra_10": {
        "statement": """\
import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat

theorem mathd_algebra_10 (x : ℝ) (h₀ : x ≠ 0) (h₁ : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry""",
        "difficulty": "hard",
    },
    "imo_mod": {
        "statement": """\
import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat

theorem imo_mod (n : ℕ) : (6 * n^2 + 5) % 12 ≠ 0 := by
  sorry""",
        "difficulty": "hard",
    },
    "sum_first_n": {
        "statement": """\
import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat

theorem sum_first_n (n : ℕ) : 2 * (∑ i in Finset.range (n + 1), i) = n * (n + 1) := by
  sorry""",
        "difficulty": "hard",
    },
}

# ── Prompt construction ──────────────────────────────────────────────────────

def build_direct_sorry_prompt(statement: str, no_think: bool = True) -> dict:
    """Build raw Qwen3/ChatML prompt: just code block, no instruction framing.

    This is the `direct_sorry` format that won our initial comparison.
    Assistant prefix starts inside a lean4 code fence so the model
    immediately generates the proof.
    """
    user_msg = f"```lean4\n{statement}\n```"
    prompt = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    if no_think:
        prompt += "<think>\n\n</think>\n\n"
    prompt += "```lean4\n"
    return {
        "prompt": prompt,
        "stop": ["```", "<|im_end|>"],
        "max_tokens": 2048,
    }


# ── Tactic extraction (Python port of Rust extract_all_tactics_structured) ──

def contains_sorry(tactic: str) -> bool:
    """Word-boundary sorry/admit detection (mirrors Rust contains_sorry)."""
    for keyword in ("sorry", "admit", "cheat"):
        for m in re.finditer(re.escape(keyword), tactic):
            start, end = m.start(), m.end()
            before_ok = start == 0 or not (tactic[start-1].isalnum() or tactic[start-1] == '_')
            after_ok = end == len(tactic) or not (tactic[end].isalnum() or tactic[end] == '_')
            if before_ok and after_ok:
                return True
    return False


def extract_all_tactics_structured(raw: str) -> list[str]:
    """Simplified Python port of the Rust extraction for validation.

    Handles: code fence stripping, declaration skipping, bullet stripping,
    comment skipping, sorry detection. Does NOT do full have-decomposition
    (that's the Rust side's job). This is for quick validation.
    """
    text = raw.strip()
    # Strip code fence
    if text.startswith("```"):
        lines = text.split("\n")
        inner = []
        for line in lines[1:]:
            if line.strip().startswith("```"):
                break
            inner.append(line)
        text = "\n".join(inner)

    tactics = []
    for line in text.split("\n"):
        trimmed = line.strip()
        if not trimmed:
            continue
        # Skip comments
        if trimmed.startswith("--") or trimmed.startswith("/-") or trimmed.startswith("-/"):
            continue
        # Skip declarations (theorem/lemma/example headers + imports + set_option + open)
        if any(trimmed.startswith(kw) for kw in
               ("theorem ", "lemma ", "example ", "import ", "set_option ", "open ", "#check")):
            continue
        # Strip bullets
        for bullet in ("· ", "∙ "):
            if trimmed.startswith(bullet):
                trimmed = trimmed[len(bullet):].strip()
        if trimmed in ("·", "∙", ""):
            continue
        # Skip dangling <;>
        if trimmed == "<;>" or trimmed.startswith("<;> "):
            continue
        tactics.append(trimmed)

    return tactics


# ── SGLang request ───────────────────────────────────────────────────────────

def generate_batch(url: str, prompt_spec: dict, temperature: float, n: int) -> list[dict]:
    """Generate n completions for a single prompt via SGLang /generate."""
    # Batch: repeat prompt n times
    payload = {
        "text": [prompt_spec["prompt"]] * n,
        "sampling_params": {
            "max_new_tokens": prompt_spec["max_tokens"],
            "temperature": temperature,
            "top_p": 0.95,
            "stop": prompt_spec.get("stop", ["<|im_end|>"]),
        },
        "return_logprob": True,
    }

    t0 = time.time()
    resp = requests.post(f"{url}/generate", json=payload, timeout=180)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list):
        data = [data]

    results = []
    for item in data:
        text = item.get("text", "")
        meta = item.get("meta_info", {})
        logprobs = meta.get("output_token_logprobs", [])
        n_tokens = len(logprobs)
        avg_lp = (sum(lp[0] for lp in logprobs if isinstance(lp, list) and lp) / max(n_tokens, 1)
                  if logprobs else 0.0)
        results.append({
            "text": text,
            "tokens": n_tokens,
            "avg_logprob": avg_lp,
            "elapsed_total": elapsed,
        })
    return results


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_result(raw_text: str) -> dict:
    """Analyze a single generation result."""
    tactics = extract_all_tactics_structured(raw_text)
    has_sorry = any(contains_sorry(t) for t in tactics)

    # Check if model re-emitted the theorem header (expected for whole-proof)
    has_theorem_header = bool(re.search(r'(theorem|lemma|example)\s+\w+', raw_text))

    # Check for common problematic patterns
    has_semicolon_spam = raw_text.count("<;>") > 3
    has_repeated_tactics = False
    if len(tactics) > 3:
        # Check if any tactic appears more than twice
        from collections import Counter
        counts = Counter(tactics)
        has_repeated_tactics = any(c > 2 for c in counts.values())

    return {
        "tactics": tactics,
        "n_tactics": len(tactics),
        "has_sorry": has_sorry,
        "has_theorem_header": has_theorem_header,
        "has_semicolon_spam": has_semicolon_spam,
        "has_repeated_tactics": has_repeated_tactics,
    }


def print_result(idx: int, result: dict, analysis: dict, verbose: bool = True):
    """Print one generation result."""
    flags = []
    if analysis["has_sorry"]:
        flags.append("SORRY")
    if analysis["has_semicolon_spam"]:
        flags.append("<;>SPAM")
    if analysis["has_repeated_tactics"]:
        flags.append("REPEAT")
    flag_str = f" [{', '.join(flags)}]" if flags else ""

    print(f"  [{idx}] {result['tokens']} tok, {result['elapsed_total']:.1f}s, "
          f"avg_lp={result['avg_logprob']:.3f}, "
          f"tactics={analysis['n_tactics']}{flag_str}")

    if verbose:
        text = result["text"].strip()
        preview = text[:600]
        if len(text) > 600:
            preview += f"\n      ... ({len(text)} chars total)"
        for line in preview.split("\n"):
            print(f"      {line}")
        print()
        # Show extracted tactics
        if analysis["tactics"]:
            print(f"      EXTRACTED TACTICS ({analysis['n_tactics']}):")
            for i, t in enumerate(analysis["tactics"][:10]):
                print(f"        {i}: {t[:100]}")
            if len(analysis["tactics"]) > 10:
                print(f"        ... +{len(analysis['tactics'])-10} more")
            print()


# ── Experiments ──────────────────────────────────────────────────────────────

def run_single(url: str, thm_name: str, thm: dict, temperature: float, n: int, verbose: bool):
    """Run one (theorem, temperature) experiment."""
    spec = build_direct_sorry_prompt(thm["statement"])

    print(f"\n{'='*72}")
    print(f"  {thm_name} [{thm['difficulty']}]  T={temperature}  n={n}")
    print(f"{'='*72}")

    results = generate_batch(url, spec, temperature=temperature, n=n)

    stats = {
        "n_sorry": 0, "n_semicolon_spam": 0, "n_repeated": 0,
        "n_empty": 0, "total_tactics": 0, "total_tokens": 0,
    }

    for i, r in enumerate(results, 1):
        analysis = analyze_result(r["text"])
        print_result(i, r, analysis, verbose=verbose)

        if analysis["has_sorry"]:
            stats["n_sorry"] += 1
        if analysis["has_semicolon_spam"]:
            stats["n_semicolon_spam"] += 1
        if analysis["has_repeated_tactics"]:
            stats["n_repeated"] += 1
        if analysis["n_tactics"] == 0:
            stats["n_empty"] += 1
        stats["total_tactics"] += analysis["n_tactics"]
        stats["total_tokens"] += r["tokens"]

    avg_tactics = stats["total_tactics"] / max(len(results), 1)
    avg_tokens = stats["total_tokens"] / max(len(results), 1)
    print(f"  SUMMARY: avg_tactics={avg_tactics:.1f}, avg_tokens={avg_tokens:.0f}, "
          f"sorry={stats['n_sorry']}/{n}, <;>spam={stats['n_semicolon_spam']}/{n}, "
          f"repeat={stats['n_repeated']}/{n}, empty={stats['n_empty']}/{n}")

    return stats


def run_temperature_sweep(url: str, thm_name: str, thm: dict, n: int):
    """Sweep temperatures matching production config range."""
    temperatures = [0.2, 0.6, 1.0, 1.4, 1.8]
    print(f"\n{'#'*72}")
    print(f"  TEMPERATURE SWEEP: {thm_name} [{thm['difficulty']}]")
    print(f"{'#'*72}")

    sweep_results = {}
    for temp in temperatures:
        stats = run_single(url, thm_name, thm, temp, n, verbose=False)
        sweep_results[temp] = stats

    # Summary table
    print(f"\n  {'T':>4} | {'avg_tac':>7} | {'sorry':>5} | {'<;>':>3} | {'repeat':>6} | {'empty':>5}")
    print(f"  {'----':>4}-+-{'-------':>7}-+-{'-----':>5}-+-{'---':>3}-+-{'------':>6}-+-{'-----':>5}")
    for temp in temperatures:
        s = sweep_results[temp]
        avg_t = s["total_tactics"] / n
        print(f"  {temp:>4.1f} | {avg_t:>7.1f} | {s['n_sorry']:>5} | {s['n_semicolon_spam']:>3} | "
              f"{s['n_repeated']:>6} | {s['n_empty']:>5}")


def main():
    parser = argparse.ArgumentParser(description="Test Goedel whole-proof generation")
    parser.add_argument("--url", default=SGLANG_URL)
    parser.add_argument("--theorem", default=None, help="Run only this theorem")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Single temperature (default: 0.6 for normal, sweep for --sweep)")
    parser.add_argument("--n", type=int, default=8, help="Samples per experiment")
    parser.add_argument("--sweep", action="store_true", help="Temperature sweep mode")
    parser.add_argument("--verbose", action="store_true", help="Show full outputs")
    parser.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard"],
                       help="Filter by difficulty")
    args = parser.parse_args()

    # Check server
    try:
        resp = requests.get(f"{args.url}/health", timeout=5)
        print(f"SGLang server: OK ({resp.status_code})")
    except Exception as e:
        print(f"Cannot reach SGLang at {args.url}: {e}")
        print("Start it: ./scripts/start_sglang.sh goedel")
        return

    # Filter theorems
    theorems = THEOREMS
    if args.theorem:
        if args.theorem not in THEOREMS:
            print(f"Unknown theorem: {args.theorem}")
            print(f"Available: {', '.join(THEOREMS.keys())}")
            return
        theorems = {args.theorem: THEOREMS[args.theorem]}
    if args.difficulty:
        theorems = {k: v for k, v in theorems.items() if v["difficulty"] == args.difficulty}

    if args.sweep:
        for name, thm in theorems.items():
            run_temperature_sweep(args.url, name, thm, args.n)
    else:
        temp = args.temperature or 0.6
        for name, thm in theorems.items():
            run_single(args.url, name, thm, temp, args.n, verbose=args.verbose or not args.theorem)


if __name__ == "__main__":
    main()
