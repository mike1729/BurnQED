#!/usr/bin/env python3
"""Prepare search theorem files from compiled Goedel 4.27 proofs.

Extracts theorem statements from .lean files, converts binder syntax to
∀-quantified Lean expressions suitable for Pantograph's goal.start, and
validates them in parallel against a running Lean pool.

Usage:
    # Full pipeline: extract + validate + write
    python -m python.data.prepare_search_theorems \
        --manifest data/lean/goedel_migration/goedel_manifest.json \
        --integrity data/traced/integrity_report.json \
        --lean-dir data/lean/goedel_migration/GoedelMigration \
        --output data/benchmarks/iter0_search_theorems.json \
        --sample 10000 \
        --validate --pantograph-url http://localhost:30000 \
        --workers 8

    # Extract only (no validation)
    python -m python.data.prepare_search_theorems \
        --manifest data/lean/goedel_migration/goedel_manifest.json \
        --integrity data/traced/integrity_report.json \
        --lean-dir data/lean/goedel_migration/GoedelMigration \
        --output data/benchmarks/iter0_search_theorems.json \
        --sample 10000
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Name qualification
# ---------------------------------------------------------------------------

# The .lean files use: open BigOperators Real Nat Topology Rat
# Pantograph loads with `import Mathlib` but no open scopes, so we must
# fully qualify bare names.
_QUALIFY_RULES = [
    (re.compile(r"(?<![.\w])sin(?!\w)"), "Real.sin"),
    (re.compile(r"(?<![.\w])cos(?!\w)"), "Real.cos"),
    (re.compile(r"(?<![.\w])tan(?!\w)"), "Real.tan"),
    (re.compile(r"(?<![.\w])exp(?![.\w])"), "Real.exp"),
    (re.compile(r"(?<![.\w])log(?!\w)"), "Real.log"),
    (re.compile(r"(?<![.\w])sqrt(?!\w)"), "Real.sqrt"),
    (re.compile(r"(?<![.\w])arctan(?!\w)"), "Real.arctan"),
    (re.compile(r"(?<![.\w])arcsin(?!\w)"), "Real.arcsin"),
    (re.compile(r"(?<![.\w])arccos(?!\w)"), "Real.arccos"),
    (re.compile(r"(?<![.\w])pi(?!\w)"), "Real.pi"),
    (re.compile(r"(?<![.\w])choose(?!\w)"), "Nat.choose"),
    (re.compile(r"(?<![.\w])factorial(?!\w)"), "Nat.factorial"),
    (re.compile(r"(?<![.\w])gcd(?!\w)"), "Nat.gcd"),
    (re.compile(r"(?<![.\w])lcm(?!\w)"), "Nat.lcm"),
    (re.compile(r"(?<![.\w])Coprime(?!\w)"), "Nat.Coprime"),
]


def qualify_names(stmt: str) -> str:
    """Replace bare names from `open Real Nat` with fully qualified versions."""
    for pat, repl in _QUALIFY_RULES:
        stmt = pat.sub(repl, stmt)
    return stmt


# ---------------------------------------------------------------------------
# Statement extraction
# ---------------------------------------------------------------------------

# Unicode bracket pairs for depth tracking
_OPEN = set("([{⟨")
_CLOSE = set(")]}⟩")


def _consume_binders(sig: str) -> tuple[str, str]:
    """Split a theorem signature into (binders, proposition).

    Lean theorem signatures: `theorem foo (a : T) {b : U} [inst] : Prop := by`
    Binders are bracket-delimited groups at the start: (...), {...}, [...].
    The ':' after all binders separates them from the proposition.

    Also handles theorems with no binders: `theorem foo : Prop := by`
    where sig starts with ': Prop'.
    """
    sig = sig.strip()

    # If sig starts with ':', theorem has no explicit binders
    if sig.startswith(":"):
        return "", sig[1:].strip()

    # Consume consecutive bracket groups as binders
    pos = 0
    depth = 0
    last_close = -1

    while pos < len(sig):
        ch = sig[pos]
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
            if depth == 0:
                last_close = pos
        elif depth == 0:
            if ch == ":":
                # Found the type-annotation colon after binders
                binders = sig[:pos].strip()
                prop = sig[pos + 1 :].strip()
                return binders, prop
            elif ch not in " \t\n\r":
                # Non-whitespace, non-bracket, non-colon at depth 0
                # after closing a bracket group means we hit something
                # unexpected. Stop looking for more binders.
                break
        pos += 1

    # Fallback: no clear colon separator found.
    # The entire sig is the type (e.g. `∀ (x : T), P`)
    return "", sig


def extract_statement(content: str, theorem_name: str) -> Optional[str]:
    """Extract a ∀-quantified type expression from a .lean theorem definition.

    Converts `theorem foo (a : T) (h : P) : Q := by` into `∀ (a : T) (h : P), Q`.
    """
    pattern = r"theorem\s+" + re.escape(theorem_name) + r"\s+(.*?)\s*:=\s*by"
    m = re.search(pattern, content, re.DOTALL)
    if not m:
        return None

    sig = m.group(1).strip()
    # Remove doc comments
    sig = re.sub(r"/\-.*?\-/", "", sig, flags=re.DOTALL).strip()

    binders, prop = _consume_binders(sig)
    if binders:
        stmt = f"∀ {binders}, {prop}"
    else:
        stmt = prop

    stmt = qualify_names(stmt)
    return stmt


def _extract_batch(args: tuple) -> list:
    """Process a batch of (pid, filepath) pairs. Used by ProcessPoolExecutor."""
    batch, lean_dir = args
    results = []
    for pid, filename in batch:
        fpath = os.path.join(lean_dir, filename)
        if not os.path.exists(fpath):
            results.append((pid, None, "file_not_found"))
            continue
        try:
            with open(fpath) as f:
                content = f.read()
            stmt = extract_statement(content, pid)
            if stmt is None:
                results.append((pid, None, "parse_error"))
            else:
                results.append((pid, stmt, "ok"))
        except Exception as e:
            results.append((pid, None, f"exception: {e}"))
    return results


# ---------------------------------------------------------------------------
# Validation against Pantograph
# ---------------------------------------------------------------------------


async def validate_statements_pantograph(
    theorems: list[dict],
    pantograph_cmd: list[str],
    num_workers: int = 8,
    timeout_per: float = 10.0,
) -> tuple[list[dict], list[dict]]:
    """Validate theorem statements by sending them to Pantograph instances.

    Returns (valid, invalid) lists.
    """
    # We'll use the Rust prover binary's validate mode if available,
    # or fall back to direct Pantograph JSON protocol
    valid = []
    invalid = []

    sem = asyncio.Semaphore(num_workers)
    total = len(theorems)
    done = 0
    t0 = time.time()

    async def check_one(thm: dict) -> tuple[dict, bool, str]:
        nonlocal done
        async with sem:
            stmt = thm["statement"]
            # Use Pantograph directly: spawn process, send goal.start
            try:
                proc = await asyncio.create_subprocess_exec(
                    *pantograph_cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                # Send goal.start command
                cmd = json.dumps({"cmd": "goal.start", "payload": {"expr": stmt}})
                proc.stdin.write((cmd + "\n").encode())
                await proc.stdin.drain()

                # Read response with timeout
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=timeout_per
                    )
                    resp = json.loads(line.decode().strip())
                    if "error" in resp:
                        return thm, False, resp.get("desc", str(resp))
                    else:
                        return thm, True, "ok"
                except asyncio.TimeoutError:
                    return thm, False, "timeout"
                finally:
                    proc.kill()
                    await proc.wait()
            except Exception as e:
                return thm, False, str(e)

    # Process in parallel
    tasks = [check_one(t) for t in theorems]

    for coro in asyncio.as_completed(tasks):
        thm, ok, reason = await coro
        done += 1
        if ok:
            valid.append(thm)
        else:
            invalid.append({**thm, "error": reason})

        if done % 500 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"  Validated {done}/{total} "
                f"({len(valid)} ok, {len(invalid)} bad) "
                f"[{elapsed:.0f}s, ETA {eta:.0f}s]"
            )

    return valid, invalid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/lean/goedel_migration/goedel_manifest.json",
        help="Goedel manifest JSON",
    )
    parser.add_argument(
        "--integrity",
        default="data/traced/integrity_report.json",
        help="Integrity report JSON (from M.6 sweep)",
    )
    parser.add_argument(
        "--lean-dir",
        default="data/lean/goedel_migration/GoedelMigration",
        help="Directory containing .lean proof files",
    )
    parser.add_argument(
        "--output",
        default="data/benchmarks/iter0_search_theorems.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Random sample size (default: use all clean theorems)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for extraction and validation",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate statements against Pantograph (requires running Lean pool)",
    )
    parser.add_argument(
        "--pantograph-cmd",
        nargs="+",
        default=None,
        help="Pantograph command (e.g., pantograph-repl --imports Mathlib)",
    )
    args = parser.parse_args()

    # ── 1. Load clean theorem IDs ──
    print("Loading clean theorem IDs...")
    with open(args.integrity) as f:
        integrity = json.load(f)
    clean_ids = set(t["problem_id"] for t in integrity["clean_theorems"])
    print(f"  {len(clean_ids)} clean theorems from integrity report")

    # ── 2. Load manifest for filename mapping ──
    with open(args.manifest) as f:
        manifest = json.load(f)
    id_to_file = {m["problem_id"]: m["filename"] for m in manifest}

    # Filter to clean IDs that are in the manifest
    work_items = [(pid, id_to_file[pid]) for pid in clean_ids if pid in id_to_file]
    print(f"  {len(work_items)} theorems with files in manifest")

    # ── 3. Extract statements in parallel ──
    print(f"Extracting statements ({args.workers} workers)...")
    t0 = time.time()

    # Split into batches for ProcessPoolExecutor
    batch_size = max(1, len(work_items) // (args.workers * 4))
    batches = []
    for i in range(0, len(work_items), batch_size):
        batches.append((work_items[i : i + batch_size], args.lean_dir))

    statements = {}
    status_counts = {"ok": 0, "toplevel_colon": 0, "parse_error": 0, "file_not_found": 0, "exception": 0}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_extract_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            for pid, stmt, status in future.result():
                if status == "ok":
                    statements[pid] = stmt
                    status_counts["ok"] += 1
                elif status == "toplevel_colon":
                    # Statement has a bare ':' at top level — likely a type
                    # ascription that wasn't properly converted. Still include
                    # it but flag for review.
                    statements[pid] = stmt
                    status_counts["toplevel_colon"] += 1
                else:
                    key = status.split(":")[0].strip() if ":" in status else status
                    status_counts[key] = status_counts.get(key, 0) + 1

    elapsed = time.time() - t0
    print(f"  Extracted {len(statements)} statements in {elapsed:.1f}s")
    for k, v in status_counts.items():
        if v > 0:
            print(f"    {k}: {v}")

    # ── 4. Sample ──
    all_ids = list(statements.keys())
    if args.sample and args.sample < len(all_ids):
        random.seed(args.seed)
        sample_ids = random.sample(all_ids, args.sample)
        print(f"Sampled {len(sample_ids)} from {len(all_ids)} theorems (seed={args.seed})")
    else:
        sample_ids = all_ids
        print(f"Using all {len(sample_ids)} theorems")

    theorems = [{"name": pid, "statement": statements[pid]} for pid in sample_ids]

    # ── 5. Optional validation ──
    if args.validate:
        if args.pantograph_cmd:
            print(f"Validating {len(theorems)} statements against Pantograph...")
            valid, invalid = asyncio.run(
                validate_statements_pantograph(
                    theorems,
                    args.pantograph_cmd,
                    num_workers=args.workers,
                )
            )
            print(f"  Valid: {len(valid)}, Invalid: {len(invalid)}")
            if invalid:
                # Save invalid for debugging
                invalid_path = args.output.replace(".json", "_invalid.json")
                with open(invalid_path, "w") as f:
                    json.dump(invalid, f, ensure_ascii=False, indent=2)
                print(f"  Invalid theorems saved to {invalid_path}")
            theorems = valid
        else:
            print("WARNING: --validate requires --pantograph-cmd, skipping validation")

    # ── 6. Write output ──
    output = {"theorems": theorems}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"\nWritten {len(theorems)} theorems to {args.output}")
    # Show samples
    for t in theorems[:3]:
        print(f"  {t['name']}: {t['statement'][:120]}")


if __name__ == "__main__":
    main()
