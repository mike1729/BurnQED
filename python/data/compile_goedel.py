#!/usr/bin/env python3
"""Resumable batch compiler for Goedel proofs via `lake env lean`.

Processes files in batches, saves state after each batch so runs can be
interrupted and resumed. Uses `lake env lean` from the Pantograph project root.

Usage:
    cd vendor/Pantograph
    # Full run:
    python ../../python/data/compile_goedel.py --data-dir goedel_data --batch-size 1000
    # Resume after interrupt:
    python ../../python/data/compile_goedel.py --data-dir goedel_data --batch-size 1000 --resume
    # Test with small batches:
    python ../../python/data/compile_goedel.py --data-dir goedel_data --batch-size 10

State file: <out-dir>/compile_state.json
"""

import argparse
import subprocess
import json
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def check_file(args_tuple):
    """Worker function. Takes (file, timeout) to avoid globals with ProcessPoolExecutor."""
    lean_file, timeout = args_tuple
    cmd = ["lake", "env", "lean", lean_file]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        log = result.stdout + result.stderr
        if result.returncode == 0:
            has_warning = "warning" in log
            return {"file": lean_file, "status": "warn" if has_warning else "ok", "log": log}
        else:
            return {"file": lean_file, "status": "error", "log": log}
    except subprocess.TimeoutExpired:
        return {"file": lean_file, "status": "timeout", "log": f"Timeout > {timeout}s"}


def load_state(state_path):
    """Load existing state or return empty state."""
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"completed_files": {}, "batches_done": 0}


def save_state(state_path, state):
    """Atomically save state (write tmp + rename)."""
    tmp = state_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    tmp.rename(state_path)


def save_batch_log(out_dir, batch_idx, results):
    """Save per-batch error log."""
    log_path = out_dir / f"batch_{batch_idx:04d}_errors.log"
    errors = [r for r in results if r["status"] in ("error", "timeout")]
    if not errors:
        return
    with open(log_path, "w") as f:
        for r in errors:
            tag = "[TIMEOUT]" if r["status"] == "timeout" else ""
            f.write(f"--- {r['file']} {tag} ---\n{r['log']}\n\n")


def save_summary(out_dir, state):
    """Save cumulative summary from state."""
    completed = state["completed_files"]
    counts = {"ok": 0, "warn": 0, "error": 0, "timeout": 0}
    for status in completed.values():
        counts[status] += 1
    total = len(completed)
    passing = counts["ok"] + counts["warn"]

    summary = {
        "total_processed": total,
        "ok": counts["ok"],
        "warnings": counts["warn"],
        "errors": counts["error"],
        "timeouts": counts["timeout"],
        "pass_rate": f"{100*passing/total:.1f}%" if total > 0 else "N/A",
        "batches_done": state["batches_done"],
        "error_files": [f for f, s in completed.items() if s == "error"],
        "timeout_files": [f for f, s in completed.items() if s == "timeout"],
    }
    with open(out_dir / "compilation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Also write a single combined error log
    with open(out_dir / "compilation_errors.log", "w") as f:
        for fpath, status in sorted(completed.items()):
            if status in ("error", "timeout"):
                # We don't store full logs in state (too big), but batch logs have them
                tag = "[TIMEOUT]" if status == "timeout" else "[ERROR]"
                f.write(f"--- {fpath} {tag} ---\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Resumable batch Goedel proof compiler")
    parser.add_argument("--data-dir", type=str, default="goedel_data")
    parser.add_argument("--out-dir", type=str, default="compile_output")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous state")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    state_path = out_dir / "compile_state.json"

    # Discover all files
    all_files = sorted(str(f) for f in data_dir.glob("Proof_*.lean"))
    print(f"Found {len(all_files)} total proof files")

    # Load or init state
    if args.resume and state_path.exists():
        state = load_state(state_path)
        already_done = set(state["completed_files"].keys())
        remaining = [f for f in all_files if f not in already_done]
        print(f"Resuming: {len(already_done)} already done, {len(remaining)} remaining")
    else:
        state = {"completed_files": {}, "batches_done": 0}
        remaining = all_files
        print(f"Starting fresh: {len(remaining)} files to process")

    if not remaining:
        print("Nothing to do — all files already processed.")
        summary = save_summary(out_dir, state)
        print(f"Pass rate: {summary['pass_rate']}")
        return

    # Split into batches
    batches = []
    for i in range(0, len(remaining), args.batch_size):
        batches.append(remaining[i:i + args.batch_size])

    print(f"Split into {len(batches)} batches of up to {args.batch_size}")
    print(f"Workers: {args.workers}, timeout: {args.timeout}s")
    print()

    global_start = time.time()

    for batch_idx, batch in enumerate(batches):
        batch_num = state["batches_done"] + 1
        batch_start = time.time()
        print(f"=== Batch {batch_num} ({len(batch)} files) ===")

        batch_results = []
        work_items = [(f, args.timeout) for f in batch]

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(check_file, item): item[0] for item in work_items}
            pbar = tqdm(as_completed(futures), total=len(batch),
                        desc=f"Batch {batch_num}")
            for future in pbar:
                res = future.result()
                batch_results.append(res)
                state["completed_files"][res["file"]] = res["status"]

                # Live counts
                c = state["completed_files"]
                ok = sum(1 for s in c.values() if s == "ok")
                warn = sum(1 for s in c.values() if s == "warn")
                err = sum(1 for s in c.values() if s == "error")
                to = sum(1 for s in c.values() if s == "timeout")
                pbar.set_postfix_str(f"ok={ok} warn={warn} err={err} to={to}")

        # Save after each batch
        state["batches_done"] = batch_num
        save_state(state_path, state)
        save_batch_log(out_dir, batch_num, batch_results)

        batch_elapsed = time.time() - batch_start
        batch_ok = sum(1 for r in batch_results if r["status"] in ("ok", "warn"))
        batch_err = sum(1 for r in batch_results if r["status"] == "error")
        batch_to = sum(1 for r in batch_results if r["status"] == "timeout")
        print(f"  Batch {batch_num} done in {batch_elapsed:.0f}s: "
              f"{batch_ok} pass, {batch_err} err, {batch_to} timeout")
        print(f"  State saved to {state_path}")
        print()

    # Final summary
    summary = save_summary(out_dir, state)
    total_elapsed = time.time() - global_start

    total = summary["total_processed"]
    passing = summary["ok"] + summary["warnings"]
    print(f"{'='*50}")
    print(f"DONE in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Total: {total} files")
    print(f"  OK:       {summary['ok']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Errors:   {summary['errors']}")
    print(f"  Timeouts: {summary['timeouts']}")
    print(f"Pass rate:  {passing}/{total} = {summary['pass_rate']}")
    print(f"Results:    {out_dir}/compilation_results.json")
    print(f"State:      {state_path} (use --resume to continue)")


if __name__ == "__main__":
    main()
