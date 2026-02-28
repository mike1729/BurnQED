#!/usr/bin/env python3
"""Apply Mathlib v4.8→v4.27 rename fixes to Goedel proof files and recompile failures.

Run from the Pantograph project root (where lakefile.toml lives).

Usage:
    cd vendor/Pantograph
    python ../../python/data/fix_goedel_renames.py --results compilation_results.json --workers 32
"""

import argparse
import json
import re
import subprocess
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Mathlib identifier renames: v4.8 -> v4.27
RENAMES = {
    r'\ble_div_iff\b': 'le_div_iff₀',
    r'\bdiv_le_div_iff\b': 'div_le_div_iff₀',
    r'\bdiv_le_iff\b': 'div_le_iff₀',
    r'\badd_left_neg\b': 'neg_add_cancel',
    r'\badd_right_neg\b': 'add_neg_cancel',
    r'\btrue_and_iff\b': 'true_and',
    r'\band_true_iff\b': 'and_true',
}

# BigOperators notation: ∑ i in S -> ∑ i ∈ S (Lean 4.27)
SUM_IN_PATTERN = re.compile(r'(∑\s+\w+)\s+in\s+')
PROD_IN_PATTERN = re.compile(r'(∏\s+\w+)\s+in\s+')


def apply_fixes(filepath: str) -> bool:
    """Apply all known fixes to a file. Returns True if modified."""
    text = Path(filepath).read_text()
    new_text = text

    for pattern, replacement in RENAMES.items():
        new_text = re.sub(pattern, replacement, new_text)

    new_text = SUM_IN_PATTERN.sub(r'\1 ∈ ', new_text)
    new_text = PROD_IN_PATTERN.sub(r'\1 ∈ ', new_text)

    if new_text != text:
        Path(filepath).write_text(new_text)
        return True
    return False


def check_file(lean_file: str, timeout: int) -> dict:
    cmd = ["lake", "env", "lean", lean_file]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        log = result.stdout + result.stderr
        if result.returncode == 0:
            return {"status": "success", "file": lean_file, "log": log}
        else:
            return {"status": "error", "file": lean_file, "log": log}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "file": lean_file, "log": f"Timeout > {timeout}s"}


def main():
    parser = argparse.ArgumentParser(description="Fix Mathlib renames and recompile failed Goedel proofs")
    parser.add_argument("--results", type=str, default="compilation_results.json",
                        help="Path to compilation_results.json from compile_goedel.py")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of parallel workers")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-file timeout in seconds")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Directory for output logs/json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    with open(args.results) as f:
        results = json.load(f)

    # Support both formats:
    # - compile.py save_summary format: {"error_files": [...], "timeout_files": [...], ...}
    # - compile_proofs.py raw state format: {"completed": {"seq": {"status": ..., "module": ...}}, ...}
    if "error_files" in results:
        failed = results["error_files"] + results["timeout_files"]
    elif "completed" in results:
        failed = []
        for seq, info in results["completed"].items():
            if info["status"] in ("error", "timeout"):
                failed.append(f"GoedelMigration/Proof_{int(seq):05d}.lean")
        # Reconstruct summary keys for round 2 reporting
        counts = {"ok": 0, "warn": 0, "error": 0, "timeout": 0}
        for info in results["completed"].values():
            counts[info["status"]] = counts.get(info["status"], 0) + 1
        results = {
            "total": len(results["completed"]),
            "clean": counts["ok"],
            "warnings": counts["warn"],
            "errors": counts["error"],
            "timeouts": counts["timeout"],
            "error_files": [f for f in failed if "timeout" not in f],
            "timeout_files": [],
        }
    else:
        print(f"ERROR: Unrecognized results format. Keys: {list(results.keys())}")
        sys.exit(1)

    print(f"Applying fixes to {len(failed)} files...")

    fixed_count = sum(1 for f in failed if apply_fixes(f))
    print(f"Modified {fixed_count}/{len(failed)} files")

    # Recompile
    successes, warnings, errors, timeouts = [], [], [], []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(check_file, f, args.timeout): f for f in failed}
        pbar = tqdm(as_completed(futures), total=len(failed), desc="Recompiling fixed")
        for future in pbar:
            res = future.result()
            if res["status"] == "success":
                if "warning" in res["log"]:
                    warnings.append(res)
                else:
                    successes.append(res["file"])
            elif res["status"] == "timeout":
                timeouts.append(res)
            else:
                errors.append(res)
            pbar.set_postfix_str(
                f"ok={len(successes)} warn={len(warnings)} err={len(errors)} to={len(timeouts)}"
            )

    with open(out_dir / "compilation_errors_round2.log", "w") as f:
        for err in errors:
            f.write(f"--- {err['file']} ---\n{err['log']}\n\n")
        for t in timeouts:
            f.write(f"--- {t['file']} [TIMEOUT] ---\n{t['log']}\n\n")

    round2 = {
        "attempted": len(failed),
        "fixed_files": fixed_count,
        "clean": len(successes),
        "warnings": len(warnings),
        "errors": len(errors),
        "timeouts": len(timeouts),
        "error_files": [e["file"] for e in errors],
        "timeout_files": [t["file"] for t in timeouts],
    }
    with open(out_dir / "compilation_results_round2.json", "w") as f:
        json.dump(round2, f, indent=2)

    total_ok = len(successes) + len(warnings)
    prev_ok = results["clean"] + results["warnings"]
    new_total = prev_ok + total_ok
    total = results["total"]
    print(f"\nRound 2: {len(successes)} clean, {len(warnings)} warn, {len(errors)} err, {len(timeouts)} timeout")
    print(f"Rescued: {total_ok}/{len(failed)}")
    print(f"New total: {new_total}/{total} = {100*new_total/total:.1f}%")
    print(f"See {out_dir}/compilation_errors_round2.log")


if __name__ == "__main__":
    main()
