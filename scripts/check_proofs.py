#!/usr/bin/env python3
"""Check generated proofs through Pantograph REPL.

Spawns a single Pantograph REPL with BenchMinIF2FV2STest imports,
then for each proof:
  1. goal.start with copyFrom(theorem_name)
  2. goal.tactic with the whole proof as a single tactic
  3. Check if result has 0 remaining goals

Usage:
    python3 scripts/check_proofs.py data/sglang_proofs/3thm_diversity_test.json
"""
import json
import subprocess
import sys
import time


def start_repl():
    """Start a Pantograph REPL with Mathlib + BenchMinIF2FV2STest imports."""
    proc = subprocess.Popen(
        ["lake", "exe", "repl", "BenchMinIF2FV2STest"],
        cwd="vendor/Pantograph",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    # Consume the "ready." line
    ready = proc.stdout.readline().strip()
    if "ready" not in ready.lower():
        print(f"WARNING: Expected 'ready', got: {ready}", file=sys.stderr)
    print("REPL ready.", file=sys.stderr)
    return proc


def send_cmd(proc, cmd: dict, timeout: float = 120.0) -> dict:
    """Send a JSON command and read the response."""
    line = json.dumps(cmd) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()
    resp_line = proc.stdout.readline()
    if not resp_line:
        raise RuntimeError("REPL closed unexpectedly")
    return json.loads(resp_line)


def check_proof(proc, theorem_name: str, proof_text: str) -> dict:
    """Check a single proof. Returns {compiled: bool, error: str|None}."""
    # Step 1: goal.start with copyFrom
    resp = send_cmd(proc, {
        "cmd": "goal.start",
        "payload": {"copyFrom": theorem_name}
    })
    if "error" in resp:
        return {"compiled": False, "error": f"goal.start failed: {resp['error']}"}

    state_id = resp.get("stateId")
    if state_id is None:
        return {"compiled": False, "error": f"No stateId in goal.start response: {resp}"}

    # Step 2: Apply the whole proof as a single tactic
    resp2 = send_cmd(proc, {
        "cmd": "goal.tactic",
        "payload": {"stateId": state_id, "goalId": 0, "tactic": proof_text}
    })

    if "error" in resp2:
        return {"compiled": False, "error": resp2["error"]}

    # Check if proof is complete (no remaining goals)
    goals = resp2.get("goals", [])
    if goals is not None and len(goals) == 0:
        return {"compiled": True, "error": None}
    else:
        return {"compiled": False, "error": f"Remaining goals: {len(goals) if goals else 'unknown'}"}


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/check_proofs.py <proofs.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    print("Starting Pantograph REPL (this takes ~30s with Mathlib)...", file=sys.stderr)
    proc = start_repl()

    results = {}
    total = 0
    compiled = 0

    for theorem_name, temps in data.items():
        results[theorem_name] = {}
        for temp, proofs in temps.items():
            results[theorem_name][temp] = []
            for i, proof in enumerate(proofs):
                total += 1
                result = check_proof(proc, theorem_name, proof)
                results[theorem_name][temp].append(result)
                if result["compiled"]:
                    compiled += 1
                status = "OK" if result["compiled"] else "FAIL"
                if total % 10 == 0 or result["compiled"]:
                    print(f"[{total}] {theorem_name} T={temp} #{i}: {status}", file=sys.stderr)

    proc.stdin.close()
    proc.terminate()

    # Summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Total: {total}, Compiled: {compiled} ({100*compiled/total:.1f}%)", file=sys.stderr)

    for theorem_name, temps in results.items():
        for temp, checks in temps.items():
            n_ok = sum(1 for c in checks if c["compiled"])
            n_total = len(checks)
            # Unique compiling proofs
            ok_proofs = set()
            ok_first_tactics = set()
            for i, c in enumerate(checks):
                if c["compiled"]:
                    proof = data[theorem_name][temp][i]
                    ok_proofs.add(proof)
                    first_tactic = proof.strip().split("\n")[0].strip()
                    ok_first_tactics.add(first_tactic)
            print(f"  {theorem_name} T={temp}: {n_ok}/{n_total} compile, "
                  f"{len(ok_proofs)} unique, {len(ok_first_tactics)} unique first tactics",
                  file=sys.stderr)

    # Write detailed results
    out_path = sys.argv[1].replace(".json", "_checked.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
