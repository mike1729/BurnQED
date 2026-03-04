"""Extract (state_before, tactic) pairs from Goedel proofs using Pantograph REPL.

Instead of LeanDojo's ExtractData (which re-elaborates everything from source,
taking 2-3h per chunk), this script uses Pantograph to replay proofs one at a
time, capturing tactic states. Each proof takes seconds, not hours.

Usage:
    python extract_pairs_pantograph.py \
        --project-dir /root/goedel_trace/tmp/tmpy_bdi4xg/merged \
        --chunk 0 \
        --output /root/BurnQED/data/traced/goedel_427_pairs.jsonl \
        --workers 8
"""
import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

REPL_BIN = Path(__file__).resolve().parents[3] / "vendor" / "Pantograph" / ".lake" / "build" / "bin" / "repl"
BANNED_RE = re.compile(r"\b(?:sorry|admit|cheat|sorryAx)\b")
PROOF_DELIM = re.compile(r"^-- ===== Proof_(\d+) =====$", re.MULTILINE)


@dataclass
class ParsedProof:
    """A parsed theorem from a chunk file."""
    proof_id: str           # e.g. "00000"
    theorem_name: str       # e.g. "lean_workbook_10009"
    raw_args: str           # "(a b c: ℝ) (ha : ...)"
    arg_names: list[str]    # ["a", "b", "c", "ha"]
    goal_type: str          # "a^3 + b^3 + ... ≥ 1/4"
    proof_body: str         # full tactic block after := by
    source_chunk: int


def parse_binder_names(raw_args: str) -> list[str]:
    """Extract argument names from Lean binder syntax.

    Examples:
        "(a b c : ℝ) (ha : ...)" → ["a", "b", "c", "ha"]
        "{a : ℝ} [inst : Fintype α]" → ["a", "inst"]
        "(n : ℕ)" → ["n"]
    """
    names = []
    # Match each binder group: (...), {...}, [...]
    depth = 0
    current = []
    for ch in raw_args:
        if ch in "({[":
            if depth == 0:
                current = []
            depth += 1
            if depth == 1:
                continue
        elif ch in ")}]":
            depth -= 1
            if depth == 0:
                group = "".join(current).strip()
                if ":" in group:
                    name_part = group.split(":")[0].strip()
                    # Split by whitespace to get individual names
                    for n in name_part.split():
                        if n and n.isidentifier() or n.startswith("h") or n.startswith("_"):
                            names.append(n)
                continue
        if depth >= 1:
            current.append(ch)
    return names


def parse_chunk_file(chunk_path: Path, chunk_idx: int) -> list[ParsedProof]:
    """Parse a merged chunk file into individual proofs."""
    text = chunk_path.read_text()

    # Split by proof delimiter
    parts = PROOF_DELIM.split(text)
    # parts: [header, id0, body0, id1, body1, ...]

    proofs = []
    for i in range(1, len(parts), 2):
        proof_id = parts[i]
        body = parts[i + 1].strip()

        if not body:
            continue

        # Skip sorry/admit contaminated proofs
        # Only check tactic lines, not docstring comments
        thm_idx = body.find("theorem ")
        if thm_idx >= 0:
            proof_part = body[thm_idx:]
            by_idx = proof_part.find(":= by")
            if by_idx >= 0:
                tactic_part = proof_part[by_idx + 5:]
                # Remove block comments before checking
                clean = re.sub(r"/\-.*?\-/", "", tactic_part, flags=re.DOTALL)
                if BANNED_RE.search(clean):
                    continue

        # Extract theorem declaration
        # Handle multi-line declarations with nested parens in type
        thm_match = re.search(
            r"theorem\s+(\w+)\s*((?:\([^)]*\)\s*|\{[^}]*\}\s*|\[[^\]]*\]\s*)*)"
            r"\s*:\s*(.*?)\s*:=\s*by\b(.*)",
            body, re.DOTALL,
        )
        if not thm_match:
            continue

        theorem_name = thm_match.group(1)
        raw_args = thm_match.group(2).strip()
        goal_type = thm_match.group(3).strip()
        proof_body = thm_match.group(4)

        # Parse argument names for intro
        arg_names = parse_binder_names(raw_args)

        proofs.append(ParsedProof(
            proof_id=proof_id,
            theorem_name=theorem_name,
            raw_args=raw_args,
            arg_names=arg_names,
            goal_type=goal_type,
            proof_body=proof_body,
            source_chunk=chunk_idx,
        ))

    return proofs


def extract_toplevel_tactics(proof_body: str) -> list[str]:
    """Extract top-level tactic lines from a proof body.

    Handles:
    - Comment lines (-- ...) → skip
    - Block comments (/- ... -/) → skip
    - Multi-line tactics (continuation lines indented deeper than first tactic)
    - Semicolons chaining (<;>)
    """
    lines = proof_body.split("\n")
    tactics = []
    current_tactic = []
    base_indent = None
    in_block_comment = False

    for line in lines:
        stripped = line.strip()

        # Handle block comments
        if in_block_comment:
            if "-/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/-"):
            if "-/" not in stripped[2:]:
                in_block_comment = True
            continue

        # Skip empty lines and line comments
        if not stripped or stripped.startswith("--"):
            continue

        # Determine indentation
        indent = len(line) - len(line.lstrip())

        if base_indent is None:
            base_indent = indent

        # If same or less indentation as base → new top-level tactic
        if indent <= base_indent and current_tactic:
            tactics.append("\n".join(current_tactic))
            current_tactic = []

        current_tactic.append(stripped)

    if current_tactic:
        tactics.append("\n".join(current_tactic))

    return tactics


def resolve_lean_path(project_dir: str) -> str:
    """Resolve LEAN_PATH once using lake. Must be called from main process where elan is in PATH."""
    elan_bin = Path.home() / ".elan" / "bin"
    env = os.environ.copy()
    env["PATH"] = f"{elan_bin}:{env.get('PATH', '')}"
    result = subprocess.run(
        ["lake", "env", "printenv", "LEAN_PATH"],
        capture_output=True, text=True,
        cwd=project_dir,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"lake env printenv LEAN_PATH failed: {result.stderr}")
    base_path = result.stdout.strip()
    project_lib = str(Path(project_dir) / ".lake" / "build" / "lib" / "lean")
    return f"{base_path}:{project_lib}"


# Module-level cache set by main process before spawning workers
_RESOLVED_LEAN_PATH: Optional[str] = None


class PantographSession:
    """A Pantograph REPL session for replaying proofs."""

    def __init__(self, project_dir: str, timeout: float = 120):
        self.project_dir = project_dir
        self.timeout = timeout
        self.proc: Optional[subprocess.Popen] = None

    def start(self, module: str = "Mathlib"):
        """Start the Pantograph REPL process importing the given module."""
        env = os.environ.copy()
        if _RESOLVED_LEAN_PATH is None:
            raise RuntimeError("LEAN_PATH not resolved. Call resolve_lean_path() first.")
        env["LEAN_PATH"] = _RESOLVED_LEAN_PATH
        # Ensure lean/lake binaries are findable
        elan_bin = str(Path.home() / ".elan" / "bin")
        env["PATH"] = f"{elan_bin}:{env.get('PATH', '')}"

        self.proc = subprocess.Popen(
            [str(REPL_BIN), module],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=self.project_dir,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Wait for "ready." signal
        ready = self.proc.stdout.readline().strip()
        if ready != "ready.":
            raise RuntimeError(f"Pantograph didn't start properly: {ready}")

    def stop(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def send_command(self, cmd: str, payload: dict) -> dict:
        """Send a command to Pantograph and return the JSON response."""
        import select
        line = json.dumps({"cmd": cmd, "payload": payload})
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        # Wait up to timeout seconds for response
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout)
        if not ready:
            raise RuntimeError(f"Pantograph timeout ({self.timeout}s) on {cmd}")
        response = self.proc.stdout.readline().strip()
        if not response:
            raise RuntimeError("Empty response from Pantograph")
        return json.loads(response)

    def goal_start(self, expr: str) -> dict:
        return self.send_command("goal.start", {"expr": expr})

    def goal_tactic(self, state_id: int, tactic: str) -> dict:
        return self.send_command("goal.tactic", {"stateId": state_id, "tactic": tactic})

    def goal_delete(self, state_ids: list[int]):
        return self.send_command("goal.delete", {"stateIds": state_ids})

    def goal_start_copy(self, theorem_name: str) -> dict:
        return self.send_command("goal.start", {"copyFrom": theorem_name})

    @staticmethod
    def format_goals(goals: list[dict]) -> str:
        """Format goals from Pantograph response into tactic state string."""
        parts = []
        for goal in goals:
            var_strs = []
            for var in goal.get("vars", []):
                name = var.get("userName", var.get("name", "?"))
                typ = var.get("type", {}).get("pp", "?")
                var_strs.append(f"{name} : {typ}")

            target = goal.get("target", {}).get("pp", "?")

            if var_strs:
                parts.append("\n".join(var_strs) + "\n⊢ " + target)
            else:
                parts.append("⊢ " + target)

        return "\n\n".join(parts)


def extract_proof_pairs(session: PantographSession, proof: ParsedProof) -> list[dict]:
    """Extract (state_before, tactic) pairs by replaying a proof through Pantograph.

    Uses copyFrom to start from the compiled theorem, then intro's args
    and replays the tactic proof body.
    """
    pairs = []

    # Start goal from compiled theorem
    start_resp = session.goal_start_copy(proof.theorem_name)
    if "error" in start_resp:
        log.debug(f"copyFrom failed for {proof.theorem_name}: {start_resp.get('error')}")
        return []

    state_id = start_resp.get("stateId", 0)
    allocated = [state_id]

    # Intro all arguments to bring them into scope
    if proof.arg_names:
        intro_tactic = "intro " + " ".join(proof.arg_names)
        resp = session.goal_tactic(state_id, intro_tactic)
        if "error" in resp:
            # Fallback: try intros (auto-names)
            resp = session.goal_tactic(state_id, "intros")
            if "error" in resp:
                log.debug(f"intro failed for {proof.theorem_name}: {resp.get('error')}")
                try:
                    session.goal_delete(allocated)
                except Exception:
                    pass
                return []

        state_id = resp.get("nextStateId", state_id)
        allocated.append(state_id)

        # Get state after intro
        goals = resp.get("goals", [])
        state_before = session.format_goals(goals) if goals else proof.goal_type
    else:
        # No args — get initial state via skip
        skip_resp = session.goal_tactic(state_id, "skip")
        if "error" not in skip_resp:
            state_before = session.format_goals(skip_resp.get("goals", []))
            skip_state = skip_resp.get("nextStateId")
            if skip_state is not None:
                allocated.append(skip_state)
        else:
            state_before = proof.goal_type

    # Extract and replay tactics
    tactics = extract_toplevel_tactics(proof.proof_body)

    for depth, tactic in enumerate(tactics):
        try:
            resp = session.goal_tactic(state_id, tactic)
        except Exception:
            break

        if "error" in resp:
            log.debug(f"Tactic error {proof.theorem_name}[{depth}]: {resp.get('error', '')[:80]}")
            break

        new_state = resp.get("nextStateId")
        if new_state is not None:
            allocated.append(new_state)

        # Record the pair
        pairs.append({
            "theorem": proof.theorem_name,
            "state": state_before,
            "tactic": tactic,
            "depth": depth,
            "source": "goedel_workbook",
        })

        # Check if proof is complete
        goals = resp.get("goals", [])
        if not goals:
            break

        state_before = session.format_goals(goals)
        state_id = new_state if new_state is not None else state_id

    # Cleanup allocated states
    try:
        session.goal_delete(allocated)
    except Exception:
        pass

    return pairs


def process_chunk(
    chunk_idx: int,
    project_dir: str,
    output_dir: str,
    reuse_session: bool = True,
) -> dict:
    """Process a single chunk file, extracting all (state, tactic) pairs."""
    chunk_path = Path(project_dir) / "GoedelMerged" / f"Chunk_{chunk_idx:03d}.lean"
    if not chunk_path.exists():
        log.error(f"Chunk file not found: {chunk_path}")
        return {"chunk": chunk_idx, "pairs": 0, "theorems": 0, "errors": 0}

    log.info(f"Parsing chunk {chunk_idx}...")
    proofs = parse_chunk_file(chunk_path, chunk_idx)
    log.info(f"Chunk {chunk_idx}: {len(proofs)} proofs parsed")

    all_pairs = []
    errors = 0
    theorem_count = 0

    # Import the chunk module so we can use copyFrom
    chunk_module = f"GoedelMerged.Chunk_{chunk_idx:03d}"

    if reuse_session:
        # Use a single Pantograph session for all proofs in the chunk
        session = PantographSession(project_dir)
        try:
            session.start(module=chunk_module)
        except Exception as e:
            log.error(f"Failed to start Pantograph for {chunk_module}: {e}")
            return {"chunk": chunk_idx, "pairs": 0, "theorems": 0, "errors": len(proofs)}

        for i, proof in enumerate(proofs):
            try:
                pairs = extract_proof_pairs(session, proof)
                if pairs:
                    all_pairs.extend(pairs)
                    theorem_count += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                log.debug(f"Error on {proof.theorem_name}: {e}")
                # Restart session on error
                session.stop()
                try:
                    session.start(module=chunk_module)
                except Exception:
                    log.error(f"Failed to restart Pantograph at proof {i}")
                    break

            if (i + 1) % 100 == 0:
                log.info(f"  Chunk {chunk_idx}: {i+1}/{len(proofs)} proofs, {len(all_pairs)} pairs, {errors} errors")

        session.stop()
    else:
        # Use a new session per proof (slower but more robust)
        for i, proof in enumerate(proofs):
            session = PantographSession(project_dir)
            try:
                session.start(module=chunk_module)
                pairs = extract_proof_pairs(session, proof)
                if pairs:
                    all_pairs.extend(pairs)
                    theorem_count += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
            finally:
                session.stop()

            if (i + 1) % 50 == 0:
                log.info(f"  Chunk {chunk_idx}: {i+1}/{len(proofs)} proofs, {len(all_pairs)} pairs")

    # Write output
    out_path = Path(output_dir) / f"chunk_{chunk_idx:03d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    log.info(f"Chunk {chunk_idx}: {theorem_count} theorems, {len(all_pairs)} pairs, {errors} errors → {out_path}")
    return {"chunk": chunk_idx, "pairs": len(all_pairs), "theorems": theorem_count, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Extract tactic pairs via Pantograph")
    parser.add_argument("--project-dir", required=True, help="Path to built Lean project")
    parser.add_argument("--chunk", type=int, default=None, help="Process single chunk (0-15)")
    parser.add_argument("--output-dir", default="data/traced/pantograph_pairs", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel chunk workers")
    parser.add_argument("--test", type=int, default=None, help="Test with N proofs from chunk 0")
    args = parser.parse_args()

    if not REPL_BIN.exists():
        log.error(f"Pantograph REPL not found at {REPL_BIN}")
        sys.exit(1)

    project_dir = args.project_dir

    # Resolve LEAN_PATH once in main process
    global _RESOLVED_LEAN_PATH
    _RESOLVED_LEAN_PATH = resolve_lean_path(project_dir)
    log.info(f"LEAN_PATH resolved: {_RESOLVED_LEAN_PATH[:80]}...")

    if args.test is not None:
        # Quick test mode
        chunk_path = Path(project_dir) / "GoedelMerged" / "Chunk_000.lean"
        proofs = parse_chunk_file(chunk_path, 0)[:args.test]
        log.info(f"Test mode: {len(proofs)} proofs")

        session = PantographSession(project_dir)
        session.start(module="GoedelMerged.Chunk_000")

        total_pairs = 0
        errors = 0
        for proof in proofs:
            pairs = extract_proof_pairs(session, proof)
            if pairs:
                total_pairs += len(pairs)
                for p in pairs:
                    print(json.dumps(p, ensure_ascii=False))
            else:
                errors += 1
                log.warning(f"No pairs for {proof.theorem_name}")

        session.stop()
        log.info(f"Test complete: {total_pairs} pairs from {len(proofs)} proofs ({errors} errors)")
        return

    # Determine chunks to process
    if args.chunk is not None:
        chunks = [args.chunk]
    else:
        # Find all chunks
        chunks = []
        for f in sorted(Path(project_dir).glob("GoedelMerged/Chunk_*.lean")):
            idx = int(f.stem.split("_")[1])
            chunks.append(idx)

    log.info(f"Processing {len(chunks)} chunks with {args.workers} workers")

    if args.workers <= 1 or len(chunks) == 1:
        # Sequential
        results = []
        for idx in chunks:
            r = process_chunk(idx, project_dir, args.output_dir)
            results.append(r)
    else:
        # Parallel (one worker per chunk, each with its own Pantograph session)
        # Use fork context so _RESOLVED_LEAN_PATH is inherited
        results = []
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(process_chunk, idx, project_dir, args.output_dir): idx
                for idx in chunks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    r = future.result()
                    results.append(r)
                except Exception as e:
                    log.error(f"Chunk {idx} failed: {e}")
                    results.append({"chunk": idx, "pairs": 0, "theorems": 0, "errors": -1})

    # Summary
    total_pairs = sum(r["pairs"] for r in results)
    total_theorems = sum(r["theorems"] for r in results)
    total_errors = sum(r["errors"] for r in results)

    log.info(f"\n=== Summary ===")
    log.info(f"Theorems: {total_theorems}")
    log.info(f"Pairs: {total_pairs}")
    log.info(f"Errors: {total_errors}")
    log.info(f"Output: {args.output_dir}/")

    # Merge all chunk files into one
    if len(chunks) > 1:
        merged_path = Path(args.output_dir) / "goedel_427_pairs.jsonl"
        with open(merged_path, "w") as out:
            for idx in sorted(chunks):
                chunk_file = Path(args.output_dir) / f"chunk_{idx:03d}.jsonl"
                if chunk_file.exists():
                    with open(chunk_file) as f:
                        out.write(f.read())
        log.info(f"Merged: {merged_path}")


if __name__ == "__main__":
    main()
