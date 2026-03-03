#!/usr/bin/env python3
"""Merge individual Goedel proof files into N large chunks for faster LeanDojo tracing.

LeanDojo's ExtractData spawns one OS process per .lean file, each loading the full
Mathlib environment (~6 GB). For 28K files, this takes 40+ hours. Merging into N
large files (e.g. 32) reduces ExtractData to N processes instead of 28K.

Usage:
    python merge_proofs.py --n-chunks 32 --output-dir /root/goedel_trace/merged
    python merge_proofs.py --n-chunks 32 --output-dir /root/goedel_trace/merged --strict
"""

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
COMPILE_RESULTS = ROOT / "data/logs/goedel_compile/compile_results.json"

# Standard header shared by all proof files
COMMON_HEADER = """\
import Mathlib
import Aesop

set_option maxHeartbeats 0
set_option autoImplicit true

open BigOperators Real Nat Topology Rat
"""


def get_passing_proofs(strict: bool = False) -> list[str]:
    """Return sorted list of seq strings for passing proofs."""
    with open(COMPILE_RESULTS) as f:
        data = json.load(f)
    allowed = ("ok",) if strict else ("ok", "warn")
    seqs = [
        seq for seq, info in data["completed"].items()
        if info["status"] in allowed
    ]
    return sorted(seqs, key=int)


def extract_body(proof_path: Path) -> str:
    """Extract the theorem body (everything after the 6-line header)."""
    lines = proof_path.read_text().splitlines()
    # Skip the 6-line header:
    #   import Mathlib
    #   import Aesop
    #   (blank)
    #   set_option maxHeartbeats 0
    #   (blank)
    #   open BigOperators Real Nat Topology Rat
    body_lines = lines[6:]
    return "\n".join(body_lines)


def merge_proofs(
    proof_dir: Path,
    output_dir: Path,
    n_chunks: int,
    passing_seqs: list[str] | None = None,
):
    """Merge proof files into N chunk files.

    Args:
        proof_dir: Directory containing Proof_XXXXX.lean files.
        output_dir: Directory to write the merged project.
        n_chunks: Number of chunk files to create.
        passing_seqs: Optional list of seq strings to include. If None, include all.
    """
    # Collect proof files
    if passing_seqs is not None:
        proof_files = []
        for seq in passing_seqs:
            fname = f"Proof_{int(seq):05d}.lean"
            fpath = proof_dir / fname
            if fpath.exists():
                proof_files.append(fpath)
            else:
                logger.warning("Missing proof file: %s", fpath)
    else:
        proof_files = sorted(proof_dir.glob("Proof_*.lean"))

    n_proofs = len(proof_files)
    logger.info("Found %d proof files to merge into %d chunks", n_proofs, n_chunks)

    if n_proofs == 0:
        logger.error("No proof files found in %s", proof_dir)
        sys.exit(1)

    # Create output structure
    src_dir = output_dir / "GoedelMerged"
    src_dir.mkdir(parents=True, exist_ok=True)

    # Copy toolchain and create lakefile
    toolchain_src = proof_dir.parent / "lean-toolchain"
    if toolchain_src.exists():
        shutil.copy(toolchain_src, output_dir / "lean-toolchain")
    else:
        (output_dir / "lean-toolchain").write_text("leanprover/lean4:v4.27.0\n")

    (output_dir / "lakefile.lean").write_text(
        'import Lake\nopen Lake DSL\n\n'
        'package goedelMerged where\n'
        '  leanOptions := #[\n'
        '    ⟨`maxHeartbeats, .ofNat 800000⟩\n'
        '  ]\n\n'
        '@[default_target]\n'
        'lean_lib GoedelMerged where\n'
        '  srcDir := "."\n\n'
        'require mathlib from git\n'
        '  "https://github.com/leanprover-community/mathlib4" @ "v4.27.0"\n'
    )

    (output_dir / ".gitignore").write_text(".lake/\nlake-packages/\n")

    # Split proofs into chunks
    chunk_size = math.ceil(n_proofs / n_chunks)
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_proofs)
        if start >= n_proofs:
            break
        chunks.append(proof_files[start:end])

    actual_chunks = len(chunks)
    logger.info("Creating %d chunks (max %d proofs each)", actual_chunks, chunk_size)

    # Write chunk files
    imports = []
    proof_count = 0
    for i, chunk in enumerate(chunks):
        chunk_name = f"Chunk_{i:03d}"
        imports.append(f"import GoedelMerged.{chunk_name}")

        bodies = []
        for pf in chunk:
            body = extract_body(pf)
            if body.strip():
                # Add a section comment for debugging
                bodies.append(f"\n-- ===== {pf.stem} =====")
                bodies.append(body)
                proof_count += 1

        chunk_content = COMMON_HEADER + "\n".join(bodies) + "\n"
        chunk_path = src_dir / f"{chunk_name}.lean"
        chunk_path.write_text(chunk_content)

        logger.info("  %s: %d proofs, %d bytes",
                     chunk_name, len(chunk), len(chunk_content))

    # Write root import file
    (output_dir / "GoedelMerged.lean").write_text("\n".join(imports) + "\n")

    logger.info("Merged %d proofs into %d chunks at %s", proof_count, actual_chunks, output_dir)

    # Init git repo
    subprocess.run(["git", "init"], cwd=output_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "trace@local"],
                    cwd=output_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "trace"],
                    cwd=output_dir, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=output_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Merged Goedel proofs"],
                    cwd=output_dir, check=True, capture_output=True)

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=output_dir, capture_output=True, text=True, check=True,
    )
    commit = result.stdout.strip()
    logger.info("Git repo initialized: %s", commit[:8])

    return output_dir, commit, actual_chunks, proof_count


def main():
    parser = argparse.ArgumentParser(description="Merge Goedel proofs into chunks")
    parser.add_argument("--n-chunks", type=int, default=32,
                        help="Number of chunk files (default: 32)")
    parser.add_argument("--proof-dir", type=str, default=None,
                        help="Directory with Proof_XXXXX.lean files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for merged project")
    parser.add_argument("--strict", action="store_true",
                        help="Only include 'ok' proofs (not 'warn')")
    args = parser.parse_args()

    proof_dir = Path(args.proof_dir) if args.proof_dir else Path("/root/goedel_trace/repo/GoedelMigration")
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        logger.info("Removing existing output dir: %s", output_dir)
        shutil.rmtree(output_dir)

    passing = get_passing_proofs(strict=args.strict)
    logger.info("Passing proofs: %d (strict=%s)", len(passing), args.strict)

    merge_proofs(proof_dir, output_dir, args.n_chunks, passing)


if __name__ == "__main__":
    main()
