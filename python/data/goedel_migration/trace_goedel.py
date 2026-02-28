#!/usr/bin/env python3
"""M.5 LeanDojo tracing of Goedel proofs migrated to Lean 4.27.

Traces compiled Goedel proofs to extract (state_before, tactic) pairs
for SFT training. Uses LeanDojo's trace() on a local git repo.

The script supports two modes:
  --test N     Create a small test repo with N proofs and trace it (~minutes)
  --full       Trace the full goedel_migration repo (~10-15h)

The goedel_migration directory must be a git repo (run --init-repo first
if needed). For --test mode, a temporary mini-repo is created automatically.

Usage:
    # Initialize the main repo (one-time setup):
    python python/data/goedel_migration/trace_goedel.py --init-repo

    # Test run with 5 proofs:
    python python/data/goedel_migration/trace_goedel.py --test 5

    # Full trace (after test validation):
    python python/data/goedel_migration/trace_goedel.py --full

Output:
    data/traced/goedel_427_pairs.jsonl  (or goedel_427_test_pairs.jsonl for --test)
"""

import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure elan-managed lake/lean are on PATH
ELAN_BIN = Path.home() / ".elan" / "bin"
if ELAN_BIN.is_dir() and str(ELAN_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{ELAN_BIN}:{os.environ.get('PATH', '')}"

# Point LeanDojo cache and temp dir to /workspace to avoid filling overlay fs
WORKSPACE_CACHE = Path("/workspace/.cache/lean_dojo")
WORKSPACE_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CACHE_DIR", str(WORKSPACE_CACHE))
WORKSPACE_TMP = Path("/workspace/.cache/lean_dojo_tmp")
WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMP_DIR", str(WORKSPACE_TMP))

ROOT = Path(__file__).resolve().parents[3]
GOEDEL_DIR = ROOT / "data/lean/goedel_migration"
PROOF_DIR = GOEDEL_DIR / "GoedelMigration"
COMPILE_RESULTS = ROOT / "data/logs/goedel_compile/compile_results.json"
MANIFEST = GOEDEL_DIR / "goedel_manifest.json"
INTEGRITY_REPORT = ROOT / "data/traced/integrity_report.json"
OUTPUT_DIR = ROOT / "data/traced"

BANNED_RE = re.compile(r"\b(?:sorry|admit|cheat|sorryAx)\b")


def get_passing_seqs() -> dict:
    """Load compile results and return {seq: info} for passing proofs."""
    with open(COMPILE_RESULTS) as f:
        data = json.load(f)
    return {
        seq: info
        for seq, info in data["completed"].items()
        if info["status"] in ("ok", "warn")
    }


def get_manifest_map() -> dict:
    """Load manifest and return {seq_str: entry}."""
    with open(MANIFEST) as f:
        entries = json.load(f)
    return {str(e["seq"]): e for e in entries}


def init_goedel_repo():
    """Initialize goedel_migration as a git repo if not already one."""
    git_dir = GOEDEL_DIR / ".git"
    if git_dir.exists():
        logger.info("goedel_migration is already a git repo")
        # Get current HEAD
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=GOEDEL_DIR, capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("HEAD: %s", result.stdout.strip())
            return result.stdout.strip()
        else:
            logger.warning("Git repo exists but no commits yet, will commit")

    logger.info("Initializing git repo in %s", GOEDEL_DIR)
    subprocess.run(["git", "init"], cwd=GOEDEL_DIR, check=True)
    subprocess.run(
        ["git", "config", "user.email", "trace@local"],
        cwd=GOEDEL_DIR, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "trace"],
        cwd=GOEDEL_DIR, check=True,
    )

    # Add essential files: lakefile, lean-toolchain, GoedelMigration.lean, proof files
    # Exclude .lake/ build artifacts
    logger.info("Staging files (this may take a moment for 30K files)...")
    subprocess.run(
        ["git", "add", "lakefile.lean", "lean-toolchain", "GoedelMigration.lean"],
        cwd=GOEDEL_DIR, check=True,
    )
    subprocess.run(
        ["git", "add", "GoedelMigration/"],
        cwd=GOEDEL_DIR, check=True,
    )

    logger.info("Creating initial commit...")
    result = subprocess.run(
        ["git", "commit", "-m", "Goedel proofs migrated to Lean 4.27"],
        cwd=GOEDEL_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("git commit failed: %s", result.stderr)
        sys.exit(1)

    # Get commit hash
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=GOEDEL_DIR, capture_output=True, text=True, check=True,
    )
    commit = result.stdout.strip()
    logger.info("Committed: %s", commit)
    return commit


def create_test_repo(n_proofs: int, passing_seqs: dict) -> tuple:
    """Create a minimal Lean project with N proof files for test tracing.

    Returns (tmpdir_path, commit_hash).
    """
    # Pick N passing proofs, spread across the range
    sorted_seqs = sorted(passing_seqs.keys(), key=int)
    step = max(1, len(sorted_seqs) // n_proofs)
    selected = sorted_seqs[:n_proofs * step:step][:n_proofs]

    logger.info("Selected %d proofs for test: seqs %s", len(selected),
                [int(s) for s in selected[:10]])

    tmpdir = tempfile.mkdtemp(prefix="goedel_trace_test_")
    tmpdir = Path(tmpdir)
    proj_dir = tmpdir / "GoedelTest"
    src_dir = proj_dir / "GoedelTest"
    src_dir.mkdir(parents=True)

    # Copy lean-toolchain
    shutil.copy(GOEDEL_DIR / "lean-toolchain", proj_dir / "lean-toolchain")

    # Write lakefile.lean
    (proj_dir / "lakefile.lean").write_text(
        'import Lake\nopen Lake DSL\n\n'
        'package goedelTest where\n'
        '  leanOptions := #[\n'
        '    ⟨`autoImplicit, false⟩,\n'
        '    ⟨`maxHeartbeats, 800000⟩\n'
        '  ]\n\n'
        '@[default_target]\n'
        'lean_lib GoedelTest where\n'
        '  srcDir := "."\n\n'
        'require mathlib from git\n'
        '  "https://github.com/leanprover-community/mathlib4" @ "v4.27.0"\n'
    )

    # Write root import file
    imports = []
    for seq in selected:
        info = passing_seqs[seq]
        old_module = info["module"]  # GoedelMigration.Proof_XXXXX
        new_module = old_module.replace("GoedelMigration", "GoedelTest")
        imports.append(f"import {new_module}")

        # Copy proof file, fixing the module reference if needed
        src_name = old_module.split(".")[-1] + ".lean"
        src_file = PROOF_DIR / src_name
        dst_file = src_dir / src_name
        shutil.copy(src_file, dst_file)

    (proj_dir / "GoedelTest.lean").write_text("\n".join(imports) + "\n")

    # Build dependencies (reuse lake-packages from main project if possible)
    logger.info("Building test project dependencies...")
    lake_packages = GOEDEL_DIR / ".lake"
    if lake_packages.exists():
        # Symlink .lake to avoid re-downloading Mathlib
        os.symlink(lake_packages, proj_dir / ".lake")
        logger.info("Symlinked .lake from main project")

    # Init git repo
    subprocess.run(["git", "init"], cwd=proj_dir, check=True,
                    capture_output=True)
    subprocess.run(["git", "config", "user.email", "trace@local"],
                    cwd=proj_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "trace"],
                    cwd=proj_dir, check=True, capture_output=True)

    # Write .gitignore to exclude .lake
    (proj_dir / ".gitignore").write_text(".lake/\nlake-packages/\n")

    subprocess.run(["git", "add", "."], cwd=proj_dir, check=True,
                    capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "test proofs"],
        cwd=proj_dir, check=True, capture_output=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=proj_dir, capture_output=True, text=True, check=True,
    )
    commit = result.stdout.strip()
    logger.info("Test repo created at %s (commit %s)", proj_dir, commit[:8])
    return proj_dir, commit, selected


def trace_repo(repo_path: Path, commit: str, dst_dir: Path = None):
    """Run LeanDojo trace on a local Lean git repo.

    Returns a TracedRepo object.
    """
    from lean_dojo import LeanGitRepo, trace

    repo_path_str = str(repo_path)
    logger.info("Creating LeanGitRepo for %s @ %s", repo_path_str, commit[:8])
    repo = LeanGitRepo(repo_path_str, commit)
    logger.info("Lean version: %s", repo.lean_version)

    logger.info("Starting trace (this may take a while)...")
    traced_repo = trace(repo, dst_dir=dst_dir)
    logger.info("Trace complete!")
    return traced_repo


def extract_pairs(traced_repo, manifest_map: dict, source_label: str = "goedel_workbook") -> list:
    """Extract (state, tactic) pairs from a TracedRepo.

    Returns list of dicts in unified tactic pair format.
    """
    pairs = []
    contaminated = 0
    errors = 0
    theorems_seen = set()

    for traced_file in traced_repo.traced_files:
        for thm in traced_file.traced_theorems:
            if not thm.has_tactic_proof:
                continue

            theorem_name = thm.theorem.full_name
            theorems_seen.add(theorem_name)

            try:
                traced_tactics = thm.get_traced_tactics()
            except Exception as e:
                errors += 1
                if errors <= 10:
                    logger.warning("Error extracting tactics from %s: %s",
                                   theorem_name, e)
                continue

            # Check for contamination in the full theorem
            theorem_contaminated = False
            for tt in traced_tactics:
                tactic_text = tt.tactic
                if BANNED_RE.search(tactic_text):
                    theorem_contaminated = True
                    contaminated += 1
                    logger.warning("Contaminated theorem %s: tactic contains banned keyword: %s",
                                   theorem_name, tactic_text[:80])
                    break

            if theorem_contaminated:
                continue

            for i, tt in enumerate(traced_tactics):
                state_before = tt.state_before
                tactic_text = tt.tactic

                if not state_before.strip() or not tactic_text.strip():
                    continue

                pairs.append({
                    "theorem": theorem_name,
                    "state": state_before,
                    "tactic": tactic_text,
                    "depth": i,
                    "source": source_label,
                    "num_goals": state_before.count("⊢"),
                })

    logger.info(
        "Extracted %d pairs from %d theorems (%d contaminated, %d errors)",
        len(pairs), len(theorems_seen), contaminated, errors,
    )
    return pairs


def write_pairs(pairs: list, output_path: Path):
    """Write tactic pairs as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info("Wrote %d pairs to %s", len(pairs), output_path)


def print_summary(pairs: list):
    """Print summary statistics of extracted pairs."""
    if not pairs:
        print("\nNo pairs extracted!")
        return

    theorems = set(p["theorem"] for p in pairs)
    depths = [p["depth"] for p in pairs]
    max_depth = max(depths)
    tactic_lengths = [len(p["tactic"]) for p in pairs]

    print("\n" + "=" * 60)
    print("  TRACING SUMMARY")
    print("=" * 60)
    print(f"  Theorems:          {len(theorems):>8}")
    print(f"  Tactic pairs:      {len(pairs):>8}")
    print(f"  Pairs/theorem:     {len(pairs)/len(theorems):>8.1f}")
    print(f"  Max depth:         {max_depth:>8}")
    print(f"  Depth distribution:")

    from collections import Counter
    depth_counts = Counter(depths)
    for d in sorted(depth_counts.keys()):
        print(f"    depth {d}: {depth_counts[d]}")

    print(f"  Avg tactic length: {sum(tactic_lengths)/len(tactic_lengths):>8.1f} chars")
    print(f"  Max tactic length: {max(tactic_lengths):>8}")

    # Multi-goal stats
    multi_goal = sum(1 for p in pairs if p["num_goals"] > 1)
    print(f"  Multi-goal states: {multi_goal:>8} ({multi_goal/len(pairs)*100:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Trace Goedel proofs with LeanDojo to extract tactic pairs",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--test", type=int, metavar="N",
        help="Test mode: trace N proofs in a temporary repo",
    )
    mode.add_argument(
        "--full", action="store_true",
        help="Full mode: trace all passing proofs",
    )
    mode.add_argument(
        "--init-repo", action="store_true",
        help="Initialize goedel_migration as a git repo (one-time setup)",
    )
    parser.add_argument(
        "--dst-dir", type=str, default=None,
        help="Directory to save LeanDojo trace cache",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: auto-generated)",
    )
    args = parser.parse_args()

    if args.init_repo:
        commit = init_goedel_repo()
        print(f"Repo initialized. Commit: {commit}")
        return

    passing_seqs = get_passing_seqs()
    manifest_map = get_manifest_map()
    logger.info("Loaded %d passing proofs", len(passing_seqs))

    if args.test:
        n = args.test
        proj_dir, commit, selected_seqs = create_test_repo(n, passing_seqs)
        output_path = Path(args.output) if args.output else OUTPUT_DIR / "goedel_427_test_pairs.jsonl"

        try:
            traced_repo = trace_repo(proj_dir, commit, dst_dir=args.dst_dir)
            pairs = extract_pairs(traced_repo, manifest_map)
            write_pairs(pairs, output_path)
            print_summary(pairs)
        finally:
            # Clean up test repo
            logger.info("Cleaning up test repo at %s", proj_dir)
            shutil.rmtree(proj_dir.parent, ignore_errors=True)

    elif args.full:
        # Ensure goedel_migration is a git repo
        git_dir = GOEDEL_DIR / ".git"
        if not git_dir.exists():
            logger.error(
                "goedel_migration is not a git repo. Run --init-repo first."
            )
            sys.exit(1)

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=GOEDEL_DIR, capture_output=True, text=True, check=True,
        )
        commit = result.stdout.strip()
        output_path = Path(args.output) if args.output else OUTPUT_DIR / "goedel_427_pairs.jsonl"

        traced_repo = trace_repo(GOEDEL_DIR, commit, dst_dir=args.dst_dir)
        pairs = extract_pairs(traced_repo, manifest_map)
        write_pairs(pairs, output_path)
        print_summary(pairs)


if __name__ == "__main__":
    main()
