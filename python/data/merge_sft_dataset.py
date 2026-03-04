"""Merge Goedel + LEAN-GitHub pairs into unified SFT dataset.

Reads:
  data/traced/pantograph_pairs/goedel_427_pairs.jsonl  (60K pairs)
  data/traced/lean_github_pairs.jsonl                  (197K pairs)

Produces:
  data/sft/train.jsonl   — SFT training data (95% of theorems)
  data/sft/val.jsonl     — SFT validation data (5% of theorems)
  data/sft/contrastive_pool.json — theorem names with depth >= 3

Format matches docs/data_format_spec.md exactly.
Split is by THEOREM NAME (Gotcha 11), not by tactic pair.
"""
import hashlib
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

VAL_FRACTION = 0.05
SEED_PREFIX = "burn-qed-v2-split"  # deterministic split


def format_sft_pair(state: str, tactic: str) -> str:
    """Format a (state, tactic) pair for SFT training.

    MUST match the Rust policy crate's format_tactic_message() exactly.
    See docs/data_format_spec.md.
    """
    return (
        f"Complete the following Lean 4 code:\n\n"
        f"```lean4\n"
        f"/- tactic state:\n"
        f"{state}\n"
        f"-/\n"
        f"```\n"
        f"{tactic}"
    )


def theorem_to_split(theorem_name: str, source: str) -> str:
    """Deterministic train/val assignment by theorem name.

    Uses hash to ensure reproducibility without random seed.
    Source-prefixed to handle potential name collisions.
    """
    key = f"{SEED_PREFIX}:{source}:{theorem_name}"
    h = hashlib.md5(key.encode()).hexdigest()
    # Use first 8 hex chars as fraction [0, 1)
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return "val" if frac < VAL_FRACTION else "train"


def main():
    out_dir = Path("data/sft")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load both sources
    goedel_path = Path("data/traced/pantograph_pairs/goedel_427_pairs.jsonl")
    github_path = Path("data/traced/lean_github_pairs.jsonl")

    all_pairs = []
    theorem_max_depth = {}  # (source, theorem) -> max depth

    for path, label in [(goedel_path, "goedel"), (github_path, "lean_github")]:
        log.info(f"Loading {path}...")
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                all_pairs.append(rec)
                key = (rec["source"], rec["theorem"])
                theorem_max_depth[key] = max(theorem_max_depth.get(key, 0), rec["depth"] + 1)

    log.info(f"Total pairs loaded: {len(all_pairs)}")
    log.info(f"Unique theorems: {len(theorem_max_depth)}")

    # Split by theorem name
    train_f = open(out_dir / "train.jsonl", "w")
    val_f = open(out_dir / "val.jsonl", "w")

    train_count = 0
    val_count = 0
    train_theorems = set()
    val_theorems = set()

    for rec in all_pairs:
        split = theorem_to_split(rec["theorem"], rec["source"])
        text = format_sft_pair(rec["state"], rec["tactic"])

        out_rec = {
            "text": text,
            "theorem": rec["theorem"],
            "source": rec["source"],
            "depth": rec["depth"],
        }
        line = json.dumps(out_rec, ensure_ascii=False) + "\n"

        if split == "train":
            train_f.write(line)
            train_count += 1
            train_theorems.add((rec["source"], rec["theorem"]))
        else:
            val_f.write(line)
            val_count += 1
            val_theorems.add((rec["source"], rec["theorem"]))

    train_f.close()
    val_f.close()

    # Build contrastive pool (theorems with depth >= 3)
    contrastive_pool = []
    for (source, theorem), max_d in theorem_max_depth.items():
        if max_d >= 3:
            contrastive_pool.append({
                "theorem": theorem,
                "source": source,
                "max_depth": max_d,
            })

    with open(out_dir / "contrastive_pool.json", "w") as f:
        json.dump(contrastive_pool, f, indent=2)

    # Summary
    log.info(f"\n=== Summary ===")
    log.info(f"Train: {train_count} pairs from {len(train_theorems)} theorems")
    log.info(f"Val:   {val_count} pairs from {len(val_theorems)} theorems")
    log.info(f"Val fraction: {val_count/(train_count+val_count):.3f}")
    log.info(f"Contrastive pool (depth>=3): {len(contrastive_pool)} theorems")

    # Per-source breakdown
    from collections import Counter
    train_sources = Counter()
    val_sources = Counter()
    for rec in all_pairs:
        split = theorem_to_split(rec["theorem"], rec["source"])
        if split == "train":
            train_sources[rec["source"]] += 1
        else:
            val_sources[rec["source"]] += 1

    log.info(f"\nPer-source breakdown:")
    for src in sorted(set(list(train_sources.keys()) + list(val_sources.keys()))):
        log.info(f"  {src}: train={train_sources[src]}, val={val_sources[src]}")

    log.info(f"\nOutput: {out_dir}/")


if __name__ == "__main__":
    main()
