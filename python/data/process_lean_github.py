"""Process LEAN-GitHub dataset: filter, compute depth, convert to SFT-ready JSONL.

Input:  data/lean/lean_github/lean-github.parquet (218K rows)
Output: data/traced/lean_github_pairs.jsonl

Filtering:
1. Sorry/admit/cheat: reject entire theorem if any tactic is contaminated
2. State length: drop rows where state_before > 4096 chars
3. Trivial tactic subsampling: keep only 10% of single-word trivial tactics
4. Compute depth per theorem by grouping on (url, file_path, full_name, start)
"""
import json
import logging
import random
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BANNED_WORDS = {"sorry", "admit", "cheat", "sorryAx"}
TRIVIAL_TACTICS = {"rfl", "trivial", "decide", "norm_num", "ring", "omega", "simp", "aesop"}
MAX_STATE_LEN = 4096
TRIVIAL_KEEP_RATE = 0.10
SEED = 42


def main():
    random.seed(SEED)

    input_path = Path("data/lean/lean_github/lean-github.parquet")
    output_path = Path("data/traced/lean_github_pairs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    log.info(f"Loaded {len(df)} rows")

    # --- 1. Identify contaminated theorems ---
    # Use (url, file_path, full_name, start) as theorem key
    df["thm_key"] = df["url"] + "|" + df["file_path"] + "|" + df["full_name"] + "|" + df["start"]

    contaminated_keys = set()
    for _, row in df.iterrows():
        tactic = row["tactic"]
        for word in BANNED_WORDS:
            if word in tactic:
                contaminated_keys.add(row["thm_key"])
                break

    n_before = len(df)
    df = df[~df["thm_key"].isin(contaminated_keys)]
    log.info(f"Sorry filter: {n_before} → {len(df)} rows ({n_before - len(df)} removed, {len(contaminated_keys)} contaminated theorems)")

    # --- 2. State length filter ---
    n_before = len(df)
    df = df[df["state_before"].str.len() <= MAX_STATE_LEN]
    log.info(f"State length filter (>{MAX_STATE_LEN}): {n_before} → {len(df)} rows ({n_before - len(df)} removed)")

    # --- 3. Trivial tactic subsampling ---
    n_before = len(df)
    is_trivial = df["tactic"].str.strip().isin(TRIVIAL_TACTICS)
    keep_mask = ~is_trivial | (pd.Series([random.random() < TRIVIAL_KEEP_RATE for _ in range(len(df))], index=df.index))
    df = df[keep_mask]
    log.info(f"Trivial subsampling ({TRIVIAL_KEEP_RATE}): {n_before} → {len(df)} rows ({n_before - len(df)} removed)")

    # --- 4. Compute depth within each theorem ---
    # Sort by theorem key and position to ensure correct ordering
    df = df.sort_values(["thm_key", "start", "end"])
    df["depth"] = df.groupby("thm_key").cumcount()

    # --- 5. Convert to output format ---
    log.info("Writing output...")
    n_theorems = df["thm_key"].nunique()
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            record = {
                "theorem": row["full_name"],
                "state": row["state_before"],
                "tactic": row["tactic"],
                "depth": int(row["depth"]),
                "source": "lean_github",
                "repo": row["url"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(df)
    log.info(f"\n=== Summary ===")
    log.info(f"Total pairs: {total}")
    log.info(f"Unique theorems: {n_theorems}")
    log.info(f"Avg pairs/theorem: {total/n_theorems:.1f}")

    # Depth stats
    thm_depths = df.groupby("thm_key")["depth"].max() + 1
    log.info(f"Median depth: {thm_depths.median():.0f}")
    log.info(f"Mean depth: {thm_depths.mean():.1f}")
    log.info(f"Theorems depth >= 3: {(thm_depths >= 3).sum()} ({(thm_depths >= 3).sum()*100/len(thm_depths):.1f}%)")
    log.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
