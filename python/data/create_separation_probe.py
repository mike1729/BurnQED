#!/usr/bin/env python3
"""Create a separation probe dataset from trajectory parquets.

Samples balanced positive/negative proof states at depth > 1 for monitoring
embedding separation during LLM training. Used by SeparationProbeCallback
in train_llm.py.

Usage:
    python python/data/create_separation_probe.py \
        --trajectories trajectories/iter_1_negatives.parquet trajectories/iter_1.parquet \
        --output data/separation_probe.json
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(
        description="Create separation probe dataset from trajectory parquets."
    )
    parser.add_argument(
        "--trajectories",
        nargs="+",
        required=True,
        help="Parquet file paths (glob-expanded by shell)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=200,
        help="Number of samples per class (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: %(default)s)",
    )
    args = parser.parse_args()

    positives = []
    negatives = []
    seen = set()

    for path in args.trajectories:
        p = Path(path)
        if not p.exists():
            print(f"WARNING: {path} does not exist, skipping")
            continue

        table = pq.read_table(str(p), columns=["state_pp", "label", "depth_from_root"])
        df = table.to_pandas()

        # Filter: depth > 1, non-empty state_pp
        df = df[df["depth_from_root"] > 1]
        df = df[df["state_pp"].astype(str).str.strip().ne("")]

        for _, row in df.iterrows():
            state = str(row["state_pp"]).strip()
            if state in seen:
                continue
            seen.add(state)

            label = str(row["label"])
            if label == "positive":
                positives.append(state)
            elif label == "negative":
                negatives.append(state)

        print(f"  {path}: {len(df)} rows at depth>1, running total: {len(positives)} pos, {len(negatives)} neg")

    print(f"\nTotal unique: {len(positives)} positive, {len(negatives)} negative")

    # Sample
    rng = random.Random(args.seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    n = args.n_per_class
    if len(positives) < n:
        print(f"WARNING: only {len(positives)} positives available (requested {n})")
        n_pos = len(positives)
    else:
        n_pos = n

    if len(negatives) < n:
        print(f"WARNING: only {len(negatives)} negatives available (requested {n})")
        n_neg = len(negatives)
    else:
        n_neg = n

    samples = []
    for state in positives[:n_pos]:
        samples.append({"state_pp": state, "label": "positive"})
    for state in negatives[:n_neg]:
        samples.append({"state_pp": state, "label": "negative"})

    # Shuffle the combined output
    rng.shuffle(samples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\nWrote {len(samples)} samples ({n_pos} pos + {n_neg} neg) to {args.output}")


if __name__ == "__main__":
    main()
