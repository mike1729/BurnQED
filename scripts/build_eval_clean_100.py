#!/usr/bin/env python3
"""Build a clean 100-theorem evaluation set from theorem_index.json,
excluding all theorems that appear in any training trajectory or
in the train_eval set. Matches the hypothesis-count distribution
of train_eval_theorems.json. Seed 42 for reproducibility."""

import json
import glob
import random
import statistics
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/root/BurnQED")
random.seed(42)

# ── 1. Load theorem index ──────────────────────────────────────────────
with open(ROOT / "data/theorem_index.json") as f:
    index = json.load(f)
all_theorems = index["theorems"]
print(f"Total theorems in index: {len(all_theorems)}")

# ── 2. Collect theorem names from ALL trajectory parquet files ─────────
trajectory_names = set()
parquet_files = sorted(glob.glob(str(ROOT / "trajectories/iter_*.parquet")))
print(f"\nTrajectory files ({len(parquet_files)}):")
for pf in parquet_files:
    names = pq.read_table(pf, columns=["theorem_name"]).column("theorem_name").to_pylist()
    unique = set(names)
    trajectory_names.update(unique)
    print(f"  {Path(pf).name}: {len(unique)} unique theorems")

# Also check all other parquet files for completeness (baseline, debug, etc.)
all_parquet = sorted(glob.glob(str(ROOT / "trajectories/*.parquet")))
other_parquet = [p for p in all_parquet if p not in parquet_files]
if other_parquet:
    print(f"\nOther trajectory files ({len(other_parquet)}) -- also excluding:")
    for pf in other_parquet:
        names = pq.read_table(pf, columns=["theorem_name"]).column("theorem_name").to_pylist()
        unique = set(names)
        trajectory_names.update(unique)
        print(f"  {Path(pf).name}: {len(unique)} unique theorems")

print(f"\nTotal unique theorem names across all trajectories: {len(trajectory_names)}")

# ── 3. Exclude train_eval theorems ─────────────────────────────────────
with open(ROOT / "data/train_eval_theorems.json") as f:
    train_eval = json.load(f)
train_eval_names = set(t["name"] for t in train_eval["theorems"])
print(f"Train_eval theorems: {len(train_eval_names)}")

excluded_names = trajectory_names | train_eval_names
print(f"Total excluded (union): {len(excluded_names)}")

# ── 4. Build clean pool ────────────────────────────────────────────────
clean_pool = [t for t in all_theorems if t["name"] not in excluded_names]
print(f"Clean pool size: {len(clean_pool)}")

# ── 5. Compute hypothesis count ("depth") for each theorem ─────────────
def hyp_count(statement: str) -> int:
    """Count hypothesis lines (lines before the turnstile ⊢)."""
    lines = statement.strip().split("\n")
    count = 0
    for line in lines:
        if line.strip().startswith("⊢"):
            break
        count += 1
    return count

# Target distribution from train_eval
te_depths = [hyp_count(t["statement"]) for t in train_eval["theorems"]]
te_median = statistics.median(te_depths)
te_mean = statistics.mean(te_depths)
te_q25 = sorted(te_depths)[len(te_depths) // 4]
te_q75 = sorted(te_depths)[3 * len(te_depths) // 4]
print(f"\nTrain_eval depth distribution: mean={te_mean:.1f}, median={te_median}, "
      f"Q25={te_q25}, Q75={te_q75}, min={min(te_depths)}, max={max(te_depths)}")
print(f"Train_eval depth histogram: {dict(sorted(Counter(te_depths).items()))}")

# Annotate clean pool with depth
for t in clean_pool:
    t["_depth"] = hyp_count(t["statement"])

# ── 6. Stratified sampling to match train_eval depth distribution ──────
# Bin depths into buckets matching the train_eval distribution
# Compute the target distribution as proportions
te_depth_counts = Counter(te_depths)
target_total = len(te_depths)  # 198
target_proportions = {d: c / target_total for d, c in te_depth_counts.items()}

# Group clean pool by depth
pool_by_depth = {}
for t in clean_pool:
    d = t["_depth"]
    pool_by_depth.setdefault(d, []).append(t)

# Allocate 100 samples proportionally to train_eval distribution
# Use the same depth buckets, round to nearest integer
TARGET_N = 100
allocations = {}
for d, prop in sorted(target_proportions.items()):
    alloc = round(prop * TARGET_N)
    # Ensure we have enough in pool
    available = len(pool_by_depth.get(d, []))
    allocations[d] = min(alloc, available)

# Adjust to exactly 100: if sum < 100, add from most available buckets
# If sum > 100, remove from largest buckets
current_sum = sum(allocations.values())
print(f"\nInitial allocation sum: {current_sum}")

# Redistribute remaining slots
if current_sum < TARGET_N:
    deficit = TARGET_N - current_sum
    # Add from depths near the median that have available capacity
    # Prefer depths near median (7) that are underrepresented
    candidates = []
    for d, pool in pool_by_depth.items():
        avail = len(pool) - allocations.get(d, 0)
        if avail > 0:
            # Score by closeness to median + availability
            candidates.append((abs(d - te_median), -avail, d))
    candidates.sort()
    for _, _, d in candidates:
        if deficit <= 0:
            break
        avail = len(pool_by_depth[d]) - allocations.get(d, 0)
        add = min(avail, deficit)
        allocations[d] = allocations.get(d, 0) + add
        deficit -= add
elif current_sum > TARGET_N:
    surplus = current_sum - TARGET_N
    # Remove from largest allocations that are farthest from median
    candidates = sorted(allocations.items(), key=lambda x: (-abs(x[0] - te_median), -x[1]))
    for d, count in candidates:
        if surplus <= 0:
            break
        remove = min(count, surplus)
        allocations[d] -= remove
        surplus -= remove

# Remove zero allocations
allocations = {d: c for d, c in allocations.items() if c > 0}
print(f"Final allocation sum: {sum(allocations.values())}")
print(f"Allocation by depth: {dict(sorted(allocations.items()))}")

# Sample from each bucket
sampled = []
for d, count in sorted(allocations.items()):
    pool = pool_by_depth.get(d, [])
    if len(pool) < count:
        print(f"  WARNING: depth {d} has only {len(pool)} theorems, requested {count}")
        count = len(pool)
    chosen = random.sample(pool, count)
    sampled.extend(chosen)
    
random.shuffle(sampled)

# ── 7. Clean up and save ───────────────────────────────────────────────
# Remove the temporary _depth field
for t in sampled:
    del t["_depth"]

output = {"theorems": sampled}
output_path = ROOT / "data/eval_clean_100.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(sampled)} theorems to {output_path}")

# ── 8. Summary ─────────────────────────────────────────────────────────
sample_depths = [hyp_count(t["statement"]) for t in sampled]
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total theorems in index:     {len(all_theorems)}")
print(f"Excluded (trajectories):     {len(trajectory_names)}")
print(f"Excluded (train_eval):       {len(train_eval_names)}")
print(f"Excluded (union):            {len(excluded_names)}")
print(f"Clean pool size:             {len(clean_pool)}")
print(f"Sampled count:               {len(sampled)}")
print(f"")
print(f"Depth distribution of sample:")
print(f"  Mean:   {statistics.mean(sample_depths):.1f}")
print(f"  Median: {statistics.median(sample_depths)}")
print(f"  Q25:    {sorted(sample_depths)[25]}")
print(f"  Q75:    {sorted(sample_depths)[75]}")
print(f"  Min:    {min(sample_depths)}")
print(f"  Max:    {max(sample_depths)}")
print(f"  Histogram: {dict(sorted(Counter(sample_depths).items()))}")
print(f"")
print(f"Train_eval depth distribution (for comparison):")
print(f"  Mean:   {te_mean:.1f}")
print(f"  Median: {te_median}")
print(f"  Histogram: {dict(sorted(Counter(te_depths).items()))}")

# Verify no overlap with excluded sets
sampled_names = set(t["name"] for t in sampled)
assert len(sampled_names) == len(sampled), "Duplicate theorem names in sample!"
assert sampled_names.isdisjoint(trajectory_names), "Sample overlaps with trajectory theorems!"
assert sampled_names.isdisjoint(train_eval_names), "Sample overlaps with train_eval theorems!"
print(f"\nVerification passed: no overlap with excluded theorems.")
