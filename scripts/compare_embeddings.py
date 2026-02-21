#!/usr/bin/env python3
"""Compare iter_2 vs iter_3 embeddings — memory-efficient streaming."""

import sys
import os
import numpy as np
import pyarrow.parquet as pq

ITER2 = "checkpoints/ebm/iter_2/embeddings.parquet"
ITER3 = "checkpoints/ebm/iter_3/embeddings.parquet"
BATCH = 5_000  # rows per batch — conservative for memory

def load_state_index(path):
    """Load only state_pp strings → row index mapping (no embeddings)."""
    pf = pq.ParquetFile(path)
    idx = {}
    offset = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=["state_pp"]):
        for i, s in enumerate(batch.column("state_pp").to_pylist()):
            idx[s] = offset + i
        offset += len(batch)
    return idx

def load_embeddings_for_rows(path, row_to_key):
    """Load embeddings for specific row indices. Returns dict key→np.array."""
    pf = pq.ParquetFile(path)
    result = {}
    offset = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=["embedding"]):
        batch_len = len(batch)
        emb_col = batch.column("embedding")
        for row_idx, key in row_to_key.items():
            if offset <= row_idx < offset + batch_len:
                local = row_idx - offset
                emb = emb_col[local].as_py()
                result[key] = np.array(emb, dtype=np.float32)
        offset += batch_len
        if len(result) == len(row_to_key):
            break
    return result

def streaming_stats(path, max_rows=10_000):
    """Compute norm/magnitude stats without loading all embeddings."""
    pf = pq.ParquetFile(path)
    norms = []
    means = []
    stds = []
    n = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=["embedding"]):
        for emb_list in batch.column("embedding").to_pylist():
            v = np.array(emb_list, dtype=np.float32)
            norms.append(np.linalg.norm(v))
            means.append(v.mean())
            stds.append(v.std())
            n += 1
        if n >= max_rows:
            break
    return np.array(norms), np.array(means), np.array(stds), n

print("=" * 70)
print("EMBEDDING COMPARISON: iter_2 vs iter_3")
print("=" * 70)

# --- Phase 1: Basic stats (streaming, sampled) ---
print("\n--- Phase 1: Embedding Distribution Stats (first 10K each) ---")

for name, path in [("iter_2", ITER2), ("iter_3", ITER3)]:
    norms, means, stds, n = streaming_stats(path, max_rows=10_000)
    print(f"\n{name} (sampled {n:,} rows, dim={4096}):")
    print(f"  L2 norm:    mean={norms.mean():.3f}  std={norms.std():.3f}  "
          f"min={norms.min():.3f}  max={norms.max():.3f}")
    print(f"  Elem mean:  mean={means.mean():.6f}  std={means.std():.6f}")
    print(f"  Elem std:   mean={stds.mean():.6f}  std={stds.std():.6f}")
    del norms, means, stds

# --- Phase 2: Find overlapping states ---
print("\n--- Phase 2: Overlapping States ---")
print("Loading iter_2 state index...")
idx2 = load_state_index(ITER2)
print(f"  iter_2: {len(idx2):,} unique states")

print("Loading iter_3 state index...")
idx3 = load_state_index(ITER3)
print(f"  iter_3: {len(idx3):,} unique states")

overlap_keys = set(idx2.keys()) & set(idx3.keys())
print(f"  Overlap: {len(overlap_keys):,} states in both")
print(f"  iter_2 only: {len(idx2) - len(overlap_keys):,}")
print(f"  iter_3 only: {len(idx3) - len(overlap_keys):,}")

if len(overlap_keys) == 0:
    print("No overlapping states — cannot compare paired embeddings.")
    sys.exit(0)

# Sample up to 3000 for pairwise comparison (memory conservative)
sample_keys = list(overlap_keys)
np.random.seed(42)
if len(sample_keys) > 3000:
    indices = np.random.choice(len(sample_keys), 3000, replace=False)
    sample_keys = [sample_keys[i] for i in indices]
print(f"  Sampling {len(sample_keys):,} for pairwise comparison")

# Build row lookups for just the sample
rows2 = {idx2[k]: k for k in sample_keys}
rows3 = {idx3[k]: k for k in sample_keys}
del idx2, idx3, overlap_keys

# --- Phase 3: Load paired embeddings ---
print("\n--- Phase 3: Paired Embedding Comparison ---")
print("Loading iter_2 embeddings for sample...")
embs2 = load_embeddings_for_rows(ITER2, rows2)
print(f"  Loaded {len(embs2):,}")
del rows2

print("Loading iter_3 embeddings for sample...")
embs3 = load_embeddings_for_rows(ITER3, rows3)
print(f"  Loaded {len(embs3):,}")
del rows3

# Compute pairwise metrics
l2_dists = []
cosine_sims = []
for key in sample_keys:
    if key not in embs2 or key not in embs3:
        continue
    v2, v3 = embs2[key], embs3[key]
    l2_dists.append(np.linalg.norm(v2 - v3))
    cos = np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3) + 1e-8)
    cosine_sims.append(cos)

l2_dists = np.array(l2_dists)
cosine_sims = np.array(cosine_sims)

print(f"\nSame-state embeddings across backbones ({len(l2_dists):,} pairs):")
print(f"  L2 distance:       mean={l2_dists.mean():.3f}  std={l2_dists.std():.3f}  "
      f"median={np.median(l2_dists):.3f}")
print(f"  Cosine similarity: mean={cosine_sims.mean():.4f}  std={cosine_sims.std():.4f}  "
      f"median={np.median(cosine_sims):.4f}")
print(f"  Cosine < 0.5: {(cosine_sims < 0.5).sum()} / {len(cosine_sims)} "
      f"({100*(cosine_sims < 0.5).mean():.1f}%)")
print(f"  Cosine < 0.9: {(cosine_sims < 0.9).sum()} / {len(cosine_sims)} "
      f"({100*(cosine_sims < 0.9).mean():.1f}%)")
print(f"  Cosine > 0.95: {(cosine_sims > 0.95).sum()} / {len(cosine_sims)} "
      f"({100*(cosine_sims > 0.95).mean():.1f}%)")

del embs2, embs3

# --- Phase 4: Within-iteration separability ---
# Use trajectories to identify proved vs failed states
print("\n--- Phase 4: Intra-set Separability (proved vs failed) ---")

TRAJ_FILES = {
    "iter_2": "trajectories/iter_1.parquet",  # iter_2 EBM trained on iter_1 trajectories
    "iter_3": "trajectories/iter_2.parquet",   # iter_3 EBM trained on iter_2 trajectories
}

for name, emb_path in [("iter_2", ITER2), ("iter_3", ITER3)]:
    traj_path = TRAJ_FILES.get(name)
    if not traj_path or not os.path.exists(traj_path):
        print(f"\n{name}: trajectory {traj_path} not found, skipping")
        continue

    # Load trajectory states
    traj_pf = pq.ParquetFile(traj_path)
    schema = traj_pf.schema_arrow
    print(f"\n{name}: Loading trajectory {traj_path}")
    print(f"  Trajectory schema columns: {[f.name for f in schema]}")

    # Read label and state columns
    # label: "positive" = on proof path, "hard_negative" / "easy_negative" = not
    proved_states = set()
    failed_states = set()
    for batch in traj_pf.iter_batches(batch_size=5000, columns=["state_pp", "label"]):
        states = batch.column("state_pp").to_pylist()
        labels = batch.column("label").to_pylist()
        for s, lbl in zip(states, labels):
            if s:
                if lbl == "positive":
                    proved_states.add(s)
                else:
                    failed_states.add(s)
        if len(proved_states) >= 3000 and len(failed_states) >= 3000:
            break

    print(f"  Found {len(proved_states):,} proved, {len(failed_states):,} failed states")

    # Match with embedding index
    state_idx = load_state_index(emb_path)
    proved_match = {s: state_idx[s] for s in list(proved_states)[:1000] if s in state_idx}
    failed_match = {s: state_idx[s] for s in list(failed_states)[:1000] if s in state_idx}
    del state_idx, proved_states, failed_states

    print(f"  Matched in embeddings: {len(proved_match):,} proved, {len(failed_match):,} failed")

    if len(proved_match) < 50 or len(failed_match) < 50:
        print("  Too few matches, skipping separability analysis")
        continue

    # Sample 300 each
    n_sample = min(300, len(proved_match), len(failed_match))
    p_keys = list(proved_match.keys())[:n_sample]
    f_keys = list(failed_match.keys())[:n_sample]

    p_rows = {proved_match[k]: k for k in p_keys}
    f_rows = {failed_match[k]: k for k in f_keys}
    del proved_match, failed_match

    p_embs = load_embeddings_for_rows(emb_path, p_rows)
    f_embs = load_embeddings_for_rows(emb_path, f_rows)
    del p_rows, f_rows

    p_vecs = np.array([p_embs[k] for k in p_keys if k in p_embs])
    f_vecs = np.array([f_embs[k] for k in f_keys if k in f_embs])
    del p_embs, f_embs

    # Norm stats per group
    p_norms = np.linalg.norm(p_vecs, axis=1)
    f_norms = np.linalg.norm(f_vecs, axis=1)
    print(f"  Proved norms:  mean={p_norms.mean():.3f}  std={p_norms.std():.3f}")
    print(f"  Failed norms:  mean={f_norms.mean():.3f}  std={f_norms.std():.3f}")
    print(f"  Norm diff:     {abs(p_norms.mean() - f_norms.mean()):.4f}")

    # Centroid distance
    p_centroid = p_vecs.mean(axis=0)
    f_centroid = f_vecs.mean(axis=0)
    centroid_l2 = np.linalg.norm(p_centroid - f_centroid)
    centroid_cos = np.dot(p_centroid, f_centroid) / (np.linalg.norm(p_centroid) * np.linalg.norm(f_centroid) + 1e-8)
    print(f"  Centroid L2 dist:  {centroid_l2:.4f}")
    print(f"  Centroid cosine:   {centroid_cos:.6f}")

    # Random pairwise cosine similarities
    np.random.seed(42)
    n_pairs = min(2000, len(p_vecs) * len(f_vecs))
    idx_p = np.random.randint(0, len(p_vecs), n_pairs)
    idx_f = np.random.randint(0, len(f_vecs), n_pairs)
    idx_p2 = np.random.randint(0, len(p_vecs), n_pairs)

    def batch_cosine(a, b):
        dots = (a * b).sum(axis=1)
        norms_ab = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return dots / (norms_ab + 1e-8)

    cross_cos = batch_cosine(p_vecs[idx_p], f_vecs[idx_f])
    within_p_cos = batch_cosine(p_vecs[idx_p], p_vecs[idx_p2])

    print(f"  Within-proved cosine:  mean={within_p_cos.mean():.4f}  std={within_p_cos.std():.4f}")
    print(f"  Cross (P↔F) cosine:   mean={cross_cos.mean():.4f}  std={cross_cos.std():.4f}")
    sep = within_p_cos.mean() - cross_cos.mean()
    print(f"  Separation (Δ cosine): {sep:.6f}  {'(positive=good)' if sep > 0 else '(negative=BAD)'}")

    del p_vecs, f_vecs

print("\n" + "=" * 70)
print("DONE")
