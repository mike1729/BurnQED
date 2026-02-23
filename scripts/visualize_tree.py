#!/usr/bin/env python3
"""Visualize proof search trees from trajectory parquet files.

Usage:
    python scripts/visualize_tree.py trajectories/debug.parquet
    python scripts/visualize_tree.py trajectories/debug.parquet --theorem "Nat.add_comm"
    python scripts/visualize_tree.py trajectories/debug.parquet --max-theorems 3
    python scripts/visualize_tree.py trajectories/debug.parquet --json  # machine-readable
"""
import argparse
import json
import sys
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq


def load_records(path, theorem_name=None):
    """Load trajectory records, optionally filtered to one theorem."""
    df = pq.read_table(path).to_pandas()
    if theorem_name:
        df = df[df["theorem_name"] == theorem_name]
    return df


def build_tree(records):
    """Build tree structure from records for a single theorem."""
    nodes = {}
    children = defaultdict(list)

    for _, r in records.iterrows():
        sid = r["state_id"]
        nodes[sid] = {
            "state_id": int(sid),
            "tactic": r["tactic_applied"],
            "label": r["label"],
            "depth": int(r["depth_from_root"]),
            "remaining": int(r["remaining_depth"]),
            "llm_log_prob": float(r["llm_log_prob"]),
            "ebm_score": float(r["ebm_score"]),
            "combined": float(r["llm_log_prob"]) + float(r["ebm_score"]),
            "is_qed": bool(r["is_proof_complete"]),
            "state_pp": r["state_pp"],
            "parent_id": int(r["parent_state_id"]) if not pd.isna(r["parent_state_id"]) else None,
        }
        pid = r["parent_state_id"]
        if not pd.isna(pid):
            children[int(pid)].append(int(sid))

    return nodes, children


def truncate_state(state_pp, max_len=80):
    """Truncate proof state for display."""
    s = state_pp.replace("\n", " ↵ ")
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s


def print_tree(nodes, children, node_id, indent=0, prefix=""):
    """Recursively print tree with scores."""
    node = nodes[node_id]

    # Color/marker based on label
    if node["is_qed"]:
        marker = "★ QED"
    elif node["label"] == "positive":
        marker = "✓"
    elif node["label"] == "negative":
        marker = "✗"
    else:
        marker = "?"

    # Format tactic
    tactic = node["tactic"] if node["tactic"] else "(root)"

    # Score string
    llm = node["llm_log_prob"]
    ebm = node["ebm_score"]
    score_str = f"llm={llm:+.3f} ebm={ebm:+.3f} combined={llm + ebm:+.3f}"

    # State preview
    state_str = truncate_state(node["state_pp"])

    # Print this node
    connector = prefix
    print(f"{connector}[{marker}] d={node['depth']} {tactic}")
    detail_prefix = " " * len(connector) + "    "
    print(f"{detail_prefix}{score_str}")
    print(f"{detail_prefix}{state_str}")

    # Print children
    kids = children.get(node_id, [])
    # Sort children by combined score descending
    kids.sort(key=lambda c: nodes[c]["llm_log_prob"] + nodes[c]["ebm_score"], reverse=True)

    for i, child_id in enumerate(kids):
        is_last = i == len(kids) - 1
        child_prefix = " " * indent + ("└── " if is_last else "├── ")
        next_indent = indent + 4
        print_tree(nodes, children, child_id, next_indent, child_prefix)


def print_theorem_summary(theorem_name, records):
    """Print summary stats for a theorem."""
    n = len(records)
    proved = records["is_proof_complete"].any()
    pos = (records["label"] == "positive").sum()
    neg = (records["label"] == "negative").sum()
    max_depth = records["depth_from_root"].max()
    llm_range = f"[{records['llm_log_prob'].min():.3f}, {records['llm_log_prob'].max():.3f}]"
    ebm_range = f"[{records['ebm_score'].min():.3f}, {records['ebm_score'].max():.3f}]"

    status = "PROVED" if proved else "FAILED"
    print(f"\n{'=' * 80}")
    print(f"  {theorem_name}  [{status}]")
    print(f"{'=' * 80}")
    print(f"  Nodes: {n}  |  Positive: {pos}  |  Negative: {neg}  |  Max depth: {max_depth}")
    print(f"  LLM log-prob range: {llm_range}")
    print(f"  EBM score range:    {ebm_range}")

    if proved:
        # Extract proof path
        proof_path = records[records["label"] == "positive"].sort_values("depth_from_root")
        tactics = [t for t in proof_path["tactic_applied"] if t]
        print(f"  Proof ({len(tactics)} steps): {' → '.join(tactics)}")

    print()


def tree_to_dict(nodes, children, node_id):
    """Convert tree to nested dict for JSON output."""
    node = nodes[node_id]
    result = {
        "state_id": node["state_id"],
        "tactic": node["tactic"] or None,
        "label": node["label"],
        "depth": node["depth"],
        "llm_log_prob": node["llm_log_prob"],
        "ebm_score": node["ebm_score"],
        "combined_score": node["llm_log_prob"] + node["ebm_score"],
        "is_qed": node["is_qed"],
        "state_pp": node["state_pp"],
    }
    kids = children.get(node_id, [])
    if kids:
        kids.sort(key=lambda c: nodes[c]["llm_log_prob"] + nodes[c]["ebm_score"], reverse=True)
        result["children"] = [tree_to_dict(nodes, children, c) for c in kids]
    return result


def print_score_stats(df):
    """Print score distribution and zero-embedding analysis."""
    probes = {'simp','ring','omega','norm_num','decide','trivial','rfl','tauto',
              'linarith','push_neg','contradiction','exfalso','constructor','left',
              'right','ext','simp_all'}

    total = len(df)
    roots = df["tactic_applied"].isna() | (df["tactic_applied"] == "")
    non_root = df[~roots]

    is_probe = non_root["tactic_applied"].isin(probes)
    probe_nodes = is_probe.sum()
    llm_nodes = len(non_root) - probe_nodes

    zero_ebm = (df["ebm_score"] == 0.0).sum()
    nonzero = df[df["ebm_score"] != 0.0]

    proved = df.groupby("theorem_name")["is_proof_complete"].any()
    n_proved = proved.sum()
    n_total = len(proved)

    print(f"\n{'=' * 60}")
    print(f"  SCORE STATISTICS ({n_total} theorems, {n_proved} proved)")
    print(f"{'=' * 60}")
    print(f"  Total nodes:     {total}")
    print(f"  Probe nodes:     {probe_nodes} ({100*probe_nodes/total:.0f}%)")
    print(f"  LLM nodes:       {llm_nodes} ({100*llm_nodes/total:.0f}%)")
    print(f"  Root nodes:      {roots.sum()}")
    print()
    print(f"  Zero EBM scores: {zero_ebm}/{total} ({100*zero_ebm/total:.1f}%)")
    print(f"  Non-zero EBM:    {len(nonzero)}")
    print()
    if len(nonzero) > 0:
        print(f"  EBM (non-zero):  median={nonzero['ebm_score'].median():.3f}  "
              f"mean={nonzero['ebm_score'].mean():.3f}  "
              f"range=[{nonzero['ebm_score'].min():.3f}, {nonzero['ebm_score'].max():.3f}]")
    print(f"  LLM log-prob:    median={df['llm_log_prob'].median():.3f}  "
          f"mean={df['llm_log_prob'].mean():.3f}  "
          f"range=[{df['llm_log_prob'].min():.3f}, {df['llm_log_prob'].max():.3f}]")

    # Per-theorem summary table
    print(f"\n  {'Theorem':<30} {'Nodes':>5} {'Proved':>6} {'Zero EBM':>9} {'EBM med':>8} {'LLM med':>8}")
    print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*9} {'-'*8} {'-'*8}")
    for thm in sorted(df["theorem_name"].unique()):
        sub = df[df["theorem_name"] == thm]
        p = "yes" if sub["is_proof_complete"].any() else "no"
        z = (sub["ebm_score"] == 0.0).sum()
        nz = sub[sub["ebm_score"] != 0.0]
        ebm_med = f"{nz['ebm_score'].median():.3f}" if len(nz) > 0 else "n/a"
        llm_med = f"{sub['llm_log_prob'].median():.3f}"
        print(f"  {thm:<30} {len(sub):>5} {p:>6} {z:>4}/{len(sub):<4} {ebm_med:>8} {llm_med:>8}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize proof search trees")
    parser.add_argument("input", help="Trajectory parquet file")
    parser.add_argument("--theorem", "-t", help="Filter to specific theorem")
    parser.add_argument("--max-theorems", "-n", type=int, default=10,
                        help="Max theorems to display (default: 10)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-tree", action="store_true",
                        help="Only print summary, skip tree rendering")
    parser.add_argument("--max-children", type=int, default=None,
                        help="Limit children shown per node (show top-N by score)")
    parser.add_argument("--stats", action="store_true",
                        help="Print score distribution and zero-embedding analysis")
    args = parser.parse_args()

    df = load_records(args.input, args.theorem)
    if df.empty:
        print(f"No records found" + (f" for theorem '{args.theorem}'" if args.theorem else ""))
        sys.exit(1)

    if args.stats:
        print_score_stats(df)
        if args.no_tree:
            return

    theorems = df["theorem_name"].unique()
    if len(theorems) > args.max_theorems:
        theorems = theorems[:args.max_theorems]
        print(f"(showing {args.max_theorems} of {df['theorem_name'].nunique()} theorems)\n")

    if args.json:
        output = []
        for thm in theorems:
            records = df[df["theorem_name"] == thm]
            nodes, children = build_tree(records)
            roots = [sid for sid, n in nodes.items() if n["parent_id"] is None]
            if roots:
                tree = tree_to_dict(nodes, children, roots[0])
                tree["theorem_name"] = thm
                tree["total_nodes"] = len(nodes)
                tree["proved"] = bool(records["is_proof_complete"].any())
                output.append(tree)
        json.dump(output, sys.stdout, indent=2)
        print()
        return

    for thm in theorems:
        records = df[df["theorem_name"] == thm]
        print_theorem_summary(thm, records)

        if not args.no_tree:
            nodes, children_map = build_tree(records)

            # Apply max-children filter
            if args.max_children:
                for pid in list(children_map.keys()):
                    kids = children_map[pid]
                    kids.sort(key=lambda c: nodes[c]["llm_log_prob"] + nodes[c]["ebm_score"], reverse=True)
                    if len(kids) > args.max_children:
                        trimmed = len(kids) - args.max_children
                        children_map[pid] = kids[:args.max_children]
                        # Add a placeholder
                        print(f"  (trimmed {trimmed} children from node {pid})")

            roots = [sid for sid, n in nodes.items() if n["parent_id"] is None]
            for root_id in roots:
                print_tree(nodes, children_map, root_id)


if __name__ == "__main__":
    main()
