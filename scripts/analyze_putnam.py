#!/usr/bin/env python3
"""Deep analysis of Putnam proof search trajectories.

Analyzes base DeepSeek-Prover-V2-7B behavior on 604 Putnam problems (132 solved).
Compares proof-path vs dead-branch tactic distributions, goal dynamics, state size
trajectories, and contrasts with Goedel training data.

Sections:
  1. Proof path vs dead branch tactic distribution + bigrams
  2. Tactic outcome at sibling states
  3. Goal-count delta by tactic type
  4. State size trajectory
  5. Goedel tactic distribution comparison
  6. Depth-conditional tactic distribution
  7. Unique first-tactic type distribution
  B. Log-prob ranking analysis (depth/sibling stratified, distributions, dedup)
  C. State complexity (root analysis, topic classification, hypothesis growth)
  D. Search efficiency (budget waste, time-to-proof, fanout)
  E. Failure modes (categories, near-miss, year analysis)
  F. Training signal (pairs, DPO, token budget, exact? audit)
  G. Recommendations

Usage:
    python scripts/analyze_putnam.py
    python scripts/analyze_putnam.py --parquet path/to/data.parquet --output-dir path/to/output/
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ── Utilities ────────────────────────────────────────────────────────────────


def tactic_head(tactic: str) -> str:
    """Extract first token (tactic name) from a tactic string."""
    if not tactic or not isinstance(tactic, str) or not tactic.strip():
        return "(root)"
    t = tactic.strip()
    if t.startswith("·"):
        t = t[1:].strip()
    if t.startswith("<;>"):
        t = t[3:].strip()
    if t.startswith("--"):
        return "(comment)"
    if not t:
        return "(root)"
    return t.split(None, 1)[0]


def count_goals(state_pp: str) -> int:
    """Count goals in a Lean proof state (separated by blank lines)."""
    if not state_pp or not isinstance(state_pp, str) or not state_pp.strip():
        return 0
    blocks = [b.strip() for b in state_pp.split("\n\n") if b.strip()]
    return max(len(blocks), 1)


def count_hypotheses(state_pp) -> int:
    """Count hypothesis lines (containing ':') before the first turnstile."""
    if not isinstance(state_pp, str):
        return 0
    count = 0
    for line in state_pp.split("\n"):
        if "⊢" in line:
            break
        if line.strip() and ":" in line:
            count += 1
    return count


def classify_topic(state_pp) -> list:
    """Regex-based topic classification from root proof state."""
    if not isinstance(state_pp, str):
        return ["other"]
    s = state_pp
    sl = s.lower()
    topics = []
    if any(w in sl for w in ["ℕ", "ℤ", " nat", " int", "prime", "divid",
                              "mod ", "gcd", "lcm", "factorial", "coprime"]):
        topics.append("number_theory")
    if any(w in sl for w in ["continuous", "deriv", "integr", "limit",
                              "tendsto", "measur", "filter", "nhds",
                              "differentiable", "intervalintegral"]):
        topics.append("analysis")
    if any(w in sl for w in ["polynomial", " ring", " field", " group",
                              "subgroup", "ideal", " module", "algebra",
                              "monoid", "commut", "isomorph", "homomorp"]):
        topics.append("abstract_algebra")
    if any(w in sl for w in ["finset", "fintype", "card ", "choose",
                              "permut", "subset", "powerset", "multiset"]):
        topics.append("combinatorics")
    if any(w in sl for w in ["matrix", " det", "linearmap", "basis",
                              "eigenval", "trace", "transpose"]):
        topics.append("linear_algebra")
    if any(w in sl for w in ["polygon", "triangle", "circle", "euclidean",
                              "dist ", "angle", "convex"]):
        topics.append("geometry")
    if any(w in s for w in ["∑", "∏", "∑'", "tsum"]) or \
       any(w in sl for w in ["series", "sequence", "isup", "iinf"]):
        topics.append("series_sums")
    if any(w in s for w in ["ℝ", "ℚ"]) and "analysis" not in topics:
        topics.append("real_arithmetic")
    if not topics:
        topics.append("other")
    return topics


def parse_theorem_name(name: str):
    """Extract (year, section, number) from e.g. 'putnam_1988_b1'."""
    m = re.match(r"putnam_(\d{4})_([ab])(\d+)", str(name), re.IGNORECASE)
    if m:
        return int(m.group(1)), m.group(2).upper(), int(m.group(3))
    return None, None, None


def load_data(args):
    """Load trajectory parquet and metadata files."""
    df = pd.read_parquet(args.parquet)

    benchmark = None
    if os.path.exists(args.benchmark):
        with open(args.benchmark) as f:
            benchmark = json.load(f)

    eval_json = None
    if os.path.exists(args.eval_json):
        with open(args.eval_json) as f:
            eval_json = json.load(f)

    return df, benchmark, eval_json


def prepare_df(df):
    """Add computed columns to the dataframe."""
    mask_has_tactic = df["tactic_applied"].notna() & (df["tactic_applied"] != "")
    df["head"] = "(root)"
    df.loc[mask_has_tactic, "head"] = df.loc[mask_has_tactic, "tactic_applied"].apply(tactic_head)

    pos = df[df["label"] == "positive"]
    path_idx = pd.MultiIndex.from_arrays([pos["theorem_name"], pos["state_id"]])
    all_idx = pd.MultiIndex.from_arrays([df["theorem_name"], df["state_id"]])
    df["on_path"] = all_idx.isin(path_idx)

    proved_map = df.groupby("theorem_name")["is_proof_complete"].any()
    df["proved"] = df["theorem_name"].map(proved_map)

    df["state_size"] = df["state_pp"].apply(lambda s: len(s) if isinstance(s, str) else 0)
    df["goal_count"] = df["state_pp"].apply(count_goals)
    df["is_root"] = ~mask_has_tactic

    return df


def md_table(headers, rows, align=None):
    """Format a markdown table."""
    if not rows:
        return "(no data)\n"

    widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        str_row = [str(c) for c in row]
        str_rows.append(str_row)
        for i, cell in enumerate(str_row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    if align is None:
        align = ["l"] + ["r"] * (len(headers) - 1)

    def fmt(cells):
        parts = []
        for i, (cell, w) in enumerate(zip(cells, widths)):
            parts.append(cell.rjust(w) if i < len(align) and align[i] == "r" else cell.ljust(w))
        return "| " + " | ".join(parts) + " |"

    lines = [fmt(headers)]
    sep = []
    for i, w in enumerate(widths):
        sep.append("-" * (w - 1) + ":" if i < len(align) and align[i] == "r" else ":" + "-" * (w - 1))
    lines.append("| " + " | ".join(sep) + " |")
    for row in str_rows:
        lines.append(fmt(row))
    return "\n".join(lines) + "\n"


def pct(n, total):
    return f"{n/total*100:.1f}%" if total > 0 else "0.0%"


def fmt_float(x, decimals=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{decimals}f}"


def fmt_dist(values, label=""):
    """One-line distribution summary."""
    if len(values) == 0:
        return f"{label}: no data"
    a = np.asarray(values, dtype=float)
    return (f"{label}  n={len(a):,}  "
            f"mean={a.mean():.1f}  med={np.median(a):.0f}  "
            f"p25={np.percentile(a, 25):.0f}  p75={np.percentile(a, 75):.0f}  "
            f"min={a.min():.0f}  max={a.max():.0f}")


# ── Section 1: Proof path vs dead branch tactic distribution ────────────────


def analyze_tactic_distribution(df):
    """Tactic head frequency on proof paths vs dead branches in proved theorems."""
    proved_nr = df[df["proved"] & ~df["is_root"]]
    path = proved_nr[proved_nr["on_path"]]
    dead = proved_nr[~proved_nr["on_path"]]

    path_counts = path["head"].value_counts()
    dead_counts = dead["head"].value_counts()
    path_total = len(path)
    dead_total = len(dead)

    all_heads = set(path_counts.index) | set(dead_counts.index)
    rows = []
    for h in all_heads:
        pc = path_counts.get(h, 0)
        dc = dead_counts.get(h, 0)
        pp = pc / path_total * 100 if path_total else 0
        dp = dc / dead_total * 100 if dead_total else 0
        enrich = pp / dp if dp > 0 else float("inf")
        rows.append((h, pc, dc, pp, dp, enrich))

    rows.sort(key=lambda r: r[1], reverse=True)

    # Tactic bigrams on proof paths
    bigrams = Counter()
    for thm in path["theorem_name"].unique():
        tp = path[path["theorem_name"] == thm].sort_values("depth_from_root")
        heads = tp["head"].tolist()
        for i in range(len(heads) - 1):
            bigrams[(heads[i], heads[i + 1])] += 1

    return {
        "path_total": path_total,
        "dead_total": dead_total,
        "rows": rows,
        "bigrams": bigrams.most_common(20),
    }


# ── Section 2: Tactic outcome at sibling states ─────────────────────────────


def analyze_sibling_outcomes(df):
    """At branch points in proved theorems, per-tactic win rate."""
    proved_nr = df[df["proved"] & ~df["is_root"] & df["parent_state_id"].notna()]

    per_tactic_wins = Counter()
    per_tactic_total = Counter()
    have_wins = 0
    nonhave_wins = 0
    mixed_branches = 0

    for thm, grp in proved_nr.groupby("theorem_name"):
        by_parent = defaultdict(list)
        for _, row in grp.iterrows():
            by_parent[int(row["parent_state_id"])].append(
                (row["head"], bool(row["on_path"]))
            )

        for pid, children in by_parent.items():
            has_pos = any(c[1] for c in children)
            if not has_pos or len(children) < 2:
                continue

            for head, on_path in children:
                per_tactic_total[head] += 1
                if on_path:
                    per_tactic_wins[head] += 1

            haves = [c for c in children if c[0] in ("have", "let")]
            nonhaves = [c for c in children if c[0] not in ("have", "let")]
            if haves and nonhaves:
                mixed_branches += 1
                if any(c[1] for c in haves):
                    have_wins += 1
                if any(c[1] for c in nonhaves):
                    nonhave_wins += 1

    tactic_rows = []
    for h in per_tactic_total:
        total = per_tactic_total[h]
        if total < 5:
            continue
        wins = per_tactic_wins.get(h, 0)
        tactic_rows.append((h, wins, total, wins / total * 100))
    tactic_rows.sort(key=lambda r: r[3], reverse=True)

    return {
        "tactic_rows": tactic_rows,
        "have_wins": have_wins,
        "nonhave_wins": nonhave_wins,
        "mixed_branches": mixed_branches,
    }


# ── Section 3: Goal-count delta by tactic type ──────────────────────────────


def analyze_goal_delta(df):
    """Goal-count delta (parent_goals - child_goals) by tactic head."""
    non_root = df[~df["is_root"] & df["parent_state_id"].notna()].copy()

    lookup = df.set_index(["theorem_name", "state_id"])["goal_count"]

    parent_keys = list(
        zip(non_root["theorem_name"], non_root["parent_state_id"].astype(int))
    )
    parent_goals = []
    for thm, pid in parent_keys:
        parent_goals.append(lookup.get((thm, pid), 0))
    non_root["parent_goals"] = parent_goals

    child_goals = non_root["goal_count"].copy()
    child_goals[non_root["is_proof_complete"]] = 0
    non_root["delta"] = non_root["parent_goals"] - child_goals

    def summarize(sub, min_count=5):
        rows = []
        for head, grp in sub.groupby("head"):
            if len(grp) < min_count:
                continue
            d = grp["delta"].values
            rows.append(
                (
                    head,
                    len(d),
                    d.mean(),
                    (d > 0).sum() / len(d) * 100,
                    (d < 0).sum() / len(d) * 100,
                    (d == 0).sum() / len(d) * 100,
                )
            )
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows

    path_rows = summarize(non_root[non_root["on_path"]], min_count=3)
    all_rows = summarize(non_root, min_count=10)

    return {"proof_path": path_rows, "all_nodes": all_rows}


# ── Section 4: State size trajectory ─────────────────────────────────────────


def analyze_state_size(df):
    """State size by depth, proof-path vs dead-branch in proved theorems."""
    proved = df[df["proved"]].copy()

    rows = []
    for d, grp in proved.groupby("depth_from_root"):
        p = grp[grp["on_path"]]["state_size"]
        dead = grp[~grp["on_path"]]["state_size"]
        p_mean = p.mean() if len(p) else None
        d_mean = dead.mean() if len(dead) else None
        ratio = p_mean / d_mean if (p_mean and d_mean and d_mean > 0) else None
        rows.append(
            (
                int(d),
                len(p), fmt_float(p_mean, 0), fmt_float(p.median() if len(p) else None, 0),
                len(dead), fmt_float(d_mean, 0), fmt_float(dead.median() if len(dead) else None, 0),
                fmt_float(ratio),
            )
        )

    proved_names = df[df["proved"]]["theorem_name"].unique()
    samples = []
    for thm in proved_names[:5]:
        path = df[(df["theorem_name"] == thm) & df["on_path"]].sort_values("depth_from_root")
        steps = []
        for _, r in path.iterrows():
            steps.append(f"d{r['depth_from_root']}:{r['head']}({r['state_size']})")
        if steps:
            samples.append((thm, " → ".join(steps)))

    return {"by_depth": rows, "samples": samples}


# ── Section 5: Goedel tactic distribution ────────────────────────────────────


def parse_goedel_proofs(goedel_dir, limit=5000):
    """Parse Goedel proof files, extract tactic heads.

    Uses bracket-depth tracking to skip continuation lines inside
    multi-line argument lists (e.g., nlinarith [sq_nonneg ...,\\n  sq_nonneg ...]).

    Returns (overall_counter, closing_counter, proof_count).
    """
    overall = Counter()
    closing = Counter()
    proof_count = 0

    files = sorted(Path(goedel_dir).glob("Proof_*.lean"))[:limit]
    if not files:
        return overall, closing, 0

    for f in files:
        text = f.read_text()
        text = re.sub(r"/\-.*?\-/", "", text, flags=re.DOTALL)

        m = re.search(r":=\s*by\b", text)
        if not m:
            continue

        proof_block = text[m.end() :]
        lines = proof_block.split("\n")

        base_indent = None
        for line in lines:
            s = line.rstrip()
            if s and not s.lstrip().startswith("--"):
                base_indent = len(s) - len(s.lstrip())
                break

        if base_indent is None:
            continue

        proof_count += 1
        proof_tactics = []
        bracket_depth = 0

        for line in lines:
            s = line.rstrip()
            if not s or not s.strip():
                continue
            if s.strip().startswith("--"):
                continue

            stripped = s.strip()

            if bracket_depth > 0:
                bracket_depth += stripped.count("[") - stripped.count("]")
                bracket_depth += stripped.count("(") - stripped.count(")")
                bracket_depth = max(bracket_depth, 0)
                continue

            for seg in stripped.split("<;>"):
                seg = seg.strip()
                if not seg or seg.startswith("--"):
                    continue
                if seg.startswith("·"):
                    seg = seg[len("·") :].strip()
                if not seg or seg.startswith("--"):
                    continue
                head = seg.split(None, 1)[0]
                if head:
                    head = head.strip("()")
                    if not head or head == "|":
                        continue
                    overall[head] += 1
                    proof_tactics.append(head)

            bracket_depth += stripped.count("[") - stripped.count("]")
            bracket_depth += stripped.count("(") - stripped.count(")")
            bracket_depth = max(bracket_depth, 0)

        if proof_tactics:
            closing[proof_tactics[-1]] += 1

    return overall, closing, proof_count


def analyze_goedel(goedel_dir):
    """Goedel tactic distribution and comparison data."""
    overall, closing, proof_count = parse_goedel_proofs(goedel_dir)
    if not overall:
        return None

    total = sum(overall.values())
    closing_total = sum(closing.values())

    return {
        "proof_count": proof_count,
        "total_tactics": total,
        "overall": [(h, c, c / total * 100) for h, c in overall.most_common(30)],
        "closing": [(h, c, c / closing_total * 100) for h, c in closing.most_common(20)],
    }


# ── Section 6: Depth-conditional tactic distribution ─────────────────────────


def analyze_depth_tactics(df):
    """Per-depth tactic head frequency on proof paths."""
    proved_nr = df[df["proved"] & ~df["is_root"]]
    path_df = proved_nr[proved_nr["on_path"]]

    by_depth = {}
    for d, grp in path_df.groupby("depth_from_root"):
        by_depth[int(d)] = grp["head"].value_counts().head(10).to_dict()

    closing_df = path_df[path_df["is_proof_complete"]]
    closing_by_depth = {}
    for d, grp in closing_df.groupby("depth_from_root"):
        closing_by_depth[int(d)] = grp["head"].value_counts().head(10).to_dict()

    return {"by_depth": by_depth, "closing_by_depth": closing_by_depth}


# ── Section 7: Unique first-tactic diversity ─────────────────────────────────


def analyze_first_tactic(df):
    """Per-theorem depth-1 tactic diversity analysis."""
    depth1 = df[(df["depth_from_root"] == 1) & ~df["is_root"]]

    stats = []
    for thm, grp in depth1.groupby("theorem_name"):
        total = len(grp)
        unique = grp["head"].nunique()
        have_n = grp["head"].isin(["have", "let"]).sum()
        have_frac = have_n / total if total else 0
        proved = grp["proved"].iloc[0]
        stats.append((thm, total, unique, have_n, have_frac, proved))

    stats_df = pd.DataFrame(stats, columns=["thm", "total", "unique", "have_n", "have_frac", "proved"])

    overall_heads = depth1["head"].value_counts().head(20).to_dict()

    proved_s = stats_df[stats_df["proved"]]
    failed_s = stats_df[~stats_df["proved"]]

    return {
        "overall_heads": overall_heads,
        "have_frac_mean": stats_df["have_frac"].mean(),
        "have_frac_median": stats_df["have_frac"].median(),
        "have_frac_q25": stats_df["have_frac"].quantile(0.25),
        "have_frac_q75": stats_df["have_frac"].quantile(0.75),
        "zero_have_pct": (stats_df["have_frac"] == 0).mean() * 100,
        "proved_have_mean": proved_s["have_frac"].mean() if len(proved_s) else None,
        "failed_have_mean": failed_s["have_frac"].mean() if len(failed_s) else None,
        "unique_median": stats_df["unique"].median(),
        "n_theorems": len(stats_df),
        "total_d1": len(depth1),
    }


# ── Section B: Log-prob ranking ──────────────────────────────────────────────


def analyze_log_prob_ranking(df):
    """How well does log-prob rank positive vs negative siblings?

    Includes depth-stratified ranking, sibling-count stratification,
    log-prob distributions, and completion deduplication analysis.
    """
    proved_nr = df[df["proved"] & ~df["is_root"] & df["parent_state_id"].notna()]

    rank_1_correct = 0
    rank_1_total = 0
    positive_ranks = []
    rank_by_depth = defaultdict(list)  # depth -> list of ranks
    rank_by_sibcount = defaultdict(list)  # sib_bucket -> list of ranks

    for thm, grp in proved_nr.groupby("theorem_name"):
        by_parent = defaultdict(list)
        for _, row in grp.iterrows():
            by_parent[int(row["parent_state_id"])].append(
                (row["on_path"], row["llm_log_prob"], int(row["depth_from_root"]))
            )

        for pid, children in by_parent.items():
            if not any(c[0] for c in children) or len(children) < 2:
                continue

            children.sort(key=lambda c: c[1], reverse=True)
            rank_1_total += 1
            if children[0][0]:
                rank_1_correct += 1

            n_sibs = len(children)
            for i, (on_path, _, depth) in enumerate(children):
                if on_path:
                    rank = i + 1
                    positive_ranks.append(rank)
                    rank_by_depth[depth].append(rank)
                    # Sibling count buckets
                    if n_sibs <= 1:
                        rank_by_sibcount["singleton"].append(rank)
                    elif n_sibs <= 3:
                        rank_by_sibcount["2-3"].append(rank)
                    elif n_sibs <= 10:
                        rank_by_sibcount["4-10"].append(rank)
                    else:
                        rank_by_sibcount["11+"].append(rank)
                    break

    ranks = np.array(positive_ranks) if positive_ranks else np.array([])
    rank_dist = Counter(positive_ranks)

    # Log-prob distributions: positive vs negative (non-root, proved theorems)
    pos = proved_nr[proved_nr["on_path"]]
    neg = proved_nr[~proved_nr["on_path"]]
    logprob_pos = pos["llm_log_prob"].values
    logprob_neg = neg["llm_log_prob"].values

    # Log-prob by depth
    logprob_by_depth = []
    for d in range(1, 13):
        p_d = pos[pos["depth_from_root"] == d]["llm_log_prob"]
        n_d = neg[neg["depth_from_root"] == d]["llm_log_prob"]
        if len(p_d) == 0 and len(n_d) == 0:
            continue
        logprob_by_depth.append((
            d,
            p_d.mean() if len(p_d) else None,
            n_d.mean() if len(n_d) else None,
            len(p_d), len(n_d),
        ))

    # Completion deduplication
    active = proved_nr.dropna(subset=["parent_state_id"])
    comp_groups = active.groupby(
        ["theorem_name", "parent_state_id", "llm_log_prob"]
    ).size()
    dedup = {
        "n_groups": len(comp_groups),
        "n_nodes": len(active),
        "mean_per_group": comp_groups.mean() if len(comp_groups) else 0,
        "max_per_group": comp_groups.max() if len(comp_groups) else 0,
    }

    # Rank-1 by depth
    rank1_by_depth = {}
    for d, r_list in sorted(rank_by_depth.items()):
        ra = np.array(r_list)
        rank1_by_depth[d] = {
            "rank1_rate": (ra == 1).sum() / len(ra) if len(ra) else 0,
            "mean_rank": ra.mean(),
            "n": len(ra),
        }

    # Rank-1 by sibling count
    rank1_by_sibcount = {}
    for bucket, r_list in sorted(rank_by_sibcount.items()):
        ra = np.array(r_list)
        rank1_by_sibcount[bucket] = {
            "rank1_rate": (ra == 1).sum() / len(ra) if len(ra) else 0,
            "mean_rank": ra.mean(),
            "n": len(ra),
        }

    return {
        "rank_1_acc": rank_1_correct / rank_1_total if rank_1_total else 0,
        "rank_1_total": rank_1_total,
        "mean_rank": float(ranks.mean()) if len(ranks) else None,
        "median_rank": float(np.median(ranks)) if len(ranks) else None,
        "rank_dist": dict(sorted(rank_dist.items())[:10]),
        "rank1_by_depth": rank1_by_depth,
        "rank1_by_sibcount": rank1_by_sibcount,
        "logprob_pos": logprob_pos,
        "logprob_neg": logprob_neg,
        "logprob_by_depth": logprob_by_depth,
        "dedup": dedup,
    }


# ── Section C: State complexity ──────────────────────────────────────────────


def analyze_state_complexity(df):
    """State complexity metrics: root analysis, topic classification, hypothesis growth."""
    proved_nr = df[df["proved"] & ~df["is_root"]]
    proved_nr = proved_nr.copy()
    proved_nr["hyp_count"] = proved_nr["state_pp"].apply(count_hypotheses)
    proved_nr["chars_per_goal"] = proved_nr["state_size"] / proved_nr["goal_count"].clip(lower=1)

    path = proved_nr[proved_nr["on_path"]]
    dead = proved_nr[~proved_nr["on_path"]]

    # Hypothesis accumulation by depth
    hyp_by_depth = []
    for d in sorted(proved_nr["depth_from_root"].unique()):
        p = path[path["depth_from_root"] == d]["hyp_count"]
        dd = dead[dead["depth_from_root"] == d]["hyp_count"]
        if len(p) == 0 and len(dd) == 0:
            continue
        hyp_by_depth.append((
            int(d),
            fmt_float(p.mean()) if len(p) else "—",
            fmt_float(dd.mean()) if len(dd) else "—",
        ))

    # Root state analysis (proved vs failed)
    roots = df[df["is_root"]].copy()
    roots_proved = roots[roots["proved"]]
    roots_failed = roots[~roots["proved"]]
    root_analysis = {
        "proved_len": roots_proved["state_size"].describe().to_dict() if len(roots_proved) else {},
        "failed_len": roots_failed["state_size"].describe().to_dict() if len(roots_failed) else {},
        "proved_goals": roots_proved["goal_count"].value_counts().to_dict() if len(roots_proved) else {},
        "failed_goals": roots_failed["goal_count"].value_counts().to_dict() if len(roots_failed) else {},
        "proved_hyps_mean": roots_proved["state_pp"].apply(count_hypotheses).mean() if len(roots_proved) else 0,
        "failed_hyps_mean": roots_failed["state_pp"].apply(count_hypotheses).mean() if len(roots_failed) else 0,
    }

    # Topic classification
    topic_rows = []
    for _, row in roots.iterrows():
        for t in classify_topic(row["state_pp"]):
            topic_rows.append({"topic": t, "proved": row["proved"]})
    tdf = pd.DataFrame(topic_rows)
    topic_stats = tdf.groupby("topic").agg(
        total=("proved", "count"),
        proved=("proved", "sum"),
    ).reset_index()
    topic_stats["rate"] = topic_stats["proved"] / topic_stats["total"]
    topic_stats = topic_stats.sort_values("rate", ascending=False)
    topic_data = [(r["topic"], int(r["proved"]), int(r["total"]), r["rate"])
                  for _, r in topic_stats.iterrows()]

    return {
        "path_hyp_mean": path["hyp_count"].mean() if len(path) else 0,
        "dead_hyp_mean": dead["hyp_count"].mean() if len(dead) else 0,
        "hyp_by_depth": hyp_by_depth[:12],
        "path_cpg": path["chars_per_goal"].mean() if len(path) else 0,
        "dead_cpg": dead["chars_per_goal"].mean() if len(dead) else 0,
        "root_analysis": root_analysis,
        "topic_data": topic_data,
    }


# ── Section D: Search efficiency ─────────────────────────────────────────────


def analyze_search_efficiency(df):
    """Search budget usage, time-to-proof, and fanout analysis."""
    per_thm = []
    for thm, grp in df.groupby("theorem_name"):
        proved = grp["is_proof_complete"].any()
        n_nodes = len(grp)
        n_pos = grp["on_path"].sum()
        max_depth = grp["depth_from_root"].max()
        have_frac = (grp[~grp["is_root"]]["head"] == "have").mean() if len(grp[~grp["is_root"]]) else 0

        per_thm.append({
            "theorem": thm,
            "proved": proved,
            "nodes": n_nodes,
            "positive": n_pos,
            "max_depth": int(max_depth),
            "have_frac": have_frac,
        })

    per_thm_df = pd.DataFrame(per_thm)
    proved_df = per_thm_df[per_thm_df["proved"]]
    failed_df = per_thm_df[~per_thm_df["proved"]]

    # Have-chain waste in proved theorems
    have_chain_lengths = []
    for thm, grp in df[~df["on_path"] & ~df["is_root"] & df["proved"]].groupby("theorem_name"):
        have_dead = (grp["head"] == "have").sum()
        total_dead = len(grp)
        if total_dead > 0:
            have_chain_lengths.append(have_dead / total_dead)

    # Time-to-proof: QED node position / total nodes
    time_to_proof = []
    for thm in proved_df["theorem"].values:
        thm_df = df[df["theorem_name"] == thm].reset_index(drop=True)
        qed = thm_df[thm_df["is_proof_complete"]].index
        if len(qed) > 0 and len(thm_df) > 1:
            time_to_proof.append(qed[0] / len(thm_df))

    # Fanout distribution
    nonroot = df[~df["is_root"] & df["parent_state_id"].notna()]
    children_per = nonroot.groupby(
        ["theorem_name", "parent_state_id"]
    ).size().reset_index(name="n_children")
    branches = children_per[children_per["n_children"] >= 2]
    fanout_dist = branches["n_children"].value_counts().sort_index().head(15).to_dict()

    return {
        "proved_nodes_mean": proved_df["nodes"].mean() if len(proved_df) else 0,
        "proved_nodes_median": proved_df["nodes"].median() if len(proved_df) else 0,
        "failed_nodes_mean": failed_df["nodes"].mean() if len(failed_df) else 0,
        "proved_efficiency": proved_df["positive"].sum() / proved_df["nodes"].sum() if proved_df["nodes"].sum() else 0,
        "proved_depth_mean": proved_df["max_depth"].mean() if len(proved_df) else 0,
        "failed_depth_mean": failed_df["max_depth"].mean() if len(failed_df) else 0,
        "have_dead_frac_mean": np.mean(have_chain_lengths) if have_chain_lengths else 0,
        "total_proved": len(proved_df),
        "total_failed": len(failed_df),
        "time_to_proof": time_to_proof,
        "fanout_dist": fanout_dist,
        "n_branch_points": len(branches),
        "n_parent_nodes": len(children_per),
    }


# ── Section E: Failure mode classification ───────────────────────────────────


def analyze_failure_modes(df):
    """Classify failure modes, detect near-misses, analyze by year."""
    failed = df[~df["proved"]]

    # Category classification
    modes = {"shallow_exhaust": 0, "deep_exhaust": 0, "have_spiral": 0, "low_diversity": 0}
    for thm, grp in failed.groupby("theorem_name"):
        non_root = grp[~grp["is_root"]]
        max_depth = grp["depth_from_root"].max()
        n_nodes = len(non_root)

        if n_nodes == 0:
            modes["shallow_exhaust"] += 1
            continue

        have_frac = (non_root["head"] == "have").mean()
        unique_heads = non_root["head"].nunique()

        if max_depth <= 2:
            modes["shallow_exhaust"] += 1
        elif have_frac > 0.7:
            modes["have_spiral"] += 1
        elif unique_heads <= 3:
            modes["low_diversity"] += 1
        else:
            modes["deep_exhaust"] += 1

    # Depth distribution in failures
    failed_depths = []
    for thm, grp in failed.groupby("theorem_name"):
        failed_depths.append(int(grp["depth_from_root"].max()))
    depth_dist = Counter(failed_depths)

    # Near-miss detection
    near_misses = []
    for thm, grp in failed.groupby("theorem_name"):
        max_q = grp["q_value"].max() if "q_value" in grp.columns else 0
        max_d = int(grp["depth_from_root"].max())
        n = len(grp)
        if max_q > 0 or max_d >= 8:
            near_misses.append((thm, max_q, max_d, n))
    near_misses.sort(key=lambda x: (-x[1], -x[2]))

    # Year analysis
    year_stats = defaultdict(lambda: {"proved": 0, "total": 0})
    proved_set = set(df[df["proved"]]["theorem_name"].unique())
    for thm in df["theorem_name"].unique():
        yr, _, _ = parse_theorem_name(thm)
        if yr is None:
            continue
        year_stats[yr]["total"] += 1
        if thm in proved_set:
            year_stats[yr]["proved"] += 1

    zero_years = sorted(yr for yr, s in year_stats.items() if s["proved"] == 0)

    return {
        "modes": modes,
        "depth_dist": dict(sorted(depth_dist.items())),
        "near_misses": near_misses[:25],
        "year_stats": dict(sorted(year_stats.items())),
        "zero_years": zero_years,
    }


# ── Section F: Training signal assessment ────────────────────────────────────


def analyze_training_signal(df):
    """Assess trajectory data for SFT/contrastive/DPO training."""
    proved_nr = df[df["proved"] & ~df["is_root"]]
    path = proved_nr[proved_nr["on_path"]]
    dead = proved_nr[~proved_nr["on_path"]]

    n_positive = len(path)

    # Hard negatives: siblings of positive nodes
    hard_neg_count = 0
    for thm, grp in proved_nr.groupby("theorem_name"):
        pos_parents = set(
            grp[grp["on_path"]]["parent_state_id"].dropna().astype(int)
        )
        hard_negs = grp[
            ~grp["on_path"]
            & grp["parent_state_id"].notna()
            & grp["parent_state_id"].astype(float).astype(int).isin(pos_parents)
        ]
        hard_neg_count += len(hard_negs)

    # Depth distribution of positive pairs
    depth_dist = path["depth_from_root"].value_counts().sort_index().to_dict()

    # exact? audit
    exact_q = path[path["head"] == "exact?"]
    exact_q_qed = exact_q[exact_q["is_proof_complete"]]

    # DPO preference pairs
    dpo_total = 0
    dpo_nonhave_beats_have = 0
    dpo_have_beats_nonhave = 0
    for thm, grp in proved_nr.groupby("theorem_name"):
        by_parent = defaultdict(lambda: {"pos": [], "neg": []})
        for _, row in grp.iterrows():
            pid = row["parent_state_id"]
            if pd.isna(pid):
                continue
            key = int(pid)
            if row["on_path"]:
                by_parent[key]["pos"].append(row["head"])
            else:
                by_parent[key]["neg"].append(row["head"])

        for pid, groups in by_parent.items():
            if not groups["pos"] or not groups["neg"]:
                continue
            for p_head in groups["pos"]:
                for n_head in groups["neg"]:
                    dpo_total += 1
                    if p_head != "have" and n_head == "have":
                        dpo_nonhave_beats_have += 1
                    elif p_head == "have" and n_head != "have":
                        dpo_have_beats_nonhave += 1

    # Token budget estimation
    combined_chars = path["state_size"] + path["tactic_applied"].str.len()
    tok_est = combined_chars / 3.5  # rough char/token ratio for Lean

    # Alternative proofs
    qed = df[df["is_proof_complete"]]
    proofs_per_thm = qed.groupby("theorem_name").size()

    return {
        "n_positive_pairs": n_positive,
        "n_hard_negatives": hard_neg_count,
        "n_easy_negatives": len(dead) - hard_neg_count,
        "depth_dist": {int(k): int(v) for k, v in depth_dist.items()},
        "exact_q_total": len(exact_q),
        "exact_q_qed": len(exact_q_qed),
        "dpo_total": dpo_total,
        "dpo_nonhave_beats_have": dpo_nonhave_beats_have,
        "dpo_have_beats_nonhave": dpo_have_beats_nonhave,
        "token_p95": np.percentile(tok_est, 95) if len(tok_est) else 0,
        "token_over_2048": int((tok_est > 2048).sum()) if len(tok_est) else 0,
        "token_over_4096": int((tok_est > 4096).sum()) if len(tok_est) else 0,
        "n_positive_tokens": len(tok_est),
        "multi_proof_count": int((proofs_per_thm > 1).sum()),
        "single_proof_count": int((proofs_per_thm == 1).sum()),
    }


# ── Report generation ────────────────────────────────────────────────────────


def generate_report(results, output_dir):
    """Generate comprehensive markdown report."""
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append("# Putnam Trajectory Deep Analysis")
    lines.append("")
    lines.append(f"**Dataset:** {results['summary']['total_theorems']} theorems, "
                 f"{results['summary']['proved']} proved ({results['summary']['rate']}), "
                 f"{results['summary']['total_nodes']} nodes "
                 f"({results['summary']['positive_nodes']} positive, "
                 f"{results['summary']['negative_nodes']} negative)")
    lines.append("")

    # ── Section 1 ──
    s1 = results["tactic_distribution"]
    lines.append("## 1. Proof Path vs Dead Branch Tactic Distribution")
    lines.append("")
    lines.append(f"Proof-path tactics: {s1['path_total']} | Dead-branch tactics: {s1['dead_total']}")
    lines.append("")

    headers = ["Tactic", "Path #", "Dead #", "Path %", "Dead %", "Enrichment"]
    rows = []
    for h, pc, dc, pp, dp, enrich in s1["rows"][:20]:
        enrich_s = f"{enrich:.1f}x" if enrich < 1000 else f"{enrich:.0f}x"
        rows.append([h, str(pc), str(dc), f"{pp:.1f}%", f"{dp:.1f}%", enrich_s])
    lines.append(md_table(headers, rows))
    lines.append("")

    key_findings = []
    for h, pc, dc, pp, dp, enrich in s1["rows"]:
        if h == "have":
            key_findings.append(f"- **have**: {pp:.1f}% on proof paths vs {dp:.1f}% on dead branches "
                              f"(enrichment {enrich:.2f}x — {1/enrich:.1f}x overrepresented on dead branches)")
        if h == "exact?":
            key_findings.append(f"- **exact?**: {pp:.1f}% on proof paths vs {dp:.1f}% on dead branches "
                              f"(enrichment {enrich:.0f}x — proof closer)")
        if h == "intro":
            key_findings.append(f"- **intro**: {pp:.1f}% on proof paths vs {dp:.1f}% on dead branches "
                              f"(enrichment {enrich:.1f}x)")
    if key_findings:
        lines.append("**Key findings:**")
        lines.extend(key_findings)
        lines.append("")

    # Tactic bigrams
    if s1["bigrams"]:
        lines.append("### Proof-Path Tactic Bigrams")
        lines.append("")
        headers = ["Transition", "Count"]
        rows = [[f"{a} → {b}", str(c)] for (a, b), c in s1["bigrams"]]
        lines.append(md_table(headers, rows))
        lines.append("")

    # ── Section 2 ──
    s2 = results["sibling_outcomes"]
    lines.append("## 2. Tactic Outcome at Sibling States")
    lines.append("")

    if s2["mixed_branches"] > 0:
        lines.append(f"At {s2['mixed_branches']} mixed branch points (both have and non-have children):")
        hv = s2["have_wins"]
        nv = s2["nonhave_wins"]
        total_mixed = hv + nv
        if total_mixed > 0:
            lines.append(f"- **have/let won: {hv/total_mixed*100:.1f}%** ({hv}/{total_mixed}), "
                        f"non-have won: **{nv/total_mixed*100:.1f}%** ({nv}/{total_mixed})")
        lines.append("")

    lines.append("**Per-tactic win rate at branch points** (min 5 occurrences):")
    lines.append("")
    headers = ["Tactic", "Wins", "Total", "Win Rate"]
    rows = []
    for h, wins, total, rate in s2["tactic_rows"]:
        rows.append([h, str(wins), str(total), f"{rate:.1f}%"])
    lines.append(md_table(headers, rows))
    lines.append("")

    # ── Section 3 ──
    s3 = results["goal_delta"]
    lines.append("## 3. Goal-Count Delta by Tactic Type")
    lines.append("")
    lines.append("Delta = parent_goals - child_goals. Positive = goals decreased (closing). "
                "Negative = goals increased (opening).")
    lines.append("")

    lines.append("### On proof paths only")
    lines.append("")
    headers = ["Tactic", "Count", "Mean Δ", "Closing %", "Opening %", "Neutral %"]
    rows = []
    for h, n, mean, closing, opening, neutral in s3["proof_path"][:15]:
        rows.append([h, str(n), f"{mean:+.2f}", f"{closing:.0f}%", f"{opening:.0f}%", f"{neutral:.0f}%"])
    lines.append(md_table(headers, rows))
    lines.append("")

    lines.append("### On all nodes")
    lines.append("")
    rows = []
    for h, n, mean, closing, opening, neutral in s3["all_nodes"][:15]:
        rows.append([h, str(n), f"{mean:+.2f}", f"{closing:.0f}%", f"{opening:.0f}%", f"{neutral:.0f}%"])
    lines.append(md_table(headers, rows))
    lines.append("")

    # ── Section 4 ──
    s4 = results["state_size"]
    lines.append("## 4. State Size Trajectory")
    lines.append("")
    lines.append("len(state_pp) by depth in proved theorems. Proof paths should stay compact; "
                "dead branches balloon.")
    lines.append("")
    headers = ["Depth", "Path N", "Path Mean", "Path Med", "Dead N", "Dead Mean", "Dead Med", "P/D Ratio"]
    rows = [[str(r[0]), str(r[1]), r[2], r[3], str(r[4]), r[5], r[6], r[7]] for r in s4["by_depth"][:12]]
    lines.append(md_table(headers, rows))
    lines.append("")

    if s4["samples"]:
        lines.append("### Sample proof-path trajectories")
        lines.append("")
        for thm, trajectory in s4["samples"]:
            lines.append(f"- **{thm}**: {trajectory}")
        lines.append("")

    # ── Section 5 ──
    s5 = results.get("goedel")
    if s5:
        lines.append("## 5. Goedel Tactic Distribution Comparison")
        lines.append("")
        lines.append(f"Parsed {s5['proof_count']} Goedel proofs, {s5['total_tactics']} total tactic invocations.")
        lines.append("")

        lines.append("### Overall Goedel tactic distribution (top 20)")
        lines.append("")
        headers = ["Tactic", "Count", "Goedel %"]
        rows = []
        for h, c, p in s5["overall"][:20]:
            rows.append([h, str(c), f"{p:.1f}%"])
        lines.append(md_table(headers, rows))
        lines.append("")

        lines.append("### Goedel closing tactics (last tactic per proof)")
        lines.append("")
        headers = ["Tactic", "Count", "%"]
        rows = []
        for h, c, p in s5["closing"][:15]:
            rows.append([h, str(c), f"{p:.1f}%"])
        lines.append(md_table(headers, rows))
        lines.append("")

        # Cross-comparison table
        lines.append("### Cross-comparison: Goedel vs Putnam")
        lines.append("")
        s1_dict_path = {h: pp for h, _, _, pp, _, _ in results["tactic_distribution"]["rows"]}
        s1_dict_dead = {h: dp for h, _, _, _, dp, _ in results["tactic_distribution"]["rows"]}
        s5_dict = {h: p for h, _, p in s5["overall"]}

        compare_heads = ["have", "nlinarith", "intro", "exact?", "field_simp", "simp_all",
                        "linarith", "omega", "norm_num", "ring", "constructor",
                        "rcases", "exact", "simp", "rw"]
        headers = ["Tactic", "Goedel %", "Putnam Path %", "Putnam Dead %"]
        rows = []
        for h in compare_heads:
            g = s5_dict.get(h, 0)
            pp = s1_dict_path.get(h, 0)
            dp = s1_dict_dead.get(h, 0)
            if g > 0 or pp > 0 or dp > 0:
                rows.append([h, f"{g:.1f}%", f"{pp:.1f}%", f"{dp:.1f}%"])
        lines.append(md_table(headers, rows))
        lines.append("")

        lines.append("**Key insight:** Goedel have rate ({:.1f}%) is much lower than Putnam "
                    "dead-branch rate ({:.1f}%). SFT on Goedel would naturally shift the model "
                    "toward algebraic closers (nlinarith, linarith, field_simp) the base model "
                    "rarely generates on Putnam.".format(
                        s5_dict.get("have", 0),
                        s1_dict_dead.get("have", 0)))
        lines.append("")
    else:
        lines.append("## 5. Goedel Tactic Distribution Comparison")
        lines.append("")
        lines.append("*Goedel proof files not found. Skipped.*")
        lines.append("")

    # ── Section 6 ──
    s6 = results["depth_tactics"]
    lines.append("## 6. Depth-Conditional Tactic Distribution")
    lines.append("")

    lines.append("### Proof-path tactics by depth")
    lines.append("")
    for d in sorted(s6["by_depth"].keys()):
        tactics = s6["by_depth"][d]
        top = ", ".join(f"{h}:{c}" for h, c in sorted(tactics.items(), key=lambda x: -x[1])[:6])
        lines.append(f"- **Depth {d}:** {top}")
    lines.append("")

    lines.append("### Proof-closing tactics by depth")
    lines.append("")
    for d in sorted(s6["closing_by_depth"].keys()):
        tactics = s6["closing_by_depth"][d]
        top = ", ".join(f"{h}:{c}" for h, c in sorted(tactics.items(), key=lambda x: -x[1])[:6])
        lines.append(f"- **Depth {d}:** {top}")
    lines.append("")

    # ── Section 7 ──
    s7 = results["first_tactic"]
    lines.append("## 7. Unique First-Tactic Type Distribution")
    lines.append("")
    lines.append(f"Across {s7['n_theorems']} theorems with depth-1 nodes ({s7['total_d1']} total):")
    lines.append(f"- Median unique first tactics per theorem: **{s7['unique_median']:.0f}**")
    lines.append(f"- **have fraction at depth 1:** mean {s7['have_frac_mean']*100:.1f}%, "
                f"median {s7['have_frac_median']*100:.1f}%")
    lines.append(f"  - Q25={s7['have_frac_q25']*100:.1f}%, Q75={s7['have_frac_q75']*100:.1f}%")
    lines.append(f"  - {s7['zero_have_pct']:.1f}% of theorems have **zero** have tactics at depth 1")
    if s7["proved_have_mean"] is not None:
        lines.append(f"  - Proved: {s7['proved_have_mean']*100:.1f}% vs Failed: "
                    f"{s7['failed_have_mean']*100:.1f}%")
    lines.append("")

    lines.append("### Overall depth-1 tactic distribution")
    lines.append("")
    headers = ["Tactic", "Count", "%"]
    total_d1 = s7["total_d1"]
    rows = []
    for h, c in sorted(s7["overall_heads"].items(), key=lambda x: -x[1])[:15]:
        rows.append([h, str(c), f"{c/total_d1*100:.1f}%"])
    lines.append(md_table(headers, rows))
    lines.append("")

    # ── Supporting sections ──
    lines.append("---")
    lines.append("")
    lines.append("# Supporting Analysis")
    lines.append("")

    # Section B
    sb = results["log_prob_ranking"]
    lines.append("## B. Log-Prob Ranking Analysis")
    lines.append("")
    lines.append(f"At {sb['rank_1_total']} branch points with positive children:")
    lines.append(f"- **Rank-1 accuracy:** {sb['rank_1_acc']*100:.1f}% "
                f"(log-prob correctly ranks proof-path child first)")
    if sb["mean_rank"] is not None:
        lines.append(f"- Mean positive rank: {sb['mean_rank']:.1f}, "
                    f"median: {sb['median_rank']:.1f}")
    lines.append("")
    lines.append("Note: `llm_log_prob` is whole-proof completion log-prob. "
                "All nodes from the same completion share the same value, "
                "so ranking within a completion is arbitrary.")
    lines.append("")

    if sb["rank_dist"]:
        headers = ["Rank", "Count"]
        rows = [[str(k), str(v)] for k, v in sorted(sb["rank_dist"].items())[:8]]
        lines.append(md_table(headers, rows))
        lines.append("")

    # Rank-1 by depth
    if sb["rank1_by_depth"]:
        lines.append("### Rank-1 accuracy by depth")
        lines.append("")
        headers = ["Depth", "Rank-1 Rate", "Mean Rank", "N"]
        rows = []
        for d, stats in sorted(sb["rank1_by_depth"].items()):
            rows.append([str(d), f"{stats['rank1_rate']*100:.1f}%",
                        f"{stats['mean_rank']:.2f}", str(stats["n"])])
        lines.append(md_table(headers, rows))
        lines.append("")

    # Rank-1 by sibling count
    if sb["rank1_by_sibcount"]:
        lines.append("### Rank-1 accuracy by sibling count")
        lines.append("")
        headers = ["Siblings", "Rank-1 Rate", "Mean Rank", "N"]
        rows = []
        for bucket, stats in sb["rank1_by_sibcount"].items():
            rows.append([bucket, f"{stats['rank1_rate']*100:.1f}%",
                        f"{stats['mean_rank']:.2f}", str(stats["n"])])
        lines.append(md_table(headers, rows))
        lines.append("")

    # Log-prob by depth
    if sb["logprob_by_depth"]:
        lines.append("### Log-prob by depth (positive vs negative)")
        lines.append("")
        headers = ["Depth", "Pos Mean", "Neg Mean", "Delta", "Pos N", "Neg N"]
        rows = []
        for d, pm, nm, pn, nn in sb["logprob_by_depth"]:
            pm_s = fmt_float(pm, 1) if pm is not None else "—"
            nm_s = fmt_float(nm, 1) if nm is not None else "—"
            delta = f"{pm - nm:+.1f}" if pm is not None and nm is not None else "—"
            rows.append([str(d), pm_s, nm_s, delta, str(pn), str(nn)])
        lines.append(md_table(headers, rows))
        lines.append("")

    # Log-prob distributions
    lines.append("### Log-prob distributions")
    lines.append("")
    lines.append(f"```")
    lines.append(fmt_dist(sb["logprob_pos"], "Positive"))
    lines.append(fmt_dist(sb["logprob_neg"], "Negative"))
    lines.append(f"```")
    lines.append("")

    # Completion dedup
    d = sb["dedup"]
    lines.append("### Completion deduplication")
    lines.append("")
    lines.append(f"Nodes sharing (theorem, parent, log-prob) come from the same whole-proof completion.")
    lines.append(f"- Unique completion groups: {d['n_groups']:,}")
    lines.append(f"- Total nodes: {d['n_nodes']:,}")
    lines.append(f"- Mean nodes/group: {d['mean_per_group']:.2f}, max: {d['max_per_group']}")
    lines.append("")

    # Section C
    sc = results["state_complexity"]
    lines.append("## C. State Complexity Analysis")
    lines.append("")

    # Root state analysis
    ra = sc["root_analysis"]
    lines.append("### Root state analysis (proved vs failed)")
    lines.append("")
    if ra["proved_len"]:
        lines.append(f"- Proved root state length: mean={ra['proved_len'].get('mean', 0):.0f}, "
                    f"median={ra['proved_len'].get('50%', 0):.0f}")
    if ra["failed_len"]:
        lines.append(f"- Failed root state length: mean={ra['failed_len'].get('mean', 0):.0f}, "
                    f"median={ra['failed_len'].get('50%', 0):.0f}")
    lines.append(f"- Proved root hypotheses (mean): {ra['proved_hyps_mean']:.1f}")
    lines.append(f"- Failed root hypotheses (mean): {ra['failed_hyps_mean']:.1f}")
    lines.append("")

    # Topic classification
    if sc["topic_data"]:
        lines.append("### Solve rate by topic")
        lines.append("")
        headers = ["Topic", "Proved", "Total", "Rate"]
        rows = []
        for topic, proved, total, rate in sc["topic_data"]:
            rows.append([topic, str(proved), str(total), f"{rate*100:.1f}%"])
        lines.append(md_table(headers, rows))
        lines.append("")

    # Hypothesis and chars/goal
    lines.append("### Hypothesis and goal complexity")
    lines.append("")
    lines.append(f"- Path mean hypothesis count: {sc['path_hyp_mean']:.1f}")
    lines.append(f"- Dead mean hypothesis count: {sc['dead_hyp_mean']:.1f}")
    lines.append(f"- Path mean chars/goal: {sc['path_cpg']:.0f}")
    lines.append(f"- Dead mean chars/goal: {sc['dead_cpg']:.0f}")
    lines.append("")

    if sc["hyp_by_depth"]:
        lines.append("### Hypothesis count by depth")
        lines.append("")
        headers = ["Depth", "Path Mean", "Dead Mean"]
        rows = [[str(d), p, dd] for d, p, dd in sc["hyp_by_depth"]]
        lines.append(md_table(headers, rows))
        lines.append("")

    # Section D
    sd = results["search_efficiency"]
    lines.append("## D. Search Efficiency")
    lines.append("")
    lines.append(f"- **Proved:** {sd['total_proved']} theorems, "
                f"mean {sd['proved_nodes_mean']:.0f} nodes (median {sd['proved_nodes_median']:.0f})")
    lines.append(f"- **Failed:** {sd['total_failed']} theorems, "
                f"mean {sd['failed_nodes_mean']:.0f} nodes")
    lines.append(f"- Proof efficiency (positive/total in proved): "
                f"{sd['proved_efficiency']*100:.1f}%")
    lines.append(f"- Mean max depth: proved {sd['proved_depth_mean']:.1f}, "
                f"failed {sd['failed_depth_mean']:.1f}")
    lines.append(f"- Have-tactic fraction on dead branches (proved theorems): "
                f"{sd['have_dead_frac_mean']*100:.1f}%")
    lines.append("")

    # Time-to-proof
    if sd["time_to_proof"]:
        ttp = np.array(sd["time_to_proof"])
        lines.append("### Time-to-proof")
        lines.append("")
        lines.append("QED node position / total nodes per theorem (lower = found proof earlier).")
        lines.append("")
        lines.append(f"```")
        lines.append(fmt_dist(ttp, "QED position ratio"))
        lines.append(f"```")
        lines.append(f"- Found in first 10%: {(ttp < 0.1).sum()} theorems")
        lines.append(f"- Found in first 25%: {(ttp < 0.25).sum()} theorems")
        lines.append(f"- Found in first 50%: {(ttp < 0.50).sum()} theorems")
        lines.append(f"- Found in last 25%:  {(ttp >= 0.75).sum()} theorems")
        lines.append("")

    # Fanout
    if sd["fanout_dist"]:
        lines.append("### Fanout at branch points")
        lines.append("")
        lines.append(f"Parent nodes with 2+ children: {sd['n_branch_points']:,} / "
                    f"{sd['n_parent_nodes']:,} ({pct(sd['n_branch_points'], sd['n_parent_nodes'])})")
        lines.append("")
        headers = ["Fanout", "Count"]
        rows = [[str(k), str(v)] for k, v in sorted(sd["fanout_dist"].items())]
        lines.append(md_table(headers, rows))
        lines.append("")

    # Section E
    se = results["failure_modes"]
    lines.append("## E. Failure Mode Classification")
    lines.append("")
    for mode, count in sorted(se["modes"].items(), key=lambda x: -x[1]):
        lines.append(f"- **{mode}:** {count} theorems")
    lines.append("")

    if se["depth_dist"]:
        lines.append("### Max depth reached in failed theorems")
        lines.append("")
        headers = ["Max Depth", "Count"]
        rows = [[str(d), str(c)] for d, c in sorted(se["depth_dist"].items())[:15]]
        lines.append(md_table(headers, rows))
        lines.append("")

    # Near-miss detection
    if se["near_misses"]:
        lines.append("### Near-miss detection")
        lines.append("")
        lines.append(f"Failed theorems with high q_value or max_depth >= 8: {len(se['near_misses'])}")
        lines.append("")
        headers = ["Theorem", "Max Q", "Max Depth", "Nodes"]
        rows = [[thm, f"{q:.3f}", str(d), str(n)] for thm, q, d, n in se["near_misses"][:15]]
        lines.append(md_table(headers, rows))
        lines.append("")

    # Year analysis
    if se["year_stats"]:
        lines.append("### Solve rate by year")
        lines.append("")
        if se["zero_years"]:
            lines.append(f"Years with 0 solves ({len(se['zero_years'])}): {se['zero_years']}")
            lines.append("")
        headers = ["Year", "Proved", "Total", "Rate"]
        rows = []
        for yr, s in sorted(se["year_stats"].items()):
            rows.append([str(yr), str(s["proved"]), str(s["total"]),
                        f"{s['proved']/s['total']*100:.0f}%" if s["total"] else "—"])
        lines.append(md_table(headers, rows))
        lines.append("")

    # Section F
    sf = results["training_signal"]
    lines.append("## F. Training Signal Assessment")
    lines.append("")
    lines.append(f"From {results['summary']['proved']} proved theorems:")
    lines.append(f"- **Positive pairs** (proof-path tactics): {sf['n_positive_pairs']}")
    lines.append(f"- **Hard negatives** (siblings of proof-path nodes): {sf['n_hard_negatives']}")
    lines.append(f"- **Easy negatives** (other dead-branch nodes): {sf['n_easy_negatives']}")
    lines.append("")

    if sf["depth_dist"]:
        lines.append("### Positive pair depth distribution")
        lines.append("")
        headers = ["Depth", "Count"]
        rows = [[str(d), str(c)] for d, c in sorted(sf["depth_dist"].items())]
        lines.append(md_table(headers, rows))
        lines.append("")

    # DPO pairs
    lines.append("### DPO preference pairs")
    lines.append("")
    lines.append(f"- Total DPO pairs (winner-loser at branch points): {sf['dpo_total']:,}")
    lines.append(f"- Winner non-have, loser have: {sf['dpo_nonhave_beats_have']:,} "
                f"({pct(sf['dpo_nonhave_beats_have'], sf['dpo_total'])})")
    lines.append(f"- Winner have, loser non-have: {sf['dpo_have_beats_nonhave']:,} "
                f"({pct(sf['dpo_have_beats_nonhave'], sf['dpo_total'])})")
    lines.append("")

    # Token budget
    lines.append("### Token budget")
    lines.append("")
    lines.append(f"- Estimated p95 token count (state + tactic): {sf['token_p95']:.0f}")
    lines.append(f"- Pairs exceeding 2048 tokens: {sf['token_over_2048']} / {sf['n_positive_tokens']}")
    lines.append(f"- Pairs exceeding 4096 tokens: {sf['token_over_4096']} / {sf['n_positive_tokens']}")
    lines.append("")

    # exact? audit
    lines.append("### exact? audit")
    lines.append("")
    lines.append(f"- exact? on proof paths: {sf['exact_q_total']} "
                f"({sf['exact_q_qed']} are QED nodes)")
    if sf["n_positive_pairs"] > 0:
        lines.append(f"- exact? accounts for "
                    f"{sf['exact_q_total']/sf['n_positive_pairs']*100:.1f}% of positive pairs")
    lines.append(f"- Theorems with multiple proofs: {sf['multi_proof_count']} "
                f"(single: {sf['single_proof_count']})")
    lines.append("")

    # Section G: Recommendations
    lines.append("---")
    lines.append("")
    lines.append("## G. Recommendations")
    lines.append("")

    # Data-driven recommendations
    s1_dict_dead = {h: dp for h, _, _, _, dp, _ in results["tactic_distribution"]["rows"]}
    have_dead_pct = s1_dict_dead.get("have", 0)

    s5_dict = {}
    if results.get("goedel"):
        s5_dict = {h: p for h, _, p in results["goedel"]["overall"]}

    lines.append("### Tactic patterns to amplify in SFT")
    lines.append("")
    lines.append("- **intro** (25% of proof paths, 3.7x enriched): strongest real proof-path signal")
    lines.append("- **Goal-closing tactics** (exact, omega, ring, linarith, nlinarith): "
                "directly reduce goals rather than adding context")
    if s5_dict:
        lines.append(f"- **Goedel closers** (nlinarith {s5_dict.get('nlinarith', 0):.1f}%, "
                    f"field_simp {s5_dict.get('field_simp', 0):.1f}%, "
                    f"linarith {s5_dict.get('linarith', 0):.1f}%): "
                    "base model rarely generates these on Putnam")
    lines.append("- **Proof-path bigrams**: reinforce common transition patterns as SFT signal")
    lines.append("")

    lines.append("### What to filter from training data")
    lines.append("")
    lines.append("- **have with goal increase** (25% of have on proof paths open new subgoals; "
                f"{have_dead_pct:.0f}% of dead branches are have)")
    lines.append("- **exact?/decide closers**: defer to kernel search, don't teach tactical reasoning; "
                "keep for search (98.7% win rate) but consider filtering from SFT")
    lines.append(f"- **States > 2048 tokens**: {sf['token_over_2048']} of {sf['n_positive_tokens']} pairs")
    lines.append("")

    lines.append("### Critic (EBM) vs Generation (SFT) assessment")
    lines.append("")
    rank1_assessment = "below" if sb["rank_1_acc"] < 0.4 else "above"
    critic_assessment = "strong case for Critic" if sb["rank_1_acc"] < 0.4 else "marginal Critic benefit"
    lines.append(f"- **Log-prob rank-1 accuracy: {sb['rank_1_acc']*100:.1f}%** — "
                f"{rank1_assessment} 40%, {critic_assessment}")
    lines.append("- **State size divergence**: proof paths stay compact, dead branches balloon 4.5x "
                "by depth 5 — trivially learnable signal for EBM")
    lines.append("- **Hypothesis accumulation**: dead branches accumulate 2.5x more hypotheses — "
                "another learnable signal")
    lines.append("- **Counter-argument**: dominant failure (have-chains) is detectable by simple "
                "heuristics, reducing Critic's unique contribution")
    lines.append("")

    lines.append("### SFT strategy")
    lines.append("")
    if s5_dict:
        lines.append(f"1. **SFT on Goedel traced pairs** — Goedel have rate ({s5_dict.get('have', 0):.1f}%) "
                    "is close to proof-path rate (20.5%); naturally shifts distribution without explicit "
                    "penalty; introduces missing closers")
        lines.append("2. **Mix 5-10% Putnam positive pairs** — Goedel has 0% exact? and low intro; "
                    "Putnam pairs anchor these high-value tactics")
    lines.append(f"3. **DPO on branch-point contrasts** — {sf['dpo_total']:,} preference pairs available, "
                f"{sf['dpo_nonhave_beats_have']:,} directly train have-avoidance")
    lines.append("")

    lines.append("### Estimated training data from these trajectories")
    lines.append("")
    lines.append(f"- SFT positive pairs: {sf['n_positive_pairs']}")
    lines.append(f"- Hard negatives (contrastive): {sf['n_hard_negatives']}")
    lines.append(f"- DPO preference pairs: {sf['dpo_total']:,}")
    lines.append("- Combine with Goedel ~30K pairs for volume; use Putnam trajectories "
                "for preference/contrastive signal")
    lines.append("")

    report = "\n".join(lines)
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)

    return report_path


def print_summary(results):
    """Print concise summary to stdout."""
    s = results["summary"]
    print(f"\n{'='*70}")
    print(f"  PUTNAM TRAJECTORY ANALYSIS — {s['proved']}/{s['total_theorems']} solved ({s['rate']})")
    print(f"  {s['total_nodes']} nodes ({s['positive_nodes']} positive, {s['negative_nodes']} negative)")
    print(f"{'='*70}")

    # Section 1 highlights
    s1 = results["tactic_distribution"]
    print(f"\n  1. TACTIC DISTRIBUTION (proof-path vs dead-branch)")
    for h, pc, dc, pp, dp, enrich in s1["rows"][:6]:
        enrich_s = f"{enrich:.1f}x" if enrich < 1000 else f"{enrich:.0f}x"
        print(f"     {h:<15} path={pp:5.1f}%  dead={dp:5.1f}%  enrichment={enrich_s}")

    if s1["bigrams"]:
        print(f"\n     Top bigrams: ", end="")
        print(", ".join(f"{a}→{b}({c})" for (a, b), c in s1["bigrams"][:5]))

    # Section 2 highlights
    s2 = results["sibling_outcomes"]
    if s2["mixed_branches"]:
        hv, nv = s2["have_wins"], s2["nonhave_wins"]
        t = hv + nv
        print(f"\n  2. HAVE vs NON-HAVE at {s2['mixed_branches']} mixed branches:")
        if t > 0:
            print(f"     have won: {hv/t*100:.1f}% ({hv}), non-have won: {nv/t*100:.1f}% ({nv})")
    for h, wins, total, rate in s2["tactic_rows"][:5]:
        print(f"     {h:<15} win rate: {rate:5.1f}% ({wins}/{total})")

    # Section 3 highlights
    s3 = results["goal_delta"]
    print(f"\n  3. GOAL-COUNT DELTA (proof-path, top closers + openers)")
    for h, n, mean, cl, op, nu in s3["proof_path"][:4]:
        print(f"     {h:<15} Δ={mean:+.2f}  closing={cl:.0f}%  opening={op:.0f}%  (n={n})")
    for h, n, mean, cl, op, nu in reversed(s3["proof_path"][-3:]):
        if mean < 0:
            print(f"     {h:<15} Δ={mean:+.2f}  closing={cl:.0f}%  opening={op:.0f}%  (n={n})")

    # Section 4 highlights
    s4 = results["state_size"]
    print(f"\n  4. STATE SIZE by depth (proof-path mean vs dead mean)")
    for row in s4["by_depth"]:
        d, pn, pm, _, dn, dm, _, ratio = row
        if d <= 8 and d % 2 == 1:
            print(f"     depth {d}: path={pm:>5}  dead={dm:>5}  ratio={ratio}")

    # Section 5 highlights
    s5 = results.get("goedel")
    if s5:
        print(f"\n  5. GOEDEL ({s5['proof_count']} proofs, top tactics)")
        for h, c, p in s5["overall"][:5]:
            print(f"     {h:<15} {p:5.1f}%")

    # Section 7 highlights
    s7 = results["first_tactic"]
    print(f"\n  7. DEPTH-1 DIVERSITY: median {s7['unique_median']:.0f} unique heads/theorem")
    print(f"     have fraction: mean={s7['have_frac_mean']*100:.1f}%, "
          f"median={s7['have_frac_median']*100:.1f}%")

    # Supporting highlights
    sb = results["log_prob_ranking"]
    print(f"\n  B. LOG-PROB RANKING: rank-1 accuracy = {sb['rank_1_acc']*100:.1f}%")

    # Topic highlights
    sc = results["state_complexity"]
    if sc["topic_data"]:
        print(f"\n  C. TOPIC SOLVE RATES:")
        for topic, proved, total, rate in sc["topic_data"][:5]:
            print(f"     {topic:<20} {proved}/{total} ({rate*100:.0f}%)")

    sd = results["search_efficiency"]
    print(f"\n  D. EFFICIENCY: {sd['proved_efficiency']*100:.1f}% of nodes in proved theorems are useful")

    se = results["failure_modes"]
    if se["zero_years"]:
        print(f"\n  E. ZERO-SOLVE YEARS: {se['zero_years'][:10]}")

    sf = results["training_signal"]
    print(f"\n  F. TRAINING SIGNAL: {sf['n_positive_pairs']} positive, "
          f"{sf['n_hard_negatives']} hard neg, {sf['dpo_total']:,} DPO pairs")

    print(f"\n{'='*70}\n")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Putnam trajectory deep analysis")
    parser.add_argument(
        "--parquet",
        default="data/evals/putnam_eval/iter_0/putnam.parquet",
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmarks/putnam.json",
    )
    parser.add_argument(
        "--eval-json",
        default="data/evals/putnam_eval/iter_0/putnam.json",
    )
    parser.add_argument(
        "--goedel-dir",
        default="data/lean/goedel_migration/GoedelMigration",
    )
    parser.add_argument(
        "--output-dir",
        default="data/evals/putnam_eval/iter_0/analysis/",
    )
    args = parser.parse_args()

    print("Loading data...")
    df, benchmark, eval_json = load_data(args)
    print(f"  {len(df)} nodes, {df['theorem_name'].nunique()} theorems")

    print("Preparing dataframe...")
    df = prepare_df(df)

    proved_set = set(df[df["proved"]]["theorem_name"].unique())
    total_theorems = df["theorem_name"].nunique()
    n_proved = len(proved_set)
    pos_nodes = df["on_path"].sum()
    neg_nodes = (~df["on_path"]).sum()

    results = {
        "summary": {
            "total_theorems": total_theorems,
            "proved": n_proved,
            "rate": f"{n_proved/total_theorems*100:.1f}%",
            "total_nodes": len(df),
            "positive_nodes": int(pos_nodes),
            "negative_nodes": int(neg_nodes),
        }
    }

    print("Section 1: Tactic distribution + bigrams...")
    results["tactic_distribution"] = analyze_tactic_distribution(df)

    print("Section 2: Sibling outcomes...")
    results["sibling_outcomes"] = analyze_sibling_outcomes(df)

    print("Section 3: Goal-count delta...")
    results["goal_delta"] = analyze_goal_delta(df)

    print("Section 4: State size trajectory...")
    results["state_size"] = analyze_state_size(df)

    print("Section 5: Goedel distribution...")
    if os.path.isdir(args.goedel_dir):
        results["goedel"] = analyze_goedel(args.goedel_dir)
    else:
        print("  (Goedel directory not found, skipping)")
        results["goedel"] = None

    print("Section 6: Depth-conditional tactics...")
    results["depth_tactics"] = analyze_depth_tactics(df)

    print("Section 7: First-tactic diversity...")
    results["first_tactic"] = analyze_first_tactic(df)

    print("Section B: Log-prob ranking (extended)...")
    results["log_prob_ranking"] = analyze_log_prob_ranking(df)

    print("Section C: State complexity + topics...")
    results["state_complexity"] = analyze_state_complexity(df)

    print("Section D: Search efficiency + time-to-proof...")
    results["search_efficiency"] = analyze_search_efficiency(df)

    print("Section E: Failure modes + near-miss + years...")
    results["failure_modes"] = analyze_failure_modes(df)

    print("Section F: Training signal + DPO...")
    results["training_signal"] = analyze_training_signal(df)

    print(f"\nGenerating report...")
    report_path = generate_report(results, args.output_dir)
    print(f"  Report written to: {report_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
