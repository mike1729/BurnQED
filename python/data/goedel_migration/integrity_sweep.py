#!/usr/bin/env python3
"""M.6 Pre-tracing integrity sweep on raw .lean proof files.

Scans the 28K+ passing Goedel proofs for sorry/admit/cheat/sorryAx
contamination *in tactic blocks only* (not in comments or docstrings).
Also flags trivial single-tactic proofs for informational purposes.

Usage:
    python python/data/goedel_migration/integrity_sweep.py

Inputs:
    data/logs/goedel_compile/compile_results.json
    data/lean/goedel_migration/GoedelMigration/Proof_*.lean
    data/lean/goedel_migration/goedel_manifest.json

Output:
    data/traced/integrity_report.json
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMPILE_RESULTS = ROOT / "data/logs/goedel_compile/compile_results.json"
PROOF_DIR = ROOT / "data/lean/goedel_migration/GoedelMigration"
MANIFEST = ROOT / "data/lean/goedel_migration/goedel_manifest.json"
REPORT_OUT = ROOT / "data/traced/integrity_report.json"

BANNED_KEYWORDS = {"sorry", "admit", "cheat", "sorryAx"}
# Match banned keywords as whole words (not substrings like "admits")
BANNED_RE = re.compile(r"\b(?:sorry|admit|cheat|sorryAx)\b")
SUSPICIOUS_RE = re.compile(r"\bDecidable\.decide\b")
TRIVIAL_TACTICS = {"trivial", "rfl", "decide"}


def strip_comments(source: str) -> list[str]:
    """Remove block comments (/- ... -/) and line comments (-- ...) from source.

    Returns a list of lines with comments replaced by empty strings,
    preserving line numbering.
    """
    # First pass: remove block comments (can be nested in Lean 4)
    result = []
    i = 0
    depth = 0
    cleaned = []
    while i < len(source):
        if source[i:i+2] == "/-":
            depth += 1
            i += 2
        elif source[i:i+2] == "-/" and depth > 0:
            depth -= 1
            i += 2
        elif depth > 0:
            # Inside block comment — replace with space to preserve structure
            if source[i] == "\n":
                cleaned.append("\n")
            i += 1
        else:
            cleaned.append(source[i])
            i += 1

    text = "".join(cleaned)

    # Second pass: remove line comments (-- to end of line)
    lines = text.split("\n")
    for line in lines:
        # Find -- that's not inside a string (simple heuristic: just strip from --)
        idx = line.find("--")
        if idx >= 0:
            result.append(line[:idx])
        else:
            result.append(line)

    return result


def extract_tactic_lines(source: str) -> list[str]:
    """Extract lines that are inside `by` blocks (tactic mode).

    Returns only the tactic content lines with comments stripped.
    """
    clean_lines = strip_comments(source)
    tactic_lines = []
    in_by_block = False

    for line in clean_lines:
        stripped = line.strip()
        # Detect `:= by` — everything after is tactic mode
        if ":= by" in line or "where" in line and in_by_block:
            # The part after `:= by` on the same line could have tactics
            idx = line.find(":= by")
            if idx >= 0:
                after = line[idx + 5:].strip()
                if after:
                    tactic_lines.append(after)
                in_by_block = True
                continue

        if in_by_block:
            if stripped:
                tactic_lines.append(stripped)

    return tactic_lines


def check_file(filepath: Path) -> dict:
    """Check a single .lean file for contamination and triviality.

    Returns dict with:
        contaminated: bool
        contamination_reasons: list[str]
        suspicious: list[str]
        trivial: bool
        tactic_count: int
    """
    source = filepath.read_text(encoding="utf-8")
    tactic_lines = extract_tactic_lines(source)

    result = {
        "contaminated": False,
        "contamination_reasons": [],
        "suspicious": [],
        "trivial": False,
        "tactic_count": len(tactic_lines),
    }

    # Check banned keywords in tactic lines
    for line in tactic_lines:
        matches = BANNED_RE.findall(line)
        if matches:
            result["contaminated"] = True
            for kw in matches:
                reason = f"{kw} in tactic: {line[:100]}"
                if reason not in result["contamination_reasons"]:
                    result["contamination_reasons"].append(reason)

    # Check suspicious axioms in tactic lines
    for line in tactic_lines:
        matches = SUSPICIOUS_RE.findall(line)
        if matches:
            result["suspicious"].append(f"Decidable.decide in: {line[:100]}")

    # Check trivial proofs: single non-empty tactic that is trivial/rfl/decide
    non_empty = [l for l in tactic_lines if l.strip()]
    if len(non_empty) == 1 and non_empty[0].strip().rstrip(";").strip() in TRIVIAL_TACTICS:
        result["trivial"] = True

    return result


def main():
    # Load compile results
    print(f"Loading compile results from {COMPILE_RESULTS}")
    with open(COMPILE_RESULTS) as f:
        compile_data = json.load(f)

    completed = compile_data["completed"]

    # Load manifest for problem_id mapping
    print(f"Loading manifest from {MANIFEST}")
    with open(MANIFEST) as f:
        manifest_list = json.load(f)

    # Build seq -> manifest entry lookup
    manifest_by_seq = {str(entry["seq"]): entry for entry in manifest_list}

    # Filter to passing proofs (ok or warn)
    passing_seqs = {
        seq: info
        for seq, info in completed.items()
        if info["status"] in ("ok", "warn")
    }
    print(f"Total compiled: {len(completed)}")
    print(f"Passing (ok/warn): {len(passing_seqs)}")

    # Sweep
    clean_theorems = []
    contaminated_theorems = []
    suspicious_count = 0
    trivial_count = 0
    contamination_breakdown = {}
    files_not_found = 0

    for seq, info in sorted(passing_seqs.items(), key=lambda x: int(x[0])):
        module = info["module"]
        # Module: GoedelMigration.Proof_NNNNN -> Proof_NNNNN.lean
        filename = module.split(".")[-1] + ".lean"
        filepath = PROOF_DIR / filename

        if not filepath.exists():
            files_not_found += 1
            continue

        manifest_entry = manifest_by_seq.get(seq, {})
        problem_id = manifest_entry.get("problem_id", f"unknown_{seq}")

        check = check_file(filepath)

        entry = {
            "seq": int(seq),
            "problem_id": problem_id,
            "module": module,
        }

        if check["contaminated"]:
            entry["reasons"] = check["contamination_reasons"]
            contaminated_theorems.append(entry)
            # Track breakdown by keyword
            for reason in check["contamination_reasons"]:
                kw = reason.split(" ")[0]
                contamination_breakdown[kw] = contamination_breakdown.get(kw, 0) + 1
        else:
            clean_theorems.append(entry)

        if check["suspicious"]:
            suspicious_count += 1

        if check["trivial"]:
            trivial_count += 1

    # Summary
    total_passing = len(passing_seqs)
    total_clean = len(clean_theorems)
    total_contaminated = len(contaminated_theorems)
    clean_pct = total_clean / total_passing * 100 if total_passing else 0
    contaminated_pct = total_contaminated / total_passing * 100 if total_passing else 0

    summary = {
        "total_compiled": len(completed),
        "total_passing": total_passing,
        "total_clean": total_clean,
        "total_contaminated": total_contaminated,
        "clean_percentage": round(clean_pct, 2),
        "contaminated_percentage": round(contaminated_pct, 2),
        "contamination_breakdown": contamination_breakdown,
        "suspicious_axiom_count": suspicious_count,
        "trivial_proof_count": trivial_count,
        "files_not_found": files_not_found,
    }

    # Print summary
    print()
    print("=" * 60)
    print("M.6 INTEGRITY SWEEP RESULTS")
    print("=" * 60)
    print(f"Total compiled:           {summary['total_compiled']:>6}")
    print(f"Passing (ok/warn):        {summary['total_passing']:>6}")
    print(f"Clean pool:               {summary['total_clean']:>6} ({clean_pct:.1f}%)")
    print(f"Contaminated:             {summary['total_contaminated']:>6} ({contaminated_pct:.1f}%)")
    if contamination_breakdown:
        print(f"  Breakdown:")
        for kw, count in sorted(contamination_breakdown.items()):
            print(f"    {kw}: {count}")
    print(f"Suspicious (Decidable):   {summary['suspicious_axiom_count']:>6}")
    print(f"Trivial (single tactic):  {summary['trivial_proof_count']:>6}")
    if files_not_found:
        print(f"Files not found:          {summary['files_not_found']:>6}")
    print("=" * 60)

    # Decision gate
    if clean_pct < 90:
        print(f"\nWARNING: Clean pool ({clean_pct:.1f}%) is below 90% of passing proofs!")
        print("Consider investigating contamination sources before proceeding to tracing.")
    else:
        print(f"\nClean pool ({clean_pct:.1f}%) is above 90% threshold. Ready for tracing.")

    if contaminated_theorems:
        print(f"\nContaminated theorems:")
        for entry in contaminated_theorems:
            print(f"  seq={entry['seq']:>5}  {entry['problem_id']:30s}  {entry['reasons']}")

    # Write report
    report = {
        "clean_theorems": clean_theorems,
        "contaminated_theorems": contaminated_theorems,
        "summary": summary,
    }

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to {REPORT_OUT}")
    return 0 if clean_pct >= 90 else 1


if __name__ == "__main__":
    sys.exit(main())
