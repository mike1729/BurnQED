#!/usr/bin/env python3
"""Compile miniF2F benchmark .lean files against Lean 4.27 / Mathlib v4.27.

Runs from project root. Orchestrates:
1. Download miniF2F datasets (if not cached)
2. Generate sorry-proof .lean files
3. Compile each file with `lake env lean` (verify statements)
4. Build .oleans via `lake build` (for fast Pantograph imports)
5. Report per-theorem results

Usage:
    cd /path/to/BurnQED
    python python/data/minif2f/compile.py                    # all variants
    python python/data/minif2f/compile.py --variant v2s_test # single variant
    python python/data/minif2f/compile.py --skip-download    # reuse existing JSONs
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

VARIANTS = ["test", "valid", "v2s_test", "v2s_valid", "v2c_test", "v2c_valid"]
PANTOGRAPH_DIR = Path("vendor/Pantograph")
BENCHMARKS_DIR = Path("data/benchmarks")


def variant_to_module(variant: str) -> str:
    """Convert variant name to Lean module name: v2s_test -> BenchMinIF2FV2sTest."""
    parts = variant.split("_")
    camel = "".join(p.capitalize() for p in parts)
    return f"BenchMinIF2F{camel}"


def download(force: bool = False):
    """Run download script."""
    cmd = [
        sys.executable,
        "python/data/minif2f/download.py",
        "--output-dir", str(BENCHMARKS_DIR),
    ]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True)


def generate(variant: str):
    """Generate .lean file for a variant."""
    module = variant_to_module(variant)
    json_path = BENCHMARKS_DIR / f"minif2f_{variant}.json"
    lean_path = PANTOGRAPH_DIR / f"{module}.lean"

    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run download first.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "python/data/minif2f/generate_lean.py",
        "--input", str(json_path),
        "--output", str(lean_path),
        "--module-name", module,
    ]
    subprocess.run(cmd, check=True)
    return lean_path


def ensure_lean_lib(module: str):
    """Register lean_lib target in lakefile.lean if not already present."""
    lakefile = PANTOGRAPH_DIR / "lakefile.lean"
    content = lakefile.read_text()
    marker = f"lean_lib {module}"
    if marker in content:
        return
    content += f"\nlean_lib {module} {{}}\n"
    lakefile.write_text(content)
    print(f"  Registered {module} in lakefile.lean")


def build_olean(module: str, timeout: int = 600) -> bool:
    """Build .olean for a module via `lake build`. Returns True on success."""
    ensure_lean_lib(module)
    cmd = ["lake", "build", module]
    print(f"  Building {module} .olean...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(PANTOGRAPH_DIR),
        )
        if result.returncode != 0:
            # Filter out sorry warnings
            errors = [
                ln for ln in (result.stdout + result.stderr).splitlines()
                if "error" in ln.lower() and "sorry" not in ln
            ]
            if errors:
                print(f"  WARNING: lake build {module} had errors:")
                for e in errors[:5]:
                    print(f"    {e}")
                return False
        print(f"  Built {module} .olean successfully")
        return True
    except subprocess.TimeoutExpired:
        print(f"  WARNING: lake build {module} timed out after {timeout}s")
        return False


def compile_file(lean_file: str, timeout: int) -> dict:
    """Compile a single .lean file with lake env lean."""
    cmd = ["lake", "env", "lean", lean_file]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(PANTOGRAPH_DIR),
        )
        stdout = result.stdout + result.stderr
        # Parse errors: lines with ":NN:NN: error:"
        errors = [
            ln for ln in stdout.splitlines()
            if ": error:" in ln
        ]
        # Parse warnings (excluding sorry)
        warnings = [
            ln for ln in stdout.splitlines()
            if ": warning:" in ln and "sorry" not in ln
        ]
        return {
            "file": lean_file,
            "returncode": result.returncode,
            "errors": errors,
            "warnings": warnings,
            "log": stdout,
        }
    except subprocess.TimeoutExpired:
        return {
            "file": lean_file,
            "returncode": -1,
            "errors": [f"TIMEOUT after {timeout}s"],
            "warnings": [],
            "log": f"Timeout > {timeout}s",
        }


def extract_theorem_errors(log: str) -> dict[str, list[str]]:
    """Parse per-theorem errors from compilation log.

    Returns dict mapping theorem_name -> list of error messages.
    """
    theorem_errors: dict[str, list[str]] = {}
    # Error format: "file.lean:LINE:COL: error: MESSAGE"
    error_pattern = re.compile(r":(\d+):\d+: error: (.+)")

    for line in log.splitlines():
        m = error_pattern.search(line)
        if m:
            line_no = int(m.group(1))
            msg = m.group(2)
            theorem_errors.setdefault(f"line_{line_no}", []).append(msg)

    return theorem_errors


def compile_variant(variant: str, timeout: int, out_dir: Path) -> dict:
    """Compile a single variant and write results."""
    module = variant_to_module(variant)
    lean_file = f"{module}.lean"

    print(f"\nCompiling {module}...")
    result = compile_file(lean_file, timeout)

    # Load theorem list to map line numbers to names
    json_path = BENCHMARKS_DIR / f"minif2f_{variant}.json"
    with open(json_path) as f:
        theorems = json.load(f)["theorems"]

    # Read the .lean file to map line numbers to theorem names
    lean_path = PANTOGRAPH_DIR / lean_file
    line_to_name = {}
    for i, line in enumerate(lean_path.read_text().splitlines(), 1):
        m = re.match(r"theorem\s+([\w'.]+)", line)
        if m:
            line_to_name[i] = m.group(1)

    # Categorize errors by theorem
    error_theorems = set()
    error_details = []
    for err_line in result["errors"]:
        m = re.search(r":(\d+):\d+: error: (.+)", err_line)
        if m:
            line_no = int(m.group(1))
            msg = m.group(2)
            # Find the theorem this error belongs to (nearest theorem line <= error line)
            thm_name = None
            for tl in sorted(line_to_name.keys(), reverse=True):
                if tl <= line_no:
                    thm_name = line_to_name[tl]
                    break
            if thm_name:
                error_theorems.add(thm_name)
            error_details.append({
                "line": line_no,
                "theorem": thm_name,
                "message": msg,
            })

    # Warning theorems (non-sorry)
    warning_theorems = set()
    for warn_line in result["warnings"]:
        m = re.search(r":(\d+):\d+: warning: (.+)", warn_line)
        if m:
            line_no = int(m.group(1))
            for tl in sorted(line_to_name.keys(), reverse=True):
                if tl <= line_no:
                    warning_theorems.add(line_to_name[tl])
                    break

    total = len(theorems)
    n_errors = len(error_theorems)
    n_warnings = len(warning_theorems - error_theorems)
    n_ok = total - n_errors - n_warnings

    summary = {
        "variant": variant,
        "module": module,
        "total": total,
        "ok": n_ok,
        "warnings": n_warnings,
        "errors": n_errors,
        "error_theorems": sorted(error_theorems),
        "warning_theorems": sorted(warning_theorems - error_theorems),
        "error_details": error_details,
        "timeout": result["returncode"] == -1,
    }

    # Write results
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{variant}_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    if error_details:
        with open(out_dir / f"{variant}_errors.log", "w") as f:
            f.write(result["log"])

    pct = 100 * (total - n_errors) / total if total else 0
    print(f"  {module}: {n_ok} ok, {n_warnings} warn, {n_errors} err / {total} total ({pct:.1f}% pass)")
    if error_details:
        for ed in error_details[:5]:
            print(f"    {ed['theorem']}: {ed['message'][:100]}")
        if len(error_details) > 5:
            print(f"    ... and {len(error_details) - 5} more errors")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compile miniF2F benchmarks against Lean 4.27")
    parser.add_argument(
        "--variant", choices=VARIANTS + ["all"], default="all",
        help="Which variant(s) to compile (default: all)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download step (reuse existing JSONs)",
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip generate step (reuse existing .lean files)",
    )
    parser.add_argument(
        "--skip-compile", action="store_true",
        help="Skip compile step (just download, generate, build oleans)",
    )
    parser.add_argument(
        "--skip-olean", action="store_true",
        help="Skip .olean build step (lake build)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-file compile timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/benchmarks/compile_results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    variants = VARIANTS if args.variant == "all" else [args.variant]
    out_dir = Path(args.out_dir)

    # Step 1: Download
    if not args.skip_download:
        print("Step 1: Downloading miniF2F datasets...")
        download()
    else:
        print("Step 1: Skipping download (--skip-download)")

    # Step 2: Generate .lean files
    if not args.skip_generate:
        print("\nStep 2: Generating .lean files...")
        for v in variants:
            generate(v)
    else:
        print("\nStep 2: Skipping generate (--skip-generate)")

    # Step 3: Compile (lake env lean — verify statements)
    all_results = []
    if not args.skip_compile:
        print(f"\nStep 3: Compiling {len(variants)} variant(s) (timeout={args.timeout}s)...")
        for v in variants:
            summary = compile_variant(v, args.timeout, out_dir)
            all_results.append(summary)
    else:
        print("\nStep 3: Skipping compile (--skip-compile)")

    # Step 4: Build .oleans (lake build — for fast Pantograph imports)
    if not args.skip_olean:
        print(f"\nStep 4: Building .oleans for {len(variants)} variant(s)...")
        for v in variants:
            module = variant_to_module(v)
            # Skip olean build if compilation had errors
            if all_results:
                matching = [r for r in all_results if r["variant"] == v]
                if matching and matching[0]["errors"] > 0:
                    print(f"  Skipping {module} .olean (has compile errors)")
                    continue
            build_olean(module)
    else:
        print("\nStep 4: Skipping .olean build (--skip-olean)")

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("COMPILATION SUMMARY")
    print("=" * 60)
    total_theorems = 0
    total_ok = 0
    total_err = 0
    for r in all_results:
        pct = 100 * (r["total"] - r["errors"]) / r["total"] if r["total"] else 0
        print(f"  {r['variant']:12s}: {r['ok']:3d} ok, {r['warnings']:3d} warn, {r['errors']:3d} err / {r['total']} ({pct:.1f}%)")
        total_theorems += r["total"]
        total_ok += r["ok"] + r["warnings"]
        total_err += r["errors"]

    total_pass = total_theorems - total_err
    pct = 100 * total_pass / total_theorems if total_theorems else 0
    print(f"  {'TOTAL':12s}: {total_pass} pass, {total_err} err / {total_theorems} ({pct:.1f}%)")
    print(f"\nResults saved to {out_dir}/")

    # Write aggregate
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "total_theorems": total_theorems,
            "total_pass": total_pass,
            "total_errors": total_err,
            "pass_rate": round(pct, 1),
            "variants": all_results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
