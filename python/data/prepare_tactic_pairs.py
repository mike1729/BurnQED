#!/usr/bin/env python3
"""Convert raw tactic pairs to the LLM training format.

The output format matches the Rust policy crate's prompt format:
    [GOAL]{state}[PROOFSTEP]{tactic}<eos>

Usage:
    python python/data/prepare_tactic_pairs.py \
        --input data/tactic_pairs/train.jsonl \
        --output data/tactic_pairs/train_formatted.jsonl

    python python/data/prepare_tactic_pairs.py \
        --input data/tactic_pairs/val.jsonl \
        --output data/tactic_pairs/val_formatted.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Must match crates/policy/src/model.rs:248 — format_prompt()
PROMPT_FORMAT_BURNQED = "[GOAL]{state}[PROOFSTEP]{tactic}"


def format_pair(state: str, tactic: str, fmt: str) -> str:
    """Format a (state, tactic) pair into the LLM training format.

    Args:
        state: Pretty-printed proof state (e.g., "n : Nat\\n⊢ n + 0 = n")
        tactic: Tactic string (e.g., "simp")
        fmt: Format name — only "burnqed" is supported.

    Returns:
        Formatted string for LLM training.
    """
    if fmt == "burnqed":
        return f"[GOAL]{state}[PROOFSTEP]{tactic}"
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Supported: burnqed")


def load_tokenizer(tokenizer_path: str):
    """Load a HuggingFace tokenizer for length checking."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        logger.warning("Could not load tokenizer from %s: %s", tokenizer_path, e)
        logger.warning("Falling back to character-based length estimation (4 chars/token).")
        return None


def estimate_token_count(text: str, tokenizer) -> int:
    """Count tokens in text, or estimate if no tokenizer available."""
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    # Rough estimate: ~4 characters per token for code/math
    return len(text) // 4


def process_pairs(
    input_path: str,
    output_path: str,
    fmt: str,
    max_seq_len: int,
    tokenizer_path: str,
):
    """Read raw tactic pairs, format them, filter by length, and write output."""
    tokenizer = load_tokenizer(tokenizer_path)

    total = 0
    skipped_long = 0
    skipped_empty = 0
    written = 0

    input_file = Path(input_path)
    if not input_file.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file) as fin, open(output_file, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", line_num)
                continue

            state = record.get("state", "").strip()
            tactic = record.get("tactic", "").strip()
            theorem = record.get("theorem", "")
            depth = record.get("depth", 0)

            if not state or not tactic:
                skipped_empty += 1
                continue

            # Format the training example
            text = format_pair(state, tactic, fmt)

            # Check token length (including EOS token)
            token_count = estimate_token_count(text, tokenizer) + 1  # +1 for EOS
            if token_count > max_seq_len:
                skipped_long += 1
                continue

            output_record = {
                "text": text,
                "theorem": theorem,
                "depth": depth,
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            written += 1

    # Summary
    print("\n" + "=" * 50)
    print("  Tactic Pair Preparation Summary")
    print("=" * 50)
    print(f"  Input:             {input_path}")
    print(f"  Output:            {output_path}")
    print(f"  Format:            {fmt}")
    print(f"  Max seq length:    {max_seq_len}")
    print(f"  Total pairs:       {total:>8}")
    print(f"  Skipped (empty):   {skipped_empty:>8}")
    print(f"  Skipped (too long):{skipped_long:>8}")
    print(f"  Written:           {written:>8}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw tactic pairs to LLM training format.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw tactic pairs JSONL (from trace_mathlib.py)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path with formatted training examples",
    )
    parser.add_argument(
        "--format",
        default="burnqed",
        choices=["burnqed"],
        help="Prompt format (default: %(default)s). Must match Rust policy crate.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum token length; skip longer examples (default: %(default)s)",
    )
    parser.add_argument(
        "--tokenizer",
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
        help="Tokenizer path for length checking (default: %(default)s)",
    )
    args = parser.parse_args()

    process_pairs(
        input_path=args.input,
        output_path=args.output,
        fmt=args.format,
        max_seq_len=args.max_seq_len,
        tokenizer_path=args.tokenizer,
    )


if __name__ == "__main__":
    main()
