#!/usr/bin/env python3
"""Merge LoRA adapters into base model and export safetensors for candle.

The output directory contains everything needed by TacticGenerator::load():
  - config.json
  - model*.safetensors (sharded)
  - model.safetensors.index.json
  - tokenizer.json
  - tokenizer_config.json

Usage:
    python python/training/export_llm.py \
        --checkpoint checkpoints/llm/iter_0 \
        --output models/llm/iter_0

    # Verify the merged model
    python python/training/export_llm.py \
        --checkpoint checkpoints/llm/iter_0 \
        --output models/llm/iter_0 \
        --verify
"""

import argparse
import logging
import shutil
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def merge_and_export(args):
    """Merge LoRA into base model and save as safetensors."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Load base model in float32 on CPU for precise merging
    logger.info("Loading base model: %s (float32, CPU)", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Load and merge LoRA adapter
    logger.info("Loading LoRA adapter from %s", checkpoint)
    model = PeftModel.from_pretrained(model, str(checkpoint))

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # Save merged model as safetensors
    logger.info("Saving merged model to %s", output)
    model.save_pretrained(
        str(output),
        safe_serialization=True,
        max_shard_size="5GB",
    )

    # Copy tokenizer files
    logger.info("Copying tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(str(output))

    # Also copy config.json if not already saved
    config_src = Path(args.base_model) / "config.json"
    config_dst = output / "config.json"
    if config_src.exists() and not config_dst.exists():
        shutil.copy2(config_src, config_dst)
        logger.info("Copied config.json from base model")

    # Report output size
    total_size = sum(f.stat().st_size for f in output.rglob("*") if f.is_file())
    logger.info("Output directory: %s (%.2f GB)", output, total_size / (1024**3))

    # List output files
    logger.info("Output files:")
    for f in sorted(output.iterdir()):
        size_mb = f.stat().st_size / (1024**2) if f.is_file() else 0
        logger.info("  %s (%.1f MB)", f.name, size_mb)

    # Verify if requested
    if args.verify:
        verify_merged(output)


def verify_merged(output_dir: Path):
    """Verify the merged model by running a forward pass."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Verifying merged model...")

    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(output_dir),
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Run a forward pass with a sample proof state (DeepSeek-native format)
    test_input = "Complete the following Lean 4 code:\n\n```lean4\n/- tactic state:\nn : Nat\n⊢ n + 0 = n\n-/\n```\n"
    inputs = tokenizer(test_input, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    logger.info(
        "Forward pass OK. Logits shape: %s, dtype: %s",
        list(logits.shape),
        logits.dtype,
    )
    logger.info("Hidden size: %d", model.config.hidden_size)
    logger.info("Vocab size: %d", model.config.vocab_size)
    logger.info("Verification passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters and export safetensors for candle.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="LoRA checkpoint directory from train_llm.py",
    )
    parser.add_argument(
        "--base-model",
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
        help="Base model name or path (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for merged safetensors",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a forward pass to verify the merged model",
    )
    args = parser.parse_args()

    merge_and_export(args)


if __name__ == "__main__":
    main()
