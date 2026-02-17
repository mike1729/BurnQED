#!/usr/bin/env python3
"""QLoRA fine-tuning of DeepSeek-Prover-V2-7B on tactic pairs.

Supports expert iteration: pass --extra-data with trajectory Parquet files from
previous iterations to augment the base training data with self-generated proofs.

Usage:
    # Iteration 0: train on Mathlib tactic pairs only
    accelerate launch python/training/train_llm.py \
        --model-name deepseek-ai/DeepSeek-Prover-V2-7B \
        --data data/tactic_pairs/train_formatted.jsonl \
        --val-data data/tactic_pairs/val_formatted.jsonl \
        --output checkpoints/llm/iter_0

    # Iteration N>0: add trajectory data from previous iterations
    accelerate launch python/training/train_llm.py \
        --model-name deepseek-ai/DeepSeek-Prover-V2-7B \
        --data data/tactic_pairs/train_formatted.jsonl \
        --extra-data trajectories/iter_*.parquet \
        --base checkpoints/llm/iter_0 \
        --output checkpoints/llm/iter_1 \
        --epochs 1 --lr 1e-4
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_base_data(data_path: str) -> list:
    """Load formatted training JSONL (output of prepare_tactic_pairs.py)."""
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d base training examples from %s", len(records), data_path)
    return records


def load_trajectory_data(parquet_globs: list) -> list:
    """Load positive trajectory examples from Parquet files.

    Reads TrajectoryRecord fields: state_pp, tactic_applied, label.
    Filters to label == "positive", removes sorry-containing tactics and
    empty state/tactic records, deduplicates, and formats as
    [GOAL]{state}[PROOFSTEP]{tactic}.
    """
    import pyarrow.parquet as pq

    records = []
    seen = set()
    files = []
    total_sorry = 0
    total_empty = 0
    total_dup = 0
    for pattern in parquet_globs:
        files.extend(sorted(glob.glob(pattern)))

    if not files:
        logger.warning("No Parquet files matched: %s", parquet_globs)
        return records

    for path in files:
        try:
            table = pq.read_table(path, columns=["state_pp", "tactic_applied", "label"])
            df = table.to_pandas()
            positive = df[df["label"] == "positive"]
            file_added = 0
            file_sorry = 0
            file_empty = 0
            file_dup = 0

            for _, row in positive.iterrows():
                state = str(row["state_pp"]).strip()
                tactic = str(row["tactic_applied"]).strip()
                if not state or not tactic:
                    file_empty += 1
                    continue
                if "sorry" in tactic:
                    file_sorry += 1
                    continue
                key = (state, tactic)
                if key in seen:
                    file_dup += 1
                    continue
                seen.add(key)
                text = f"[GOAL]{state}[PROOFSTEP]{tactic}"
                records.append({"text": text, "theorem": "", "depth": 0})
                file_added += 1

            total_sorry += file_sorry
            total_empty += file_empty
            total_dup += file_dup
            logger.info(
                "Loaded %d usable examples from %s (%d positive, %d total rows, "
                "filtered: %d empty, %d sorry, %d dup)",
                file_added, path, len(positive), len(df),
                file_empty, file_sorry, file_dup,
            )
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)

    logger.info(
        "Total trajectory examples: %d unique (filtered: %d empty, %d sorry, %d dup)",
        len(records), total_empty, total_sorry, total_dup,
    )
    return records


def build_dataset(records: list, tokenizer, max_seq_len: int):
    """Build a HuggingFace Dataset from formatted text records."""
    from datasets import Dataset

    texts = [r["text"] for r in records]

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    logger.info("Built dataset with %d examples", len(dataset))
    return dataset


def pack_sequences(dataset, eos_token_id: int, max_seq_len: int):
    """Pack tokenized sequences into fixed-length chunks for efficient training.

    Concatenates all input_ids with EOS separators, then chunks into
    fixed max_seq_len pieces. Each chunk becomes one training example
    with labels = input_ids (standard causal LM training).
    """
    from datasets import Dataset as HFDataset

    # Concatenate all sequences with EOS between them
    all_ids = []
    for ids in dataset["input_ids"]:
        all_ids.extend(ids)
        all_ids.append(eos_token_id)

    # Chunk into fixed-length sequences
    packed_input_ids = []
    for i in range(0, len(all_ids) - max_seq_len + 1, max_seq_len):
        chunk = all_ids[i : i + max_seq_len]
        packed_input_ids.append(chunk)

    # Drop remainder (< max_seq_len tokens)
    original_count = len(dataset)
    packed_count = len(packed_input_ids)
    total_tokens = len(all_ids)
    logger.info(
        "Packed %d examples → %d chunks of %d tokens (%.1fx compression, %d tokens total, %d remainder dropped)",
        original_count,
        packed_count,
        max_seq_len,
        original_count / max(packed_count, 1),
        total_tokens,
        total_tokens - packed_count * max_seq_len,
    )

    packed_dataset = HFDataset.from_dict({
        "input_ids": packed_input_ids,
        "labels": packed_input_ids,  # causal LM: labels = input_ids
        "attention_mask": [[1] * max_seq_len for _ in range(packed_count)],
    })
    return packed_dataset


def train(args):
    """Run QLoRA fine-tuning."""
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    # Quantization config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    logger.info("Loading base model: %s", args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # Required for gradient checkpointing
    model.enable_input_require_grads()  # Required for QLoRA + gradient checkpointing

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    if args.base:
        # Load previous LoRA adapter for continued training
        logger.info("Loading LoRA adapter from %s", args.base)
        model = PeftModel.from_pretrained(model, args.base, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %d / %d (%.2f%%)",
        trainable_params,
        total_params,
        100.0 * trainable_params / total_params,
    )

    # Load training data (optionally subsampled)
    import random
    train_records = load_base_data(args.data)
    if args.base_subsample and args.base_subsample < len(train_records):
        random.seed(args.seed)
        train_records = random.sample(train_records, args.base_subsample)
        logger.info("Subsampled base data: %d examples", len(train_records))

    # Add trajectory data from previous iterations (optionally upsampled)
    if args.extra_data:
        extra_records = load_trajectory_data(args.extra_data)
        if extra_records and args.trajectory_upsample > 1:
            original_count = len(extra_records)
            extra_records = extra_records * args.trajectory_upsample
            logger.info(
                "Upsampled trajectory data: %d → %d (×%d)",
                original_count,
                len(extra_records),
                args.trajectory_upsample,
            )
        if extra_records:
            train_records.extend(extra_records)
            logger.info(
                "Combined training set: %d examples (base: %d, trajectory: %d)",
                len(train_records),
                len(train_records) - len(extra_records),
                len(extra_records),
            )

    train_dataset = build_dataset(train_records, tokenizer, args.max_seq_len)

    # Pack training sequences for efficiency (unless --no-pack)
    if args.pack:
        train_dataset = pack_sequences(train_dataset, tokenizer.eos_token_id, args.max_seq_len)

    # Validation dataset (optional, always unpacked — padding is fine for eval)
    eval_dataset = None
    if args.val_data:
        val_records = load_base_data(args.val_data)
        eval_dataset = build_dataset(val_records, tokenizer, args.max_seq_len)

    if args.pack and not eval_dataset:
        # All sequences are fixed-length, no padding needed
        data_collator = default_data_collator
    else:
        # Need padding collator (for unpacked train, or packed train + unpacked eval).
        # DataCollatorForSeq2Seq handles both: fixed-length inputs pass through,
        # variable-length inputs get padded.
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
        )

    # Detect bf16 support
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Training arguments
    output_dir = Path(args.output)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    # Save final adapter
    logger.info("Saving LoRA adapter to %s", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Log metrics
    metrics = train_result.metrics
    logger.info("Training complete. Metrics: %s", json.dumps(metrics, indent=2))

    if eval_dataset:
        eval_metrics = trainer.evaluate()
        logger.info("Eval metrics: %s", json.dumps(eval_metrics, indent=2))

    # Save training summary
    summary = {
        "model_name": args.model_name,
        "base_adapter": args.base,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_examples": len(train_records),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "metrics": metrics,
    }
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training summary saved to %s", summary_path)


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of DeepSeek-Prover-V2-7B on tactic pairs.",
    )
    parser.add_argument(
        "--model-name",
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
        help="Base model name or path (default: %(default)s)",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Training JSONL path (output of prepare_tactic_pairs.py)",
    )
    parser.add_argument(
        "--val-data",
        default=None,
        help="Validation JSONL path (optional)",
    )
    parser.add_argument(
        "--extra-data",
        action="append",
        default=None,
        help="Glob pattern for trajectory Parquet files (repeatable, e.g. "
        "--extra-data 'iter_0*.parquet' --extra-data 'negatives_*.parquet')",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Previous LoRA checkpoint to resume from (iterations > 0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Max training steps (0 = use epochs, default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: %(default)s, effective batch = 32)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: %(default)s)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: %(default)s)",
    )
    parser.add_argument(
        "--base-subsample",
        type=int,
        default=None,
        help="Subsample base data to this many examples (default: use all)",
    )
    parser.add_argument(
        "--trajectory-upsample",
        type=int,
        default=1,
        help="Repeat trajectory examples this many times (default: %(default)s, no upsampling)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Maximum sequence length (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--no-pack",
        action="store_true",
        default=False,
        help="Disable sequence packing (use dynamic padding instead)",
    )
    args = parser.parse_args()
    args.pack = not args.no_pack

    train(args)


if __name__ == "__main__":
    main()
