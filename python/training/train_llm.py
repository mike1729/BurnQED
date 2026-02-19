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


class SeparationProbeCallback:
    """Monitors embedding separation between positive/negative proof states during training.

    Computes centroid distance, cosine similarity gap, and norm statistics
    every `probe_interval` steps. Saves the checkpoint with best centroid L2
    separation to `{output_dir}/best_separation/`.

    Implements TrainerCallback interface (on_step_end). The actual base class
    import is deferred to train() to avoid top-level transformers dependency.
    """

    def __init__(self, probe_path: str, tokenizer, max_seq_len: int, output_dir: str,
                 probe_interval: int = 500, batch_size: int = 8):
        probe_data = json.load(open(probe_path))
        pos_texts = [f"[GOAL]{d['state_pp']}" for d in probe_data if d["label"] == "positive"]
        neg_texts = [f"[GOAL]{d['state_pp']}" for d in probe_data if d["label"] == "negative"]
        self.n_pos = len(pos_texts)
        self.n_neg = len(neg_texts)
        logger.info("Separation probe: %d positive, %d negative states from %s",
                     self.n_pos, self.n_neg, probe_path)

        all_texts = pos_texts + neg_texts
        self.tokens = tokenizer(
            all_texts, padding=True, truncation=True,
            max_length=max_seq_len, return_tensors="pt",
        )
        self.output_dir = output_dir
        self.probe_interval = probe_interval
        self.batch_size = batch_size
        self.best_centroid_l2 = 0.0
        self.trainer = None  # Set after Trainer creation

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.probe_interval != 0:
            return

        model = kwargs.get("model") or self.trainer.model
        model.eval()

        all_embs = []
        input_ids = self.tokens["input_ids"]
        attention_mask = self.tokens["attention_mask"]
        n_total = input_ids.shape[0]

        with torch.no_grad():
            for i in range(0, n_total, self.batch_size):
                batch_ids = input_ids[i:i + self.batch_size].to(model.device)
                batch_mask = attention_mask[i:i + self.batch_size].to(model.device)

                outputs = model(batch_ids, attention_mask=batch_mask, output_hidden_states=True)
                hidden = outputs.hidden_states[-1].float()  # [B, seq_len, hidden_dim]

                # Mean-pool using attention mask
                mask_expanded = batch_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
                pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
                all_embs.append(pooled.cpu())

        model.train()

        all_embs = torch.cat(all_embs, dim=0)  # [n_total, hidden_dim]
        pos_embs = all_embs[:self.n_pos]
        neg_embs = all_embs[self.n_pos:]

        # Centroid L2 distance
        pos_centroid = pos_embs.mean(dim=0)
        neg_centroid = neg_embs.mean(dim=0)
        centroid_l2 = (pos_centroid - neg_centroid).norm().item()

        # Norms
        pos_norms = pos_embs.norm(dim=1)
        neg_norms = neg_embs.norm(dim=1)
        pos_norm_mean = pos_norms.mean().item()
        neg_norm_mean = neg_norms.mean().item()
        norm_gap = neg_norm_mean - pos_norm_mean

        # Cosine similarities
        pos_normed = pos_embs / pos_norms.unsqueeze(1).clamp(min=1e-9)
        neg_normed = neg_embs / neg_norms.unsqueeze(1).clamp(min=1e-9)
        # Within-positive cosine (sample pairs to avoid O(n^2))
        n_sample = min(100, self.n_pos)
        idx = torch.randperm(self.n_pos)[:n_sample]
        within_pos_cos = (pos_normed[idx] @ pos_normed[idx].T).fill_diagonal_(0).sum() / (n_sample * (n_sample - 1))
        # Cross-class cosine
        n_cross = min(100, self.n_pos, self.n_neg)
        cross_cos = (pos_normed[:n_cross] @ neg_normed[:n_cross].T).mean()
        delta_cosine = (within_pos_cos - cross_cos).item()

        metrics = {
            "sep_centroid_l2": round(centroid_l2, 4),
            "sep_delta_cosine": round(delta_cosine, 4),
            "sep_norm_gap": round(norm_gap, 4),
            "sep_pos_norm": round(pos_norm_mean, 4),
            "sep_neg_norm": round(neg_norm_mean, 4),
        }

        logger.info("Step %d separation probe: %s", state.global_step, metrics)
        if self.trainer:
            self.trainer.log(metrics)

        # Save best separation checkpoint
        if centroid_l2 > self.best_centroid_l2:
            self.best_centroid_l2 = centroid_l2
            save_dir = os.path.join(self.output_dir, "best_separation")
            logger.info("New best centroid L2: %.4f — saving to %s", centroid_l2, save_dir)
            model.save_pretrained(save_dir)


def train(args):
    """Run QLoRA fine-tuning."""
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
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
        # Attention-only LoRA by default.
        # Use --lora-mlp to include MLP layers (gate/up/down_proj).
        # Analysis shows MLP LoRA is critical for creating L2 norm-based
        # separation between positive/negative proof states in embeddings
        # (see docs/revelations.md). Recommend enabling for EBM training.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if args.lora_mlp:
            target_modules += ["gate_proj", "up_proj", "down_proj"]
            logger.info("LoRA targeting attention + MLP layers")
        else:
            logger.info("LoRA targeting attention layers only")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
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

    # Add trajectory data from previous iterations (90/10 train/val split)
    traj_val_records = []
    if args.extra_data:
        extra_records = load_trajectory_data(args.extra_data)
        if extra_records:
            random.seed(args.seed)
            random.shuffle(extra_records)
            split_idx = max(1, len(extra_records) // 10)
            traj_val_records = extra_records[:split_idx]
            traj_train_records = extra_records[split_idx:]
            logger.info(
                "Trajectory split: %d train / %d val",
                len(traj_train_records),
                len(traj_val_records),
            )
            base_count = len(train_records)
            train_records.extend(traj_train_records)
            logger.info(
                "Combined training set: %d examples (base: %d, trajectory: %d)",
                len(train_records),
                base_count,
                len(traj_train_records),
            )

    train_dataset = build_dataset(train_records, tokenizer, args.max_seq_len)

    # Pack training sequences for efficiency (unless --no-pack)
    if args.pack:
        train_dataset = pack_sequences(train_dataset, tokenizer.eos_token_id, args.max_seq_len)

    # Validation dataset
    eval_dataset = None
    full_eval_dataset = None
    if args.val_data:
        val_records = load_base_data(args.val_data)
        full_eval_dataset = build_dataset(val_records, tokenizer, args.max_seq_len)
        if args.pack:
            full_eval_dataset = pack_sequences(full_eval_dataset, tokenizer.eos_token_id, args.max_seq_len)
        # Quick subset for frequent eval (every 100 steps)
        eval_subset_size = 500
        if len(val_records) > eval_subset_size:
            import random as _rng
            _rng.seed(args.seed)
            eval_subset_records = _rng.sample(val_records, eval_subset_size)
            eval_dataset = build_dataset(eval_subset_records, tokenizer, args.max_seq_len)
            if args.pack:
                eval_dataset = pack_sequences(eval_dataset, tokenizer.eos_token_id, args.max_seq_len)
            logger.info("Eval subset: %d / %d examples (quick eval every 100 steps, full eval every 500)",
                        eval_subset_size, len(val_records))
        else:
            eval_dataset = full_eval_dataset

    # Trajectory eval dataset (from held-out 10% of trajectory data)
    traj_eval_dataset = None
    if traj_val_records:
        traj_eval_dataset = build_dataset(traj_val_records, tokenizer, args.max_seq_len)
        if args.pack:
            traj_eval_dataset = pack_sequences(traj_eval_dataset, tokenizer.eos_token_id, args.max_seq_len)

    unit = "packed chunks" if args.pack else "examples"
    traj_str = f", traj_val: {len(traj_eval_dataset)} {unit}" if traj_eval_dataset else ""
    logger.info(
        "Dataset split — train: %d %s, val: %s%s",
        len(train_dataset),
        unit,
        f"{len(eval_dataset)} {unit} (quick) / {len(full_eval_dataset)} {unit} (full)" if eval_dataset else "none",
        traj_str,
    )

    if args.pack:
        # All datasets are fixed-length packed chunks, no padding needed
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
        )

    # Detect bf16 support
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Training arguments
    output_dir = Path(args.output)
    if args.save_steps:
        save_steps = args.save_steps
    elif args.max_steps >= 10000:
        save_steps = 2000
    else:
        save_steps = 500
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size if args.pack else args.batch_size * 4,
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
        save_steps=save_steps,
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

    # Full eval callback every 500 steps (quick subset runs every 100 via eval_steps)
    has_periodic_eval = (full_eval_dataset is not None and full_eval_dataset is not eval_dataset) or traj_eval_dataset is not None
    if has_periodic_eval:
        class FullEvalCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step > 0 and state.global_step % 500 == 0:
                    if full_eval_dataset is not None and full_eval_dataset is not eval_dataset:
                        metrics = trainer.evaluate(
                            eval_dataset=full_eval_dataset,
                            metric_key_prefix="full_eval",
                        )
                        trainer.log(metrics)
                    if traj_eval_dataset is not None:
                        traj_metrics = trainer.evaluate(
                            eval_dataset=traj_eval_dataset,
                            metric_key_prefix="traj_eval",
                        )
                        trainer.log(traj_metrics)

        trainer.add_callback(FullEvalCallback())

    # Separation probe callback for monitoring embedding separation during training
    if args.probe_data:
        probe_cb = SeparationProbeCallback(
            probe_path=args.probe_data,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            output_dir=str(output_dir),
        )
        probe_cb.trainer = trainer
        trainer.add_callback(probe_cb)

    logger.info("Starting training...")
    train_result = trainer.train()

    # Save final adapter
    logger.info("Saving LoRA adapter to %s", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Log metrics
    metrics = train_result.metrics
    logger.info("Training complete. Metrics: %s", json.dumps(metrics, indent=2))

    if full_eval_dataset:
        eval_metrics = trainer.evaluate(eval_dataset=full_eval_dataset, metric_key_prefix="final_eval")
        logger.info("Final eval metrics: %s", json.dumps(eval_metrics, indent=2))
    if traj_eval_dataset:
        traj_eval_metrics = trainer.evaluate(eval_dataset=traj_eval_dataset, metric_key_prefix="final_traj_eval")
        logger.info("Final traj eval metrics: %s", json.dumps(traj_eval_metrics, indent=2))

    # Save training summary
    summary = {
        "model_name": args.model_name,
        "base_adapter": args.base,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_mlp": args.lora_mlp,
        "probe_data": args.probe_data,
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
        "--lora-mlp",
        action="store_true",
        default=False,
        help="Also apply LoRA to MLP layers (gate/up/down_proj). "
        "Recommended for EBM training — MLP LoRA creates L2 norm separation.",
    )
    parser.add_argument(
        "--base-subsample",
        type=int,
        default=None,
        help="Subsample base data to this many examples (default: use all)",
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
    parser.add_argument(
        "--probe-data",
        default=None,
        help="Path to separation probe JSON (enables SeparationProbeCallback)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Override save_steps (default: 2000 if max_steps>=10000, else 500)",
    )
    args = parser.parse_args()
    args.pack = not args.no_pack

    train(args)


if __name__ == "__main__":
    main()
