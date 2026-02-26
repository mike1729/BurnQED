#!/usr/bin/env python3
"""Standalone separation probe: measures embedding separation between
positive/negative proof states for a given model checkpoint.

Usage:
    python python/run_separation_probe.py \
        --model models/llm/iter_5 \
        --probe-data data/separation_probe.json \
        --dtype nf4

    # With LoRA adapter (unmerged):
    python python/run_separation_probe.py \
        --model models/llm/iter_4 \
        --adapter checkpoints/llm/iter_5 \
        --probe-data data/separation_probe.json
"""

import argparse
import json
import logging
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path, adapter_path=None, dtype="nf4"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto", "trust_remote_code": True}

    if dtype == "nf4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == "fp16":
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float32

    logger.info("Loading model from %s (dtype=%s)", model_path, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    if adapter_path:
        from peft import PeftModel
        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def compute_embeddings(model, tokenizer, texts, max_seq_len=2048, batch_size=4):
    """Mean-pool last hidden state embeddings."""
    tokens = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_seq_len, return_tensors="pt",
    )

    all_embs = []
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    n_total = input_ids.shape[0]

    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            batch_ids = input_ids[i:i + batch_size].to(model.device)
            batch_mask = attention_mask[i:i + batch_size].to(model.device)

            outputs = model(batch_ids, attention_mask=batch_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1].float()

            mask_expanded = batch_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            all_embs.append(pooled.cpu())

            if (i // batch_size) % 10 == 0:
                logger.info("  Encoded %d/%d", min(i + batch_size, n_total), n_total)

    return torch.cat(all_embs, dim=0)


def compute_metrics(pos_embs, neg_embs):
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

    n_pos = pos_embs.shape[0]
    n_neg = neg_embs.shape[0]

    # Within-positive cosine
    n_sample = min(100, n_pos)
    idx = torch.randperm(n_pos)[:n_sample]
    within_pos_cos = (pos_normed[idx] @ pos_normed[idx].T).fill_diagonal_(0).sum() / (n_sample * (n_sample - 1))

    # Cross-class cosine
    n_cross = min(100, n_pos, n_neg)
    cross_cos = (pos_normed[:n_cross] @ neg_normed[:n_cross].T).mean()
    delta_cosine = (within_pos_cos - cross_cos).item()

    # Linear probe accuracy (simple logistic regression on centroids)
    all_embs = torch.cat([pos_embs, neg_embs], dim=0)
    labels = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)])

    # Project onto centroid difference direction
    direction = pos_centroid - neg_centroid
    direction = direction / direction.norm()
    projections = all_embs @ direction
    threshold = projections.mean()
    predictions = (projections > threshold).float()
    accuracy = (predictions == labels).float().mean().item()

    return {
        "centroid_l2": round(centroid_l2, 4),
        "delta_cosine": round(delta_cosine, 4),
        "norm_gap": round(norm_gap, 4),
        "pos_norm": round(pos_norm_mean, 4),
        "neg_norm": round(neg_norm_mean, 4),
        "linear_probe_acc": round(accuracy, 4),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def main():
    parser = argparse.ArgumentParser(description="Run separation probe on a model checkpoint")
    parser.add_argument("--model", required=True, help="Path to base model")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--probe-data", required=True, help="Path to separation_probe.json")
    parser.add_argument("--dtype", default="nf4", choices=["nf4", "bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--output", default=None, help="Save results JSON to file")
    args = parser.parse_args()

    # Load probe data
    probe_data = json.load(open(args.probe_data))
    # Embeddings use raw proof state text (no instruction prefix or code fences)
    # See docs/data_format_spec.md "Embedding extraction" section
    pos_texts = [d['state_pp'] for d in probe_data if d["label"] == "positive"]
    neg_texts = [d['state_pp'] for d in probe_data if d["label"] == "negative"]
    logger.info("Probe data: %d positive, %d negative", len(pos_texts), len(neg_texts))

    # Load model
    model, tokenizer = load_model(args.model, args.adapter, args.dtype)

    # Compute embeddings
    logger.info("Computing positive embeddings...")
    pos_embs = compute_embeddings(model, tokenizer, pos_texts, args.max_seq_len, args.batch_size)
    logger.info("Computing negative embeddings...")
    neg_embs = compute_embeddings(model, tokenizer, neg_texts, args.max_seq_len, args.batch_size)

    # Compute metrics
    metrics = compute_metrics(pos_embs, neg_embs)

    logger.info("=" * 60)
    logger.info("SEPARATION PROBE RESULTS")
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info("  %-20s %s", k, v)
    logger.info("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
