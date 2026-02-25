"""JointProver: shared backbone with LoRA + GoalConditioned EBM head.

Single forward pass produces both:
1. Logits for SFT cross-entropy loss (LM head)
2. Energy scores for InfoNCE contrastive loss (EBM head)

Shared gradients through LoRA ensure embedding quality is preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .ebm_head import EmbeddingExtractor, GoalConditionedEnergyHead


class JointProver(nn.Module):
    """Joint LLM policy + EBM value model.

    Wraps a HuggingFace causal LM with LoRA adapters and attaches
    a GoalConditionedEnergyHead for value estimation.

    Forward pass extracts hidden states for both LM logits and
    mean-pooled embeddings for the EBM head.
    """

    def __init__(
        self,
        model_name: str,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        d_encoder: int = 4096,
        ebm_dropout: float = 0.15,
        ebm_n_power_iterations: int = 5,
    ) -> None:
        """Initialize JointProver.

        Gotcha #14: Check if [GOAL]/[PROOFSTEP] tokens are in vocab.
        If not, add_special_tokens() + resize_token_embeddings().
        modules_to_save=["embed_tokens", "lm_head"] in LoRA config.

        Args:
            model_name: HuggingFace model ID (e.g., "deepseek-ai/DeepSeek-Prover-V2-7B").
            lora_r: LoRA rank.
            lora_alpha: LoRA alpha scaling.
            lora_dropout: LoRA dropout.
            d_encoder: Hidden size of the backbone.
            ebm_dropout: Dropout for EBM head.
            ebm_n_power_iterations: Spectral norm power iterations.
        """
        raise NotImplementedError("Phase 0 stub — implement in Phase 3")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_input_ids: Optional[torch.Tensor] = None,
        state_attention_mask: Optional[torch.Tensor] = None,
        goal_input_ids: Optional[torch.Tensor] = None,
        goal_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Joint forward pass.

        Args:
            input_ids: SFT sequence tokens (B_sft, seq_len).
            attention_mask: SFT attention mask (B_sft, seq_len).
            state_input_ids: Contrastive state tokens (B_con, seq_len).
            state_attention_mask: Contrastive state mask (B_con, seq_len).
            goal_input_ids: Contrastive goal tokens (B_con, seq_len).
            goal_attention_mask: Contrastive goal mask (B_con, seq_len).

        Returns:
            Dict with keys:
            - "logits": LM logits (B_sft, seq_len, vocab_size)
            - "energies": Energy scores (B_con, 1) if contrastive inputs provided
        """
        raise NotImplementedError("Phase 0 stub — implement in Phase 3")

    def save_ebm_head(self, path: Path) -> None:
        """Save only the EBM head weights (for decoupled inference)."""
        raise NotImplementedError("Phase 0 stub — implement in Phase 3")

    def save_lora(self, path: Path) -> None:
        """Save LoRA adapter weights."""
        raise NotImplementedError("Phase 0 stub — implement in Phase 3")
