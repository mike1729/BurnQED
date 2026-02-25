"""Loss functions for joint LLM+EBM training.

Gotcha #1: InfoNCE has NO temperature parameter. The EBM head's learnable
temperature handles scaling. Do NOT add tau to info_nce_loss().
"""

from __future__ import annotations

import torch


def info_nce_loss(
    energies_pos: torch.Tensor,
    energies_neg: torch.Tensor,
) -> torch.Tensor:
    """InfoNCE contrastive loss over energy scores.

    Lower energy = more provable. The loss encourages positive (proved) states
    to have lower energy than negative (failed) states.

    No temperature parameter — the EBM head's learnable temperature handles
    scaling (Gotcha #1).

    Args:
        energies_pos: Energy scores for positive (proved) states (B, 1).
        energies_neg: Energy scores for negative (failed) states (B, K).
            K = number of negatives per positive.

    Returns:
        Scalar InfoNCE loss.
    """
    raise NotImplementedError("Phase 0 stub — implement in Phase 3")


def sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard cross-entropy loss for next-token prediction.

    Labels should be masked (ignore_index) for non-tactic tokens (Gotcha #13).

    Args:
        logits: Model output logits (B, seq_len, vocab_size).
        labels: Target token IDs (B, seq_len). Non-tactic positions = ignore_index.
        ignore_index: Token ID to ignore in loss computation.

    Returns:
        Scalar cross-entropy loss.
    """
    raise NotImplementedError("Phase 0 stub — implement in Phase 3")
