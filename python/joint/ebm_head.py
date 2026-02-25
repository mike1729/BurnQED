"""GoalConditionedEnergyHead and EmbeddingExtractor.

Architecture: [z_state; z_goal; z_state*z_goal] -> 12288 -> 2048 -> 1024 -> 512 -> 1
All layers: spectral norm, SiLU activation, dropout 0.15.
Learnable temperature parameter (Gotcha #1: InfoNCE has NO temperature — head handles it).
Init: first layer weight *= 0.1 (Gotcha #2: prevents 25M param init explosion).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class GoalConditionedEnergyHead(nn.Module):
    """Scores (state, goal) pairs via energy function.

    Input: z_state (B, d_encoder), z_goal (B, d_encoder)
    Output: energy scalar (B, 1) — lower = more provable.
    """

    def __init__(
        self,
        d_encoder: int = 4096,
        d_hidden: tuple[int, ...] = (2048, 1024, 512),
        dropout: float = 0.15,
        n_power_iterations: int = 5,
    ) -> None:
        raise NotImplementedError("Phase 0 stub — implement in Phase 2")

    def forward(
        self, z_state: torch.Tensor, z_goal: torch.Tensor
    ) -> torch.Tensor:
        """Compute energy for (state, goal) pairs.

        Args:
            z_state: Proof state embeddings (B, d_encoder).
            z_goal: Goal statement embeddings (B, d_encoder).

        Returns:
            Scaled energy (B, 1). Scaling uses learnable temperature.
        """
        raise NotImplementedError("Phase 0 stub — implement in Phase 2")

    @property
    def temperature(self) -> float:
        """Current temperature value (for monitoring, Gotcha #3)."""
        raise NotImplementedError("Phase 0 stub — implement in Phase 2")


class EmbeddingExtractor(nn.Module):
    """Mean-pool hidden states from the shared backbone.

    Extracts embeddings from the last hidden layer, mean-pooling over
    non-padding tokens. Used for both z_state and z_goal extraction.
    """

    def __init__(self, hidden_size: int = 4096) -> None:
        raise NotImplementedError("Phase 0 stub — implement in Phase 2")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool hidden states over non-padding tokens.

        Args:
            hidden_states: (B, seq_len, hidden_size) from backbone.
            attention_mask: (B, seq_len) binary mask.

        Returns:
            Pooled embeddings (B, hidden_size).
        """
        raise NotImplementedError("Phase 0 stub — implement in Phase 2")
