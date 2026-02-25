"""Monitoring utilities for joint training health.

Gotcha #3: Monitor temperature every 50 steps. Healthy range [0.5, 3.0].
If temperature hits floor or ceiling → ABORT training.
"""

from __future__ import annotations

from typing import Optional

import torch


def separation_probe(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_ratio: float = 0.8,
) -> dict[str, float]:
    """Linear probe measuring embedding class separation.

    Trains a logistic regression on (embedding, proved/failed label) pairs
    and reports accuracy. High accuracy → embeddings maintain discriminative
    structure for the EBM.

    Args:
        embeddings: Embedding vectors (N, d_encoder).
        labels: Binary labels (N,) — 1 for proved, 0 for failed.
        train_ratio: Fraction of data used for training the probe.

    Returns:
        Dict with "linear_probe_acc", "centroid_l2", "train_acc".
    """
    raise NotImplementedError("Phase 0 stub — implement in Phase 2")


def ebm_metrics(
    energies: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Compute EBM ranking quality metrics.

    Args:
        energies: Predicted energy scores (N,).
        labels: Binary labels (N,) — 1 for proved, 0 for failed.

    Returns:
        Dict with "rank_1_acc", "mean_energy_pos", "mean_energy_neg",
        "energy_gap", "auroc".
    """
    raise NotImplementedError("Phase 0 stub — implement in Phase 2")


def log_temperature(
    temperature: float,
    step: int,
    logger: Optional[object] = None,
    abort_on_boundary: bool = True,
) -> None:
    """Log EBM head temperature and check health bounds.

    Gotcha #3: Healthy range is [0.5, 3.0]. Outside this range indicates
    training instability.

    Args:
        temperature: Current learnable temperature value.
        step: Training step number.
        logger: Optional wandb/tensorboard logger.
        abort_on_boundary: If True, raise RuntimeError when temperature
            exits healthy range.

    Raises:
        RuntimeError: If temperature < 0.5 or > 3.0 and abort_on_boundary is True.
    """
    raise NotImplementedError("Phase 0 stub — implement in Phase 2")
