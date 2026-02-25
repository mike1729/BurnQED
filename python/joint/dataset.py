"""JointDataset: interleaved SFT and contrastive training streams.

SFTStream: tactic pairs formatted as [GOAL]state[PROOFSTEP]tactic for next-token prediction.
ContrastiveStream: (state, goal, label) triples for InfoNCE contrastive learning.

Gotcha #11: Validation split by THEOREM NAME, not tactic pairs.
Gotcha #12: Reject entire theorem if any tactic contains sorry/admit/cheat.
Gotcha #13: DataCollatorForCompletionOnlyLM — only train on tactic tokens after [PROOFSTEP].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, IterableDataset


@dataclass
class SFTExample:
    """A single SFT training example."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


@dataclass
class ContrastiveExample:
    """A single contrastive training example."""
    state_text: str
    goal_text: str
    is_positive: bool


class SFTStream(IterableDataset):
    """Streams SFT examples from tactic pairs JSONL.

    Format per line: {"state": "...", "tactic": "...", "theorem": "..."}
    Tokenizes as: [GOAL]{state}[PROOFSTEP]{tactic}
    Loss masked to tactic tokens only (Gotcha #13).
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer_name: str,
        max_seq_len: int = 2048,
        sorry_filter: bool = True,
    ) -> None:
        raise NotImplementedError("Phase 0 stub — implement in Phase 3")

    def __iter__(self) -> Iterator[SFTExample]:
        raise NotImplementedError("Phase 0 stub — implement in Phase 3")


class ContrastiveStream(IterableDataset):
    """Streams contrastive examples from trajectory data.

    Positive: (proved_state, goal) pairs from successful proof paths.
    Negative: (failed_state, goal) pairs from failed branches.
    """

    def __init__(
        self,
        data_path: Path,
        sorry_filter: bool = True,
    ) -> None:
        raise NotImplementedError("Phase 0 stub — implement in Phase 3")

    def __iter__(self) -> Iterator[ContrastiveExample]:
        raise NotImplementedError("Phase 0 stub — implement in Phase 3")


class JointDataset(Dataset):
    """Combines SFT and contrastive streams for joint training.

    Each batch contains both SFT examples (for cross-entropy loss) and
    contrastive examples (for InfoNCE loss). The ratio is controlled by
    `sft_weight` in TrainingConfig.
    """

    def __init__(
        self,
        sft_path: Path,
        contrastive_path: Path,
        tokenizer_name: str,
        max_seq_len: int = 2048,
        sorry_filter: bool = True,
    ) -> None:
        raise NotImplementedError("Phase 0 stub — implement in Phase 3")

    def __len__(self) -> int:
        raise NotImplementedError("Phase 0 stub — implement in Phase 3")

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError("Phase 0 stub — implement in Phase 3")
