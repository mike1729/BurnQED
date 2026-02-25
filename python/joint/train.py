"""Main training loop for joint LLM+EBM training.

Orchestrates the full training pipeline:
1. Load base model + LoRA
2. Attach GoalConditionedEnergyHead
3. Alternate SFT and contrastive batches
4. Monitor embedding health (separation probe, temperature)
5. Save checkpoints (LoRA + EBM head separately)

Gotcha #4: Tokenizer padding_side="right". Verify last-token indexing.
Gotcha #15: Cap NUM_WORKERS = min(16, cpu_count * 0.75).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for joint training."""

    # Model
    model_name: str = "deepseek-ai/DeepSeek-Prover-V2-7B"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # EBM head
    d_encoder: int = 4096
    ebm_dropout: float = 0.15
    ebm_n_power_iterations: int = 5

    # Training
    total_steps: int = 10_000
    batch_size_sft: int = 4
    batch_size_contrastive: int = 32
    lr_lora: float = 2e-5
    lr_ebm: float = 3e-4
    warmup_steps: int = 500
    sft_weight: float = 1.0
    contrastive_weight: float = 0.1
    max_seq_len: int = 2048
    gradient_accumulation_steps: int = 4

    # Data
    sft_data_path: Path = field(default_factory=lambda: Path("data/traced/sft_train.jsonl"))
    contrastive_data_path: Path = field(default_factory=lambda: Path("iterations/iter_0/trajectories"))

    # Monitoring
    log_interval: int = 50
    save_interval: int = 1000
    probe_interval: int = 500
    temperature_abort: bool = True

    # Output
    output_dir: Path = field(default_factory=lambda: Path("iterations/iter_1/model"))
    ebm_output_dir: Path = field(default_factory=lambda: Path("iterations/iter_1/ebm"))

    # Wandb
    wandb_project: Optional[str] = "burn-qed-v2"
    wandb_run_name: Optional[str] = None


def train(config: TrainingConfig) -> None:
    """Run joint LLM+EBM training.

    Steps:
    1. Initialize JointProver (base model + LoRA + EBM head)
    2. Create JointDataset (SFT + contrastive streams)
    3. Training loop:
       a. SFT forward → cross-entropy loss
       b. Contrastive forward → InfoNCE loss
       c. Combined backward (shared LoRA gradients)
       d. Monitor: temperature, separation probe, EBM metrics
    4. Save final checkpoint (LoRA adapter + EBM head)

    Args:
        config: Training configuration.
    """
    raise NotImplementedError("Phase 0 stub — implement in Phase 3")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Joint LLM+EBM training")
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    args = parser.parse_args()

    # TODO: Load config from file if provided, else use defaults
    cfg = TrainingConfig()
    train(cfg)
