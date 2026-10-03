"""Persist model weights and the state needed to resume training."""

from dataclasses import asdict
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from ventris.common import RunConfig, TrainingConfig
from ventris.models import Model, load_model

TRAINING_STATE_FILE = "training_state.pt"


def load_checkpoint(
    path: Path,
    target: torch.device,
    expected_training_config: TrainingConfig,
) -> tuple[dict, Model]:
    """Load a checkpoint and verify its training configuration."""
    if not path.is_dir():
        raise FileNotFoundError(
            f"checkpoint not found at {path}; run scripts/train.py or choose an existing checkpoint"
        )
    model = load_model(path)
    model.to(target)  # pyright: ignore[reportArgumentType]
    training_state = torch.load(path / TRAINING_STATE_FILE, map_location=target)
    if training_state.get("training_config") != asdict(expected_training_config):
        raise ValueError("training config does not match checkpoint")
    return training_state, model


def save_checkpoint(
    path: Path,
    model: Model,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    training_config: TrainingConfig,
    run_config: RunConfig,
    tokenizer_dir: Path,
    training_seconds_elapsed: float,
    best_validation_loss: float | None,
    best_validation_step: int | None,
) -> Path:
    """Save model, tokenizer, and training state in one checkpoint directory."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True)
    tokenizer.model_max_length = model.config.max_position_embeddings
    tokenizer.save_pretrained(path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "training_config": asdict(training_config),
            "run_config": asdict(run_config),
            "step": step,
            "training_seconds_elapsed": training_seconds_elapsed,
            "best_validation_loss": best_validation_loss,
            "best_validation_step": best_validation_step,
        },
        path / TRAINING_STATE_FILE,
    )
    return path
