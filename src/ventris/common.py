"""Shared configuration and execution policies for Ventris."""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingConfig:
    """The choices that determine a pretraining run's model updates."""

    steps: int = 16_384
    effective_batch_size: int = 512
    warmup_steps: int = 715
    peak_learning_rate: float = 6e-4
    minimum_learning_rate: float = 6e-5
    gradient_clip_norm: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class RunConfig:
    """The trainer's execution, validation, checkpointing, and reporting choices."""

    device_batch_size: int = 4
    validation_interval_steps: int = 250
    milestone_interval_checkpoints: int = 4
    validation_tokens: int = 80 * 524_288
    compile_model: bool = False
    wandb_project: str | None = "ventris"


DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_RUN_CONFIG = RunConfig()


def default_device() -> torch.device:
    """Return the preferred device for model execution."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mixed_precision(target: torch.device):
    """Use native BF16 autocast on CUDA and full FP32 otherwise."""
    if target.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@contextmanager
def evaluation_mode(model: torch.nn.Module) -> Iterator[None]:
    """Run inference and restore the model's previous training mode afterward."""
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        model.train(was_training)
