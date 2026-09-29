"""Measurements passed between training, run state, and reporting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StepResult:
    """Measurements produced by one completed optimizer step."""

    completed: int
    loss: float
    learning_rate: float
    elapsed: float
    gradient_norm: float
    gradient_clipped: bool


@dataclass(frozen=True)
class ValidationResult:
    """Measurements and samples produced by one validation pass."""

    loss: float
    elapsed: float
    tokens: int
    samples: list[tuple[str, str]]
