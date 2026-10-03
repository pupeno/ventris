"""Own the mutable state, reporting, and checkpoints of a training run."""

import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType

import torch

from ventris.checkpoint import load_checkpoint, save_checkpoint
from ventris.common import RunConfig, TrainingConfig
from ventris.models import create_model
from ventris.models.vanilla import Ventris
from ventris.reporting import TrainingReporter
from ventris.training_results import StepResult, ValidationResult


@dataclass
class TrainingState:
    """The model and mutable progress needed to continue training."""

    checkpoint_dir: Path
    model: Ventris
    optimizer: torch.optim.AdamW
    completed_steps: int = 0
    training_seconds_elapsed: float = 0.0
    best_validation_loss: float | None = None
    best_validation_step: int | None = None

    @classmethod
    def initialize(
        cls,
        training_conf: TrainingConfig,
        *,
        checkpoint_dir: Path,
        resume: Path | None,
        target: torch.device,
    ) -> "TrainingState":
        """Create a fresh state or restore one from a checkpoint."""
        if resume is None:
            model = create_model()
            model.to(target)  # pyright: ignore[reportArgumentType]
            optimizer = build_optimizer(model, training_conf.peak_learning_rate)
            state = cls(checkpoint_dir, model, optimizer)
        else:
            training_state, model = load_checkpoint(resume, target, training_conf)
            optimizer = build_optimizer(model, training_conf.peak_learning_rate)
            optimizer.load_state_dict(training_state["optimizer"])
            saved_best_loss = training_state.get("best_validation_loss")
            saved_best_step = training_state.get("best_validation_step")
            state = cls(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                completed_steps=int(training_state["step"]),
                training_seconds_elapsed=float(training_state.get("training_seconds_elapsed", 0.0)),
                best_validation_loss=(None if saved_best_loss is None else float(saved_best_loss)),
                best_validation_step=None if saved_best_step is None else int(saved_best_step),
            )

        if state.completed_steps >= training_conf.steps:
            raise ValueError("the checkpoint already contains the requested final step")
        return state


class TrainingRun:
    """Record training results and own the run's reporting and checkpoints."""

    def __init__(
        self,
        state: TrainingState,
        training_conf: TrainingConfig,
        run_conf: RunConfig,
        *,
        accumulation_steps: int,
        world_size: int,
        is_primary: bool,
        tokenizer_dir: Path,
    ) -> None:
        self.state = state
        self.training_conf = training_conf
        self.run_conf = run_conf
        self.tokenizer_dir = tokenizer_dir
        self.is_primary = is_primary
        self.latest_checkpoint = state.checkpoint_dir / "latest"
        self.best_checkpoint = state.checkpoint_dir / "best"
        self._reporter = None
        if is_primary:
            tokens_per_step = (
                training_conf.effective_batch_size * state.model.config.max_position_embeddings
            )
            self._reporter = TrainingReporter(
                total_steps=training_conf.steps,
                start_step=state.completed_steps,
                accumulation_steps=accumulation_steps,
                validation_interval_steps=run_conf.validation_interval_steps,
                validation_tokens=run_conf.validation_tokens,
                tokens_per_step=tokens_per_step,
                wandb_project=run_conf.wandb_project,
                run_id=state.checkpoint_dir.name,
                config={
                    **state.model.config.shape_dict(),
                    **asdict(training_conf),
                    **asdict(run_conf),
                    "world_size": world_size,
                },
            )

    def __enter__(self) -> "TrainingRun":
        if self._reporter is not None:
            self._reporter.__enter__()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._reporter is not None:
            self._reporter.__exit__(exception_type, exception, traceback)

    def start_step(self) -> None:
        """Begin progress reporting for an optimizer step."""
        if self._reporter is not None:
            self._reporter.start_step()

    def device_batch_completed(self) -> None:
        """Record one device batch within the current optimizer step."""
        if self._reporter is not None:
            self._reporter.device_batch_completed()

    def start_validation(self) -> None:
        """Begin progress reporting for a validation pass."""
        if self._reporter is not None:
            self._reporter.start_validation()

    def validation_batch_completed(self, tokens: int) -> None:
        """Record tokens measured by one validation device batch."""
        if self._reporter is not None:
            self._reporter.validation_batch_completed(tokens)

    def initial_validation_completed(self, validation: ValidationResult | None) -> None:
        """Record validation of the initial or resumed state."""
        if not self.is_primary:
            return
        assert self._reporter is not None
        assert validation is not None
        self._reporter.validation_completed(
            completed=self.state.completed_steps,
            validation=validation,
            training_seconds_elapsed=self.state.training_seconds_elapsed,
        )
        self._save_validation_checkpoint(validation)

    def step_completed(
        self,
        step: StepResult,
        validation: ValidationResult | None,
    ) -> None:
        """Advance run state and record the completed optimizer step."""
        expected_step = self.state.completed_steps + 1
        if step.completed != expected_step:
            raise ValueError(f"expected completed step {expected_step}, got {step.completed}")
        self.state.completed_steps = step.completed
        self.state.training_seconds_elapsed += step.elapsed

        if not self.is_primary:
            return
        assert self._reporter is not None
        self._reporter.step_completed(
            step=step,
            validation=validation,
            training_seconds_elapsed=self.state.training_seconds_elapsed,
        )
        if validation is not None:
            self._save_validation_checkpoint(validation)
        milestone_interval_steps = (
            self.run_conf.validation_interval_steps * self.run_conf.milestone_interval_checkpoints
        )
        if validation is not None and step.completed % milestone_interval_steps == 0:
            self._save_milestone_checkpoint()
        self._reporter.finish_step()

    def _save_milestone_checkpoint(self) -> None:
        """Retain a scheduled validation checkpoint with its complete training state."""
        path = self.state.checkpoint_dir / f"step-{self.state.completed_steps:06d}"
        if path.exists():
            raise FileExistsError(f"milestone checkpoint already exists: {path}")
        shutil.copytree(self.latest_checkpoint, path)
        assert self._reporter is not None
        self._reporter.milestone_checkpoint_retained(path, step=self.state.completed_steps)

    def _save_validation_checkpoint(self, validation: ValidationResult) -> None:
        """Save the current state and retain it separately when it is the best."""
        is_best = (
            self.state.best_validation_loss is None
            or validation.loss < self.state.best_validation_loss
        )
        if is_best:
            self.state.best_validation_loss = validation.loss
            self.state.best_validation_step = self.state.completed_steps

        checkpoint_started = time.perf_counter()
        save_checkpoint(
            self.latest_checkpoint,
            self.state.model,
            self.state.optimizer,
            self.state.completed_steps,
            training_config=self.training_conf,
            run_config=self.run_conf,
            tokenizer_dir=self.tokenizer_dir,
            training_seconds_elapsed=self.state.training_seconds_elapsed,
            best_validation_loss=self.state.best_validation_loss,
            best_validation_step=self.state.best_validation_step,
        )
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        assert self._reporter is not None
        self._reporter.checkpoint_saved(self.latest_checkpoint, elapsed=checkpoint_seconds)

        if is_best:
            _retain_best_checkpoint(self.latest_checkpoint, self.best_checkpoint)
            self._reporter.best_checkpoint_retained(
                self.best_checkpoint,
                loss=validation.loss,
                step=self.state.completed_steps,
            )


def build_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.AdamW:
    """Build the optimizer used for both fresh and resumed training."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        fused=next(model.parameters()).is_cuda,
    )


def _retain_best_checkpoint(checkpoint_path: Path, best_checkpoint_path: Path) -> None:
    """Keep a copy of the latest checkpoint when it has the best validation loss."""
    if best_checkpoint_path.exists():
        shutil.rmtree(best_checkpoint_path)
    shutil.copytree(checkpoint_path, best_checkpoint_path)
