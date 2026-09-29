"""Report training progress to the terminal and Weights & Biases."""

import sys
from contextlib import ExitStack
from pathlib import Path
from types import TracebackType

from tqdm import tqdm

import wandb
from ventris.training_results import StepResult, ValidationResult

_PROGRESS_BAR_FORMAT = (
    "{desc:<11} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


class TrainingReporter:
    """Manage progress bars and optional experiment tracking for one run.

    ``step_completed`` reports results and prepares metrics; ``checkpoint_saved``
    adds checkpoint duration; ``finish_step`` logs the combined metrics. Initial
    validation is logged immediately, before its checkpoint is saved.
    """

    def __init__(
        self,
        *,
        total_steps: int,
        start_step: int,
        accumulation_steps: int,
        validation_interval_steps: int,
        validation_tokens: int,
        tokens_per_step: int,
        wandb_project: str | None,
        run_id: str,
        config: dict[str, object],
    ) -> None:
        self.total_steps = total_steps
        self.initial_step = start_step
        self.accumulation_steps = accumulation_steps
        self.validation_interval_steps = validation_interval_steps
        self.validation_tokens = validation_tokens
        self.tokens_per_step = tokens_per_step
        self.wandb_project = wandb_project
        self.run_id = run_id
        self.config = config
        self.next_checkpoint = min(
            (start_step // validation_interval_steps + 1) * validation_interval_steps,
            total_steps,
        )
        self._stack = ExitStack()
        self._tracking_run = None
        self._pending_step: int | None = None
        self._pending_metrics: dict[str, float | str] | None = None
        self._validation_tokens_completed = 0

    def __enter__(self) -> "TrainingReporter":
        self._start_tracking()
        self._open_progress_bars()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stack.__exit__(exception_type, exception, traceback)

    def start_step(self) -> None:
        self._device_batch_progress.reset(total=self.accumulation_steps)

    def device_batch_completed(self) -> None:
        self._device_batch_progress.update()

    def start_validation(self) -> None:
        """Begin progress reporting for a validation pass."""
        self._validation_tokens_completed = 0
        self._validation_progress.reset(total=self.validation_tokens)

    def validation_batch_completed(self, tokens: int) -> None:
        """Record tokens measured by one validation device batch."""
        remaining = self.validation_tokens - self._validation_tokens_completed
        completed = min(tokens, remaining)
        self._validation_tokens_completed += completed
        self._validation_progress.update(completed)

    def validation_completed(
        self,
        *,
        completed: int,
        validation: ValidationResult,
        training_seconds_elapsed: float,
    ) -> None:
        """Report validation performed before the first optimizer step."""
        self._write_block(
            f"{completed:6d}/{self.total_steps} | validation loss: {validation.loss:.4f} | "
            f"validation time: {validation.elapsed:6.2f}s | "
            f"throughput: {validation.tokens / validation.elapsed:7,.0f} tokens/s"
        )
        metrics: dict[str, float | str] = {
            "training/tokens_processed": completed * self.tokens_per_step,
            "training/seconds_elapsed": training_seconds_elapsed,
        }
        metrics.update(_validation_metrics(validation))
        self._write_samples(validation.samples)
        if self._tracking_run is not None:
            self._tracking_run.log(metrics, step=completed)

    def step_completed(
        self,
        *,
        step: StepResult,
        training_seconds_elapsed: float,
        validation: ValidationResult | None = None,
    ) -> None:
        """Report optimizer results and any validation performed at this step."""
        self._overall_progress.set_postfix(
            loss=f"{step.loss:.4f}", learning_rate=f"{step.learning_rate:.2e}"
        )
        message = (
            f"{step.completed:6d}/{self.total_steps} | loss: {step.loss:7.4f} | "
            f"learning rate: {step.learning_rate:8.2e} | elapsed time: {step.elapsed:6.2f}s | "
            f"throughput: {self.tokens_per_step / step.elapsed:7,.0f} tokens/s | "
            f"gradient norm: {step.gradient_norm:.4f}"
        )
        if step.gradient_clipped:
            message += " (clipped)"
        metrics = _training_metrics(step, self.tokens_per_step, training_seconds_elapsed)
        if validation is None:
            self._write_row(message)
        else:
            message += f" | validation loss: {validation.loss:.4f}"
            metrics.update(_validation_metrics(validation))
            self._write_block(message)
        self._advance_progress(step)

        self._pending_step = step.completed
        self._pending_metrics = metrics
        if validation is not None:
            self._write_samples(validation.samples)

    def checkpoint_saved(self, path: Path, *, elapsed: float) -> None:
        """Report a checkpoint immediately after it reaches disk."""
        if self._pending_metrics is not None:
            self._pending_metrics["checkpoint/seconds_per_checkpoint"] = elapsed
        self._write_block(f"Saved checkpoint in {elapsed:.2f}s: {path}")

    def best_checkpoint_retained(self, path: Path, *, loss: float, step: int) -> None:
        """Report a newly retained best checkpoint."""
        self._write_block(
            f"Retained best checkpoint at step {step} with validation loss {loss:.4f}: {path}"
        )

    def milestone_checkpoint_retained(self, path: Path, *, step: int) -> None:
        """Report a retained optimizer-step milestone."""
        self._write_block(f"Retained milestone checkpoint at step {step}: {path}")

    def finish_step(self) -> None:
        """Commit all metrics collected for the current optimizer step."""
        assert self._pending_step is not None
        assert self._pending_metrics is not None
        if self._tracking_run is not None:
            self._tracking_run.log(self._pending_metrics, step=self._pending_step)
        self._pending_step = None
        self._pending_metrics = None

    def _start_tracking(self) -> None:
        if self.wandb_project is not None:
            self._tracking_run = self._stack.enter_context(
                wandb.init(
                    project=self.wandb_project,
                    id=self.run_id,
                    resume="allow",
                    config=self.config,
                    save_code=True,
                )
            )
            self._tracking_run.define_metric("training/loss", summary="last")
            self._tracking_run.define_metric("training/gradient_norm", summary="max")
            self._tracking_run.define_metric("training/gradient_clipped", summary="mean")
            self._tracking_run.define_metric("training/seconds_elapsed", summary="last")
            self._tracking_run.define_metric("training/seconds_per_step", summary="mean")
            self._tracking_run.define_metric("training/tokens_per_second", summary="mean")
            self._tracking_run.define_metric(
                "checkpoint/seconds_per_checkpoint",
                summary="mean",
            )
            self._tracking_run.define_metric("validation/loss", summary="min,last")
            self._tracking_run.define_metric(
                "validation/seconds_per_validation",
                summary="mean",
            )
            self._tracking_run.define_metric(
                "validation/tokens_per_second",
                summary="mean",
            )

    def _open_progress_bars(self) -> None:
        self._overall_progress = self._stack.enter_context(
            tqdm(
                total=self.total_steps,
                initial=self.initial_step,
                desc="Overall:",
                unit="step",
                position=0,
                bar_format=_PROGRESS_BAR_FORMAT,
                dynamic_ncols=True,
            )
        )
        self._checkpoint_progress = self._stack.enter_context(
            tqdm(
                total=self.next_checkpoint - self.initial_step,
                desc="Checkpoint:",
                unit="step",
                position=1,
                leave=False,
                bar_format=_PROGRESS_BAR_FORMAT,
                dynamic_ncols=True,
            )
        )
        self._device_batch_progress = self._stack.enter_context(
            tqdm(
                total=self.accumulation_steps,
                desc="Train step:",
                unit="device-batch",
                position=2,
                leave=False,
                bar_format=_PROGRESS_BAR_FORMAT,
                dynamic_ncols=True,
            )
        )
        self._validation_progress = self._stack.enter_context(
            tqdm(
                total=self.validation_tokens,
                desc="Validation:",
                unit="token",
                unit_scale=True,
                position=3,
                leave=False,
                bar_format=_PROGRESS_BAR_FORMAT,
                dynamic_ncols=True,
            )
        )

    def _advance_progress(self, step: StepResult) -> None:
        self._overall_progress.update()
        self._checkpoint_progress.update()

        if step.completed == self.next_checkpoint and step.completed < self.total_steps:
            self.next_checkpoint = min(
                step.completed + self.validation_interval_steps,
                self.total_steps,
            )
            self._checkpoint_progress.reset(total=self.next_checkpoint - step.completed)

    def _write_samples(self, samples: list[tuple[str, str]]) -> None:
        if samples:
            self._write_block(_format_prompt_samples(samples, color_prompt=sys.stdout.isatty()))

    def _write_row(self, message: str) -> None:
        """Write one row in a sequence of related results."""
        message = message.strip("\n")
        self._overall_progress.write(message)

    def _write_block(self, message: str) -> None:
        """Write a standalone report block followed by one blank line."""
        message = message.strip("\n")
        self._overall_progress.write(f"{message}\n")


def _training_metrics(
    step: StepResult, tokens_per_step: int, training_seconds_elapsed: float
) -> dict[str, float | str]:
    """Build the optimizer-step metrics recorded by experiment tracking."""
    return {
        "training/loss": step.loss,
        "training/learning_rate": step.learning_rate,
        "training/gradient_norm": step.gradient_norm,
        "training/gradient_clipped": float(step.gradient_clipped),
        "training/tokens_processed": step.completed * tokens_per_step,
        "training/seconds_elapsed": training_seconds_elapsed,
        "training/seconds_per_step": step.elapsed,
        "training/tokens_per_second": tokens_per_step / step.elapsed,
    }


def _validation_metrics(validation: ValidationResult) -> dict[str, float | str]:
    """Build the validation metrics used both before and during training."""
    metrics: dict[str, float | str] = {
        "validation/loss": validation.loss,
        "validation/seconds_per_validation": validation.elapsed,
        "validation/tokens_per_second": validation.tokens / validation.elapsed,
    }
    metrics.update(
        {
            f"sample_{index}": prompt + continuation
            for index, (prompt, continuation) in enumerate(validation.samples, start=1)
        }
    )
    return metrics


def _format_prompt_samples(samples: list[tuple[str, str]], *, color_prompt: bool = False) -> str:
    """Format prompt continuations as readable, separated blocks."""
    divider = "-" * 70
    blocks = []
    for index, (prompt, continuation) in enumerate(samples, start=1):
        heading = f"{'-' * 30} Sample {index} {'-' * 30}"
        if color_prompt:
            prompt = f"\033[36m{prompt}\033[0m"
        blocks.append(f"{heading}\n\n{prompt}{continuation}")
    formatted = "\n\n".join(blocks)
    return f"{formatted}\n\n{divider}"
