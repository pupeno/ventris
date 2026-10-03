from pathlib import Path

import pytest
import wandb

import ventris.reporting as reporting_module
from ventris.reporting import TrainingReporter
from ventris.training_results import StepResult, ValidationResult


def step_result(
    completed: int, *, gradient_norm: float = 0.75, gradient_clipped: bool = False
) -> StepResult:
    return StepResult(
        completed=completed,
        loss=1.0,
        learning_rate=1e-3,
        elapsed=1.0,
        gradient_norm=gradient_norm,
        gradient_clipped=gradient_clipped,
    )


def validation_result(*, samples: bool = False) -> ValidationResult:
    return ValidationResult(
        loss=0.5,
        elapsed=0.25,
        tokens=16,
        samples=[("Once upon a time", " there was a model.")] if samples else [],
    )


def progress_factory(progress_bars):
    class Progress:
        def __init__(self, **options):
            self.options = options
            self.updates = 0
            self.resets = []
            self.messages = []
            progress_bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def update(self, amount=1):
            self.updates += amount

        def reset(self, *, total):
            self.resets.append(total)

        def set_postfix(self, **values):
            return None

        def write(self, message):
            self.messages.append(message)

    return Progress


def test_reporter_tracks_overall_checkpoint_and_device_batch_progress(monkeypatch, capsys):
    progress_bars = []
    monkeypatch.setattr(reporting_module, "tqdm", progress_factory(progress_bars))
    checkpoint = Path("latest.pt")

    with TrainingReporter(
        total_steps=3,
        start_step=0,
        accumulation_steps=2,
        validation_interval_steps=2,
        validation_tokens=16,
        tokens_per_step=16,
        wandb_project=None,
        run_id="test-run",
        config={"model_type": "ventris-vanilla-v1", "parameter_count": 11_616},
    ) as reporter:
        reporter.start_validation()
        reporter.validation_batch_completed(8)
        reporter.validation_batch_completed(8)
        for completed in range(1, 4):
            reporter.start_step()
            reporter.device_batch_completed()
            reporter.device_batch_completed()
            should_checkpoint = completed >= 2
            reporter.step_completed(
                step=step_result(completed),
                training_seconds_elapsed=float(completed),
                validation=validation_result() if should_checkpoint else None,
            )
            if should_checkpoint:
                reporter.checkpoint_saved(checkpoint, elapsed=0.5)
            reporter.finish_step()

    assert [progress.options["desc"] for progress in progress_bars] == [
        "Overall:",
        "Checkpoint:",
        "Train step:",
        "Validation:",
    ]
    assert [progress.options["position"] for progress in progress_bars] == [0, 1, 2, 3]
    assert progress_bars[3].options["unit_scale"] is True
    assert [progress.updates for progress in progress_bars] == [3, 3, 6, 16]
    assert progress_bars[1].resets == [1]
    assert progress_bars[2].resets == [2, 2, 2]
    assert progress_bars[3].resets == [16]
    assert len(progress_bars[0].messages) == 5
    assert progress_bars[0].messages[2] == f"Saved checkpoint in 0.50s: {checkpoint}\n"
    assert progress_bars[0].messages[4] == f"Saved checkpoint in 0.50s: {checkpoint}\n"
    assert not progress_bars[0].messages[0].endswith("\n")
    assert all(not message.startswith("\n") for message in progress_bars[0].messages)
    assert all(not message.endswith("\n\n") for message in progress_bars[0].messages)
    assert capsys.readouterr().out == "Model: ventris-vanilla-v1 | Parameters: 11,616\n"


def test_reporter_keeps_consecutive_training_results_on_adjacent_lines(monkeypatch):
    progress_bars = []
    monkeypatch.setattr(reporting_module, "tqdm", progress_factory(progress_bars))

    with TrainingReporter(
        total_steps=2,
        start_step=0,
        accumulation_steps=1,
        validation_interval_steps=2,
        validation_tokens=16,
        tokens_per_step=8,
        wandb_project=None,
        run_id="test-run",
        config={"model_type": "ventris-vanilla-v1", "parameter_count": 11_616},
    ) as reporter:
        for completed in range(1, 3):
            reporter.step_completed(
                step=step_result(completed),
                training_seconds_elapsed=float(completed),
            )
            reporter.finish_step()

    assert len(progress_bars[0].messages) == 2
    assert all(not message.endswith("\n") for message in progress_bars[0].messages)


class RecordingRun:
    def __init__(self):
        self.history = []
        self.metric_definitions = []
        self.exit_exception = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.exit_exception = exception

    def log(self, metrics, step):
        self.history.append((step, metrics))

    def define_metric(self, name, **options):
        self.metric_definitions.append((name, options))


@pytest.fixture
def tracked_reporter(monkeypatch):
    run = RecordingRun()
    init_options = {}

    def init_wandb(**options):
        init_options.update(options)
        return run

    progress_bars = []
    monkeypatch.setattr(reporting_module, "tqdm", progress_factory(progress_bars))
    monkeypatch.setattr(wandb, "init", init_wandb)

    reporter = TrainingReporter(
        total_steps=2,
        start_step=0,
        accumulation_steps=1,
        validation_interval_steps=1,
        validation_tokens=16,
        tokens_per_step=8,
        wandb_project="test-project",
        run_id="test-run",
        config={
            "wandb_project": "test-project",
            "model_type": "ventris-vanilla-v1",
            "parameter_count": 11_616,
        },
    )
    return reporter, run, init_options, progress_bars


def test_reporter_configures_wandb(tracked_reporter):
    reporter, run, init_options, _ = tracked_reporter

    with reporter:
        pass

    assert init_options["project"] == "test-project"
    assert init_options["config"]["wandb_project"] == "test-project"
    assert init_options["save_code"] is True
    assert run.metric_definitions == [
        ("training/loss", {"summary": "last"}),
        ("training/gradient_norm", {"summary": "max"}),
        ("training/gradient_clipped", {"summary": "mean"}),
        ("training/seconds_elapsed", {"summary": "last"}),
        ("training/seconds_per_step", {"summary": "mean"}),
        ("training/tokens_per_second", {"summary": "mean"}),
        ("checkpoint/seconds_per_checkpoint", {"summary": "mean"}),
        ("validation/loss", {"summary": "min,last"}),
        ("validation/seconds_per_validation", {"summary": "mean"}),
        ("validation/tokens_per_second", {"summary": "mean"}),
    ]


def test_reporter_sends_metrics_to_wandb(tracked_reporter):
    reporter, run, _, progress_bars = tracked_reporter

    with reporter:
        reporter.start_step()
        reporter.device_batch_completed()
        reporter.step_completed(
            step=step_result(1, gradient_norm=1.25, gradient_clipped=True),
            training_seconds_elapsed=1.0,
            validation=validation_result(samples=True),
        )
        checkpoint = Path("latest.pt")
        reporter.checkpoint_saved(checkpoint, elapsed=0.5)
        reporter.best_checkpoint_retained(checkpoint, loss=0.5, step=1)
        reporter.finish_step()

    assert run.history[0][0] == 1
    assert set(run.history[0][1]) == {
        "training/loss",
        "training/learning_rate",
        "training/gradient_norm",
        "training/gradient_clipped",
        "training/tokens_processed",
        "training/seconds_elapsed",
        "validation/loss",
        "validation/seconds_per_validation",
        "validation/tokens_per_second",
        "training/seconds_per_step",
        "training/tokens_per_second",
        "checkpoint/seconds_per_checkpoint",
        "sample_1",
    }
    assert run.history[0][1]["training/tokens_processed"] == 8
    assert run.history[0][1]["training/seconds_elapsed"] == 1.0
    assert run.history[0][1]["training/gradient_norm"] == 1.25
    assert run.history[0][1]["training/gradient_clipped"] == 1.0
    assert run.history[0][1]["training/tokens_per_second"] == 8
    assert run.history[0][1]["validation/tokens_per_second"] == 64
    assert run.history[0][1]["checkpoint/seconds_per_checkpoint"] == 0.5
    assert run.history[0][1]["sample_1"] == "Once upon a time there was a model."
    assert (
        f"Retained best checkpoint at step 1 with validation loss 0.5000: {checkpoint}\n"
        in progress_bars[0].messages
    )
    sample_report = next(message for message in progress_bars[0].messages if "Sample 1" in message)
    assert sample_report.startswith("------------------------------ Sample 1")
    assert "Once upon a time there was a model." in sample_report
    assert sample_report.endswith("-" * 70 + "\n")
    assert "gradient norm: 1.2500 (clipped)" in progress_bars[0].messages[0]


def test_reporter_logs_validation_before_first_optimizer_step(tracked_reporter):
    reporter, run, _, progress_bars = tracked_reporter

    with reporter:
        reporter.validation_completed(
            completed=0,
            validation=validation_result(samples=True),
            training_seconds_elapsed=0.0,
        )

    assert run.history == [
        (
            0,
            {
                "validation/loss": 0.5,
                "validation/seconds_per_validation": 0.25,
                "validation/tokens_per_second": 64,
                "training/tokens_processed": 0,
                "training/seconds_elapsed": 0.0,
                "sample_1": "Once upon a time there was a model.",
            },
        )
    ]
    assert "     0/2 | validation loss: 0.5000" in progress_bars[0].messages[0]
    assert "Once upon a time there was a model." in progress_bars[0].messages[1]


def test_reporter_marks_wandb_run_failed_without_discarding_logged_history(tracked_reporter):
    reporter, run, _, _ = tracked_reporter

    with pytest.raises(FloatingPointError, match="training loss is not finite"):
        with reporter:
            reporter.validation_completed(
                completed=0,
                validation=validation_result(),
                training_seconds_elapsed=0.0,
            )
            raise FloatingPointError("training loss is not finite: nan")

    assert isinstance(run.exit_exception, FloatingPointError)
    assert run.history[0][0] == 0
    assert run.history[0][1]["validation/loss"] == 0.5
