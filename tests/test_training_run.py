from dataclasses import asdict

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

import ventris.training_run as training_run_module
from tests.helpers import tiny_model
from ventris.checkpoint import TRAINING_STATE_FILE, save_checkpoint
from ventris.common import RunConfig, TrainingConfig
from ventris.models.vanilla import Ventris
from ventris.training_run import (
    StepResult,
    TrainingRun,
    TrainingState,
    ValidationResult,
    build_optimizer,
)


class QuietReporter:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def start_step(self):
        pass

    def device_batch_completed(self):
        pass

    def start_validation(self):
        pass

    def validation_batch_completed(self, tokens):
        pass

    def validation_completed(self, **kwargs):
        pass

    def step_completed(self, **kwargs):
        pass

    def checkpoint_saved(self, path, *, elapsed):
        pass

    def best_checkpoint_retained(self, path, *, loss, step):
        pass

    def milestone_checkpoint_retained(self, path, *, step):
        pass

    def finish_step(self):
        pass


def tiny_tokenizer(path):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer).save_pretrained(path)
    return path


def short_training(steps: int) -> TrainingConfig:
    return TrainingConfig(
        steps=steps,
        effective_batch_size=2,
        warmup_steps=1,
    )


def short_run() -> RunConfig:
    return RunConfig(
        device_batch_size=2,
        validation_interval_steps=2,
        validation_tokens=16,
        wandb_project=None,
    )


def step(completed: int) -> StepResult:
    return StepResult(
        completed=completed,
        loss=1.0,
        learning_rate=1e-3,
        elapsed=1.0,
        gradient_norm=0.5,
        gradient_clipped=False,
    )


def validation(loss: float, *, samples: bool = False) -> ValidationResult:
    return ValidationResult(
        loss=loss,
        elapsed=0.25,
        tokens=16,
        samples=[("prompt", " continuation")] if samples else [],
    )


@pytest.fixture
def run_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(training_run_module, "TrainingReporter", QuietReporter)
    tokenizer_dir = tiny_tokenizer(tmp_path / "tokenizer")

    def create(state, training, run_config=None):
        return TrainingRun(
            state,
            training,
            run_config or short_run(),
            accumulation_steps=1,
            world_size=1,
            is_primary=True,
            tokenizer_dir=tokenizer_dir,
        )

    return create


def test_validation_retains_lowest_loss_as_best_checkpoint(tmp_path, run_factory):
    training = short_training(2)
    model = tiny_model()
    state = TrainingState(
        checkpoint_dir=tmp_path,
        model=model,
        optimizer=build_optimizer(model, training.peak_learning_rate),
    )
    run = run_factory(state, training)

    with run:
        run.initial_validation_completed(validation(4.0))
        run.step_completed(step(1), validation(2.0))
        run.step_completed(step(2), validation(3.0))

    latest_state = torch.load(run.latest_checkpoint / TRAINING_STATE_FILE)
    best_state = torch.load(run.best_checkpoint / TRAINING_STATE_FILE)
    assert latest_state["step"] == 2
    assert best_state["step"] == 1
    assert latest_state["best_validation_loss"] == 2.0
    assert latest_state["best_validation_step"] == 1
    assert best_state["best_validation_loss"] == 2.0
    assert best_state["best_validation_step"] == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ["best", "latest", "tokenizer"]


def test_milestones_align_with_every_nth_validation_checkpoint(tmp_path, run_factory):
    training = short_training(9)
    run_config = RunConfig(
        device_batch_size=2,
        validation_interval_steps=2,
        milestone_interval_checkpoints=2,
        validation_tokens=16,
        wandb_project=None,
    )
    model = tiny_model()
    state = TrainingState(
        checkpoint_dir=tmp_path,
        model=model,
        optimizer=build_optimizer(model, training.peak_learning_rate),
    )
    run = run_factory(state, training, run_config)

    with run:
        run.initial_validation_completed(validation(4.0))
        for completed in range(1, 10):
            with torch.no_grad():
                next(model.parameters()).fill_(completed)
            measured = (
                validation(4.0 - completed / 10) if completed % 2 == 0 or completed == 9 else None
            )
            run.step_completed(step(completed), measured)

    for completed in (4, 8):
        milestone = tmp_path / f"step-{completed:06d}"
        saved = torch.load(milestone / TRAINING_STATE_FILE)
        assert saved["step"] == completed
        assert saved["run_config"] == asdict(run_config)
        assert "optimizer" in saved
        restored = Ventris.from_pretrained(milestone, local_files_only=True)
        assert torch.all(next(restored.parameters()) == completed)
    assert torch.load(run.latest_checkpoint / TRAINING_STATE_FILE)["step"] == 9
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "best",
        "latest",
        "step-000004",
        "step-000008",
        "tokenizer",
    ]


def test_best_validation_state_persists_across_resume(tmp_path, run_factory):
    training = short_training(3)
    model = tiny_model()
    checkpoint_options = {
        "training_config": training,
        "run_config": short_run(),
        "tokenizer_dir": tiny_tokenizer(tmp_path / "tokenizer"),
        "training_seconds_elapsed": 0.0,
        "best_validation_loss": 0.5,
        "best_validation_step": 1,
    }
    latest = save_checkpoint(
        tmp_path / "latest",
        model,
        build_optimizer(model, training.peak_learning_rate),
        step=1,
        **checkpoint_options,
    )
    save_checkpoint(
        tmp_path / "best",
        model,
        build_optimizer(model, training.peak_learning_rate),
        step=1,
        **checkpoint_options,
    )
    state = TrainingState.initialize(
        training,
        checkpoint_dir=tmp_path,
        resume=latest,
        target=torch.device("cpu"),
    )
    run = run_factory(state, training)

    with run:
        run.initial_validation_completed(validation(0.6))
        run.step_completed(step(2), validation(0.4))
        run.step_completed(step(3), validation(0.7))

    latest_state = torch.load(run.latest_checkpoint / TRAINING_STATE_FILE)
    best_state = torch.load(run.best_checkpoint / TRAINING_STATE_FILE)
    assert latest_state["step"] == 3
    assert best_state["step"] == 2
    assert latest_state["best_validation_loss"] == 0.4
    assert latest_state["best_validation_step"] == 2
    assert best_state["best_validation_loss"] == 0.4
    assert best_state["best_validation_step"] == 2


def test_validation_and_checkpoint_phases_are_reported_as_each_completes(
    tmp_path,
    monkeypatch,
    run_factory,
):
    events = []

    class Reporter(QuietReporter):
        def validation_completed(self, *, validation, **kwargs):
            events.append("validation reported")
            if validation.samples:
                events.append("samples reported")

        def step_completed(self, *, validation, **kwargs):
            events.append("step reported")
            if validation is not None and validation.samples:
                events.append("samples reported")

        def checkpoint_saved(self, path, *, elapsed):
            events.append("checkpoint reported")

        def best_checkpoint_retained(self, path, *, loss, step):
            events.append("best checkpoint reported")

    def save(*args, **kwargs):
        events.append("checkpoint saved")

    def retain_best(*args):
        events.append("best checkpoint retained")

    monkeypatch.setattr(training_run_module, "TrainingReporter", Reporter)
    monkeypatch.setattr(training_run_module, "save_checkpoint", save)
    monkeypatch.setattr(training_run_module, "_retain_best_checkpoint", retain_best)
    training = short_training(1)
    model = tiny_model()
    state = TrainingState(
        checkpoint_dir=tmp_path,
        model=model,
        optimizer=build_optimizer(model, training.peak_learning_rate),
    )
    run = run_factory(state, training)

    with run:
        run.initial_validation_completed(validation(1.0, samples=True))
        run.step_completed(step(1), validation(1.0, samples=True))

    assert events == [
        "validation reported",
        "samples reported",
        "checkpoint saved",
        "checkpoint reported",
        "best checkpoint retained",
        "best checkpoint reported",
        "step reported",
        "samples reported",
        "checkpoint saved",
        "checkpoint reported",
    ]


def test_current_run_config_is_saved_after_resume(tmp_path, run_factory):
    training = short_training(3)
    saved_run = RunConfig(
        device_batch_size=1,
        validation_interval_steps=1,
        validation_tokens=16,
        compile_model=True,
        wandb_project="old-project",
    )
    model = tiny_model()
    checkpoint = save_checkpoint(
        tmp_path / "latest",
        model,
        build_optimizer(model, training.peak_learning_rate),
        step=1,
        training_config=training,
        run_config=saved_run,
        tokenizer_dir=tiny_tokenizer(tmp_path / "tokenizer"),
        training_seconds_elapsed=0.0,
        best_validation_loss=1.0,
        best_validation_step=1,
    )
    state = TrainingState.initialize(
        training,
        checkpoint_dir=tmp_path,
        resume=checkpoint,
        target=torch.device("cpu"),
    )
    run = run_factory(state, training)

    with run:
        run.initial_validation_completed(validation(1.0))

    assert torch.load(run.latest_checkpoint / TRAINING_STATE_FILE)["run_config"] == asdict(
        short_run()
    )


def test_secondary_process_advances_state_without_reporting_or_saving(tmp_path, monkeypatch):
    def unexpected_reporter(**kwargs):
        pytest.fail("secondary process started reporting")

    def unexpected_save(*args, **kwargs):
        pytest.fail("secondary process saved a checkpoint")

    monkeypatch.setattr(training_run_module, "TrainingReporter", unexpected_reporter)
    monkeypatch.setattr(training_run_module, "save_checkpoint", unexpected_save)
    model = tiny_model()
    training = short_training(3)
    state = TrainingState(
        checkpoint_dir=tmp_path,
        model=model,
        optimizer=build_optimizer(model, training.peak_learning_rate),
    )

    with TrainingRun(
        state,
        training,
        short_run(),
        accumulation_steps=1,
        world_size=2,
        is_primary=False,
        tokenizer_dir=tmp_path / "unused-tokenizer",
    ) as run:
        run.start_validation()
        run.validation_batch_completed(16)
        run.initial_validation_completed(None)
        for completed in (1, 2):
            run.start_step()
            run.device_batch_completed()
            run.step_completed(step(completed), None)

    assert state.completed_steps == 2
    assert state.training_seconds_elapsed == 2.0
    assert list(tmp_path.iterdir()) == []
