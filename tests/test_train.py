import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel
from transformers import PreTrainedTokenizerFast

import ventris.data as data_module
import ventris.train as train_module
import ventris.training_run as training_run_module
from tests.helpers import tiny_model
from ventris.checkpoint import TRAINING_STATE_FILE, load_checkpoint, save_checkpoint
from ventris.common import RunConfig, TrainingConfig
from ventris.data import EOS_TEXT
from ventris.models import load_model
from ventris.train import (
    _learning_rate,
    _next_token_loss,
    _optimizer_step,
    _should_validate,
    _training_device,
)
from ventris.training_run import TrainingState, build_optimizer


class FixedLogitModel(torch.nn.Module):
    """Assign low loss to token 0 and high loss to token 1 when the label is 0."""

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        values = input_ids.float()
        logits = torch.stack((2 - 2 * values, 2 * values), dim=-1)
        return SimpleNamespace(logits=logits)


def tiny_data(training_sequences: int = 10, validation_sequences: int = 2) -> DatasetDict:
    def split(size: int) -> Dataset:
        sequences = [(torch.arange(9) + index) % 32 for index in range(size)]
        return Dataset.from_dict({"input_ids": sequences}).with_format("torch")

    return DatasetDict(
        {"train": split(training_sequences), "validation": split(validation_sequences)}
    )


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


def write_tiny_tokenizer(path):
    vocab = {f"token{i}": i for i in range(31)}
    vocab[EOS_TEXT] = 31
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="token0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token=EOS_TEXT).save_pretrained(path)


def _distributed_optimizer_step(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_directory: str,
    device_batches: list[dict[str, torch.Tensor]],
    architecture: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(0)
        training_run_module.create_model = tiny_model
        state = TrainingState.initialize(
            short_training(3),
            checkpoint_dir=Path(output_directory),
            resume=None,
            target=torch.device("cpu"),
            architecture=architecture,
        )
        model, optimizer = state.model, state.optimizer
        training_model = DistributedDataParallel(model, gradient_as_bucket_view=True)
        local_device_batches = device_batches[rank::world_size]

        loss, gradient_norm, gradient_clipped = _optimizer_step(
            training_model,
            optimizer,
            iter(local_device_batches),
            accumulation_steps=len(local_device_batches),
            learning_rate=1e-2,
            target=torch.device("cpu"),
        )

        torch.save(
            {
                "model": model.state_dict(),
                "loss": loss,
                "gradient_norm": gradient_norm,
                "gradient_clipped": gradient_clipped,
            },
            f"{output_directory}/rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _distributed_validation(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_directory: str,
    validation_sequences: int,
    device_batch_size: int,
    max_tokens: int,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(0)
        model = load_model(Path(output_directory) / "validation-model")
        model.train()

        def generate_samples(model):
            assert rank == 0
            return [("prompt", "sample")]

        train_module._generate_prompt_samples = generate_samples
        progress = []
        result = train_module._validate_model(
            model,
            tiny_data(validation_sequences=validation_sequences)["validation"],
            device_batch_size=device_batch_size,
            target=torch.device("cpu"),
            max_tokens=max_tokens,
            distributed=train_module.DistributedContext(rank, world_size),
            on_batch=progress.append,
        )
        torch.save(
            {
                "validation": None
                if result is None
                else {"loss": result.loss, "tokens": result.tokens, "samples": result.samples},
                "progress": progress,
                "training": model.training,
            },
            f"{output_directory}/validation-rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_learning_rate_boundaries_use_zero_based_steps():
    config = TrainingConfig(
        steps=6,
        effective_batch_size=2,
        warmup_steps=2,
        peak_learning_rate=1.0,
        minimum_learning_rate=0.0,
    )

    assert _learning_rate(0, config) == 0.5
    assert _learning_rate(1, config) == 1.0
    assert _learning_rate(2, config) == 1.0
    assert _learning_rate(config.steps - 1, config) == pytest.approx(
        0.5 * (1 + math.cos(3 * math.pi / 4))
    )
    assert _learning_rate(config.steps, config) == 0.0


@pytest.mark.parametrize(
    ("completed", "expected"),
    [(1, False), (2, False), (3, True), (4, False), (5, True)],
)
def test_should_validate_at_interval_and_final_step(completed, expected):
    assert _should_validate(completed, interval_steps=3, total_steps=5) is expected


def test_next_token_loss_matches_known_cross_entropy():
    loss = _next_token_loss(
        FixedLogitModel(),
        torch.tensor([[0, 1]]),
        torch.tensor([[0, 1]]),
    )

    assert loss.item() == pytest.approx(0.126928011)


def test_measure_validation_loss_restores_training_mode_after_error(monkeypatch):
    model = tiny_model()

    def fail(*args):
        raise RuntimeError("validation failed")

    monkeypatch.setattr(train_module, "_next_token_loss", fail)

    with pytest.raises(RuntimeError, match="validation failed"):
        train_module._measure_validation_loss(
            model,
            tiny_data()["validation"],
            device_batch_size=2,
            target=torch.device("cpu"),
        )

    assert model.training


def test_measure_validation_loss_stops_at_token_budget():
    _, tokens = train_module._measure_validation_loss(
        tiny_model(),
        tiny_data()["validation"],
        device_batch_size=1,
        target=torch.device("cpu"),
        max_tokens=8,
    )

    assert tokens == 8


@pytest.mark.parametrize(
    ("max_tokens", "expected_tokens", "expected_progress"),
    [(1, 2, [2]), (3, 4, [2, 2]), (5, 5, [2, 2, 1])],
)
def test_validation_budget_selects_complete_device_batches(
    max_tokens, expected_tokens, expected_progress
):
    validation = Dataset.from_dict(
        {"input_ids": [[0, 0], [1, 0], [0, 0], [1, 0], [0, 0]]}
    ).with_format("torch")
    progress = []
    model = FixedLogitModel()

    loss, tokens = train_module._measure_validation_loss(
        model,
        validation,
        device_batch_size=2,
        target=torch.device("cpu"),
        max_tokens=max_tokens,
        on_batch=progress.append,
    )
    prefix = validation[:expected_tokens]["input_ids"]
    expected_loss = _next_token_loss(model, prefix[:, :-1], prefix[:, 1:]).item()

    assert tokens == expected_tokens
    assert loss == pytest.approx(expected_loss)
    assert progress == expected_progress


def test_validation_rejects_a_budget_larger_than_the_split():
    model = tiny_model()

    with pytest.raises(ValueError, match="contains 16 tokens, fewer than the requested 17"):
        train_module._measure_validation_loss(
            model,
            tiny_data()["validation"],
            device_batch_size=2,
            target=torch.device("cpu"),
            max_tokens=17,
        )

    assert model.training


def test_measure_validation_loss_reports_each_completed_batch():
    completed_batches = []

    train_module._measure_validation_loss(
        tiny_model(),
        tiny_data()["validation"],
        device_batch_size=1,
        target=torch.device("cpu"),
        max_tokens=16,
        on_batch=completed_batches.append,
    )

    assert completed_batches == [8, 8]


def test_validation_loss_weights_a_short_final_device_batch_by_its_tokens():
    validation = Dataset.from_dict({"input_ids": [[0, 0], [1, 0], [0, 0]]}).with_format("torch")

    loss, tokens = train_module._measure_validation_loss(
        FixedLogitModel(),
        validation,
        device_batch_size=2,
        target=torch.device("cpu"),
    )

    assert tokens == 3
    assert loss == pytest.approx(0.793594678)


def test_distributed_validation_rejects_an_empty_split():
    empty = Dataset.from_dict({"input_ids": []}).with_format("torch")

    with pytest.raises(ValueError, match="validation split is empty"):
        train_module._measure_validation_loss(
            tiny_model(),
            empty,
            device_batch_size=1,
            target=torch.device("cpu"),
            distributed=train_module.DistributedContext(rank=0, world_size=2),
        )


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
@pytest.mark.parametrize(
    ("world_size", "validation_sequences", "device_batch_size", "max_tokens", "expected_tokens"),
    [(2, 5, 2, 33, 40), (3, 2, 1, 8, 8)],
)
def test_distributed_validation_matches_single_process(
    tmp_path,
    architecture,
    world_size,
    validation_sequences,
    device_batch_size,
    max_tokens,
    expected_tokens,
):
    torch.manual_seed(0)
    model = tiny_model(architecture)
    model.save_pretrained(tmp_path / "validation-model")
    reference_loss, reference_tokens = train_module._measure_validation_loss(
        model,
        tiny_data(validation_sequences=validation_sequences)["validation"],
        device_batch_size=device_batch_size,
        target=torch.device("cpu"),
        max_tokens=max_tokens,
    )

    spawn(
        _distributed_validation,
        args=(
            world_size,
            str(tmp_path / "validation-rendezvous"),
            str(tmp_path),
            validation_sequences,
            device_batch_size,
            max_tokens,
        ),
        nprocs=world_size,
    )

    assert reference_tokens == expected_tokens
    for rank in range(world_size):
        result = torch.load(tmp_path / f"validation-rank-{rank}.pt")
        assert result["training"]
        if rank == 0:
            assert result["validation"]["loss"] == pytest.approx(reference_loss, abs=1e-6)
            assert result["validation"]["tokens"] == expected_tokens
            assert result["validation"]["samples"] == [("prompt", "sample")]
            assert sum(result["progress"]) == expected_tokens
        else:
            assert result["validation"] is None


def test_prompt_samples_use_stable_prompts_and_sampling(monkeypatch):
    calls = []
    tokenizer = object()

    def generate(model, received_tokenizer, prompt, **options):
        calls.append((model, received_tokenizer, prompt, options))
        return " continuation"

    monkeypatch.setattr(train_module, "load_tokenizer", lambda: tokenizer)
    monkeypatch.setattr("ventris.generate.generate_from_model", generate)
    model = tiny_model()

    samples = train_module._generate_prompt_samples(model)

    assert [prompt for prompt, _ in samples] == list(train_module.PROMPT_TESTS)
    assert [continuation for _, continuation in samples] == [" continuation"] * 5
    assert [call[3] for call in calls] == [
        {
            "max_new_tokens": 80,
            "temperature": 0.8,
            "top_k": 50,
            "seed": index,
        }
        for index in range(5)
    ]
    assert all(call[0] is model and call[1] is tokenizer for call in calls)


def test_training_requires_cuda(monkeypatch):
    monkeypatch.setattr(train_module, "default_device", lambda: torch.device("cpu"))

    with pytest.raises(RuntimeError, match="CUDA GPU"):
        _training_device()


@pytest.mark.parametrize(
    ("training", "run", "message"),
    [
        (
            replace(short_training(3), effective_batch_size=0),
            short_run(),
            "effective_batch_size must be positive",
        ),
        (
            short_training(3),
            replace(short_run(), device_batch_size=0),
            "device_batch_size must be positive",
        ),
        (
            short_training(3),
            replace(short_run(), validation_interval_steps=0),
            "validation_interval_steps must be positive",
        ),
        (
            short_training(3),
            replace(short_run(), milestone_interval_checkpoints=0),
            "milestone_interval_checkpoints must be positive",
        ),
        (
            short_training(3),
            replace(short_run(), validation_tokens=0),
            "validation_tokens must be positive",
        ),
        (
            replace(short_training(3), minimum_learning_rate=-1e-4),
            short_run(),
            "0 <= minimum_learning_rate <= peak_learning_rate",
        ),
        (
            replace(short_training(3), minimum_learning_rate=1e-3),
            short_run(),
            "0 <= minimum_learning_rate <= peak_learning_rate",
        ),
        (
            replace(short_training(3), gradient_clip_norm=0),
            short_run(),
            "gradient_clip_norm must be positive",
        ),
    ],
)
def test_train_rejects_invalid_config(training, run, message):
    with pytest.raises(ValueError, match=message):
        train_module.train(training_conf=training, run_conf=run)


def test_device_batches_across_all_processes_must_divide_effective_batch():
    with pytest.raises(ValueError, match=r"3 \* 2 does not divide 8"):
        train_module._validate_run_config(
            replace(short_training(3), effective_batch_size=8),
            replace(short_run(), device_batch_size=3),
            world_size=2,
        )


def test_training_rejects_bfloat16_emulation(monkeypatch):
    support_checks = []
    monkeypatch.setattr(train_module, "default_device", lambda: torch.device("cuda"))

    def supports_bfloat16(*, including_emulation):
        support_checks.append(including_emulation)
        return False

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supports_bfloat16)

    with pytest.raises(RuntimeError, match="emulation is disabled"):
        _training_device()
    assert support_checks == [False]


def test_training_sequences_resume_without_repetition_across_ranks():
    training = TrainingConfig(steps=3, effective_batch_size=4, warmup_steps=1, seed=7)
    run = RunConfig(device_batch_size=1)
    sequences = Dataset.from_dict({"input_ids": [[index, index + 1] for index in range(12)]})
    sequences = sequences.with_format("torch")

    def order(
        distributed: train_module.DistributedContext, start_step: int
    ) -> tuple[list[int], int]:
        batches, accumulation_steps = train_module._prepare_training_batches(
            sequences,
            training,
            run,
            distributed,
            start_step,
            torch.device("cpu"),
        )
        return [int(batch["input_ids"][0, 0]) for batch in batches], accumulation_steps

    single, single_accumulation = order(train_module.DistributedContext(), 0)
    first_rank = train_module.DistributedContext(rank=0, world_size=2)
    second_rank = train_module.DistributedContext(rank=1, world_size=2)
    rank_zero, rank_accumulation = order(first_rank, 0)
    rank_one, _ = order(second_rank, 0)

    assert single_accumulation == 4
    assert rank_accumulation == 2
    for completed in range(training.steps):
        local_start = completed * rank_accumulation
        local_stop = local_start + rank_accumulation
        global_start = completed * single_accumulation
        global_stop = global_start + single_accumulation
        assert sorted(
            rank_zero[local_start:local_stop] + rank_one[local_start:local_stop]
        ) == sorted(single[global_start:global_stop])
    assert order(first_rank, 1)[0] == rank_zero[rank_accumulation:]
    assert order(second_rank, 1)[0] == rank_one[rank_accumulation:]


def test_repeated_optimizer_steps_reduce_loss():
    torch.manual_seed(0)
    model = tiny_model()
    optimizer = build_optimizer(model, 1e-2)
    data = tiny_data(1)["train"]
    sequence = data[0]["input_ids"].unsqueeze(0)
    batch = {"input_ids": sequence}
    initial = _next_token_loss(model, sequence[:, :-1], sequence[:, 1:]).item()

    result = None
    for _ in range(20):
        result = _optimizer_step(
            model,
            optimizer,
            iter([batch]),
            accumulation_steps=1,
            learning_rate=1e-2,
            target=torch.device("cpu"),
        )

    final = _next_token_loss(model, sequence[:, :-1], sequence[:, 1:]).item()
    assert result is not None
    loss, gradient_norm, gradient_clipped = result
    assert math.isfinite(loss)
    assert gradient_norm > 0
    assert isinstance(gradient_clipped, bool)
    assert final < initial


def test_optimizer_step_reports_when_gradients_are_clipped():
    model = tiny_model()
    optimizer = build_optimizer(model, learning_rate=1e-2)
    sequence = tiny_data(1)["train"][0]["input_ids"].unsqueeze(0)

    _, gradient_norm, gradient_clipped = _optimizer_step(
        model,
        optimizer,
        iter([{"input_ids": sequence}]),
        accumulation_steps=1,
        learning_rate=1e-2,
        target=torch.device("cpu"),
        gradient_clip_norm=1e-12,
    )

    assert gradient_norm > 1e-12
    assert gradient_clipped


@pytest.mark.parametrize("nonfinite_loss", [float("nan"), float("inf")])
def test_optimizer_step_rejects_nonfinite_training_loss_before_update(monkeypatch, nonfinite_loss):
    model = tiny_model()
    optimizer = build_optimizer(model, learning_rate=1e-2)
    optimizer_step = Mock()
    monkeypatch.setattr(optimizer, "step", optimizer_step)
    monkeypatch.setattr(
        train_module,
        "_next_token_loss",
        lambda *args: torch.tensor(nonfinite_loss, requires_grad=True),
    )
    sequence = tiny_data(1)["train"][0]["input_ids"].unsqueeze(0)

    with pytest.raises(FloatingPointError, match="training loss is not finite"):
        _optimizer_step(
            model,
            optimizer,
            iter([{"input_ids": sequence}]),
            accumulation_steps=1,
            learning_rate=1e-2,
            target=torch.device("cpu"),
        )

    optimizer_step.assert_not_called()


@pytest.mark.parametrize("nonfinite_norm", [float("nan"), float("inf")])
def test_optimizer_step_rejects_nonfinite_gradient_norm_before_update(monkeypatch, nonfinite_norm):
    model = tiny_model()
    optimizer = build_optimizer(model, learning_rate=1e-2)
    optimizer_step = Mock()
    monkeypatch.setattr(optimizer, "step", optimizer_step)
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(nonfinite_norm),
    )
    sequence = tiny_data(1)["train"][0]["input_ids"].unsqueeze(0)

    with pytest.raises(FloatingPointError, match="gradient norm is not finite"):
        _optimizer_step(
            model,
            optimizer,
            iter([{"input_ids": sequence}]),
            accumulation_steps=1,
            learning_rate=1e-2,
            target=torch.device("cpu"),
        )

    optimizer_step.assert_not_called()


def test_nonfinite_loss_stops_short_training_run_at_initial_checkpoint(tmp_path, monkeypatch):
    loss_calls = 0

    def nonfinite_loss(*args):
        nonlocal loss_calls
        loss_calls += 1
        return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(train_module, "DATA_DIR", tmp_path)
    write_tiny_tokenizer(tmp_path)
    monkeypatch.setattr(train_module, "_training_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(training_run_module, "create_model", lambda *args: tiny_model())
    monkeypatch.setattr(train_module, "load_prepared_data", lambda: tiny_data(4))
    monkeypatch.setattr(train_module, "_measure_validation_loss", lambda *args, **kwargs: (1.0, 16))
    monkeypatch.setattr(train_module, "_generate_prompt_samples", lambda *args: [])
    monkeypatch.setattr(train_module, "_next_token_loss", nonfinite_loss)

    with pytest.raises(FloatingPointError, match="training loss is not finite: nan"):
        train_module.train(training_conf=short_training(2), run_conf=short_run())

    assert loss_calls == 1
    run_directory = next((tmp_path / "checkpoints").iterdir())
    assert (run_directory / "latest").is_dir()
    assert (run_directory / "best").is_dir()
    assert (run_directory / "latest" / "tokenizer.json").is_file()
    assert torch.load(run_directory / "latest" / TRAINING_STATE_FILE)["step"] == 0


def test_train_exits_before_model_setup_when_prepared_data_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    write_tiny_tokenizer(tmp_path)
    monkeypatch.setattr(
        train_module,
        "_training_device",
        lambda: (_ for _ in ()).throw(AssertionError("model setup started")),
    )

    with pytest.raises(
        FileNotFoundError, match=r"prepared training data.*scripts/prepare_data\.py"
    ):
        train_module.train(training_conf=short_training(2), run_conf=short_run())

    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
def test_distributed_accumulation_matches_one_process_effective_batch(tmp_path, architecture):
    device_batches = [
        {"input_ids": ((torch.arange(9) + index) % 32).unsqueeze(0)} for index in range(4)
    ]
    torch.manual_seed(0)
    reference_model = tiny_model(architecture)
    reference_optimizer = build_optimizer(reference_model, learning_rate=1e-2)
    reference_loss, reference_gradient_norm, reference_gradient_clipped = _optimizer_step(
        reference_model,
        reference_optimizer,
        iter(device_batches),
        accumulation_steps=len(device_batches),
        learning_rate=1e-2,
        target=torch.device("cpu"),
    )

    spawn(
        _distributed_optimizer_step,
        args=(2, str(tmp_path / "rendezvous"), str(tmp_path), device_batches, architecture),
        nprocs=2,
    )

    for rank in range(2):
        distributed = torch.load(tmp_path / f"rank-{rank}.pt")
        assert distributed["loss"] == pytest.approx(reference_loss)
        assert distributed["gradient_norm"] == pytest.approx(reference_gradient_norm)
        assert distributed["gradient_clipped"] is reference_gradient_clipped
        for name, parameter in reference_model.state_dict().items():
            torch.testing.assert_close(distributed["model"][name], parameter)


@pytest.fixture
def resume_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "DATA_DIR", tmp_path)
    write_tiny_tokenizer(tmp_path)
    monkeypatch.setattr(train_module, "_training_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(training_run_module, "create_model", lambda *args: tiny_model())
    monkeypatch.setattr(train_module, "load_prepared_data", lambda: tiny_data(6))
    monkeypatch.setattr(train_module, "_generate_prompt_samples", lambda *args: [])
    branch_dir = tmp_path / "checkpoints" / "branch"
    monkeypatch.setattr(train_module, "_checkpoint_directory", lambda distributed: branch_dir)

    return branch_dir


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
@pytest.mark.parametrize("continue_run", [False, True])
def test_latest_checkpoint_can_start_new_or_continue_source_run(
    tmp_path, resume_environment, continue_run, architecture
):
    branch_dir = resume_environment
    training = short_training(3)
    model = tiny_model(architecture)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoints" / "run" / "latest",
        model,
        build_optimizer(model, training.peak_learning_rate),
        step=2,
        training_config=training,
        run_config=short_run(),
        tokenizer_dir=tmp_path,
        training_seconds_elapsed=4.5,
        best_validation_loss=1.0,
        best_validation_step=2,
    )
    resumed = train_module.train(
        training_conf=training,
        run_conf=short_run(),
        resume=checkpoint,
        continue_run=continue_run,
    )

    assert resumed == (checkpoint if continue_run else branch_dir / "latest")
    state, _ = load_checkpoint(resumed, torch.device("cpu"), training)
    assert state["step"] == 3
    assert state["training_seconds_elapsed"] > 4.5
    if not continue_run:
        source_state, _ = load_checkpoint(checkpoint, torch.device("cpu"), training)
        assert source_state["step"] == 2


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
@pytest.mark.parametrize("continue_run", [False, True])
def test_milestone_checkpoint_can_start_new_or_continue_source_run(
    tmp_path, resume_environment, continue_run, architecture
):
    branch_dir = resume_environment
    training = short_training(3)
    model = tiny_model(architecture)
    milestone = save_checkpoint(
        tmp_path / "checkpoints" / "original" / "step-000002",
        model,
        build_optimizer(model, training.peak_learning_rate),
        step=2,
        training_config=training,
        run_config=short_run(),
        tokenizer_dir=tmp_path,
        training_seconds_elapsed=4.5,
        best_validation_loss=1.0,
        best_validation_step=2,
    )
    resumed = train_module.train(
        training_conf=training,
        run_conf=short_run(),
        resume=milestone,
        continue_run=continue_run,
    )

    assert resumed == (milestone.parent / "latest" if continue_run else branch_dir / "latest")
    source_state, _ = load_checkpoint(milestone, torch.device("cpu"), training)
    branch_state, _ = load_checkpoint(resumed, torch.device("cpu"), training)
    assert source_state["step"] == 2
    assert branch_state["step"] == 3


def test_validation_interval_also_controls_checkpoint_interval(tmp_path, monkeypatch):
    validations = []
    saved_steps = []

    def measure_validation_loss(*args, **kwargs):
        validations.append(None)
        return 1.0, 16

    real_save_checkpoint = save_checkpoint

    def save(*args, **kwargs):
        saved_steps.append(args[3])
        return real_save_checkpoint(*args, **kwargs)

    monkeypatch.setattr(train_module, "DATA_DIR", tmp_path)
    write_tiny_tokenizer(tmp_path)
    monkeypatch.setattr(train_module, "_training_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(training_run_module, "create_model", lambda *args: tiny_model())
    monkeypatch.setattr(train_module, "load_prepared_data", lambda: tiny_data())
    monkeypatch.setattr(
        train_module,
        "_optimizer_step",
        lambda *args, **kwargs: (1.0, 0.5, False),
    )
    monkeypatch.setattr(train_module, "_measure_validation_loss", measure_validation_loss)
    monkeypatch.setattr(train_module, "_generate_prompt_samples", lambda *args: [])
    monkeypatch.setattr(training_run_module, "save_checkpoint", save)

    run = replace(short_run(), validation_interval_steps=3)
    train_module.train(training_conf=short_training(5), run_conf=run)

    assert len(validations) == 3
    assert saved_steps == [0, 3, 5]


@pytest.mark.parametrize(
    ("selection", "expectation"),
    [(None, None), ("vanilla", None), ("vanilla", "vanilla"), ("rope", None), ("rope", "rope")],
)
def test_selected_training_and_resumed_update_match_uninterrupted_run(
    tmp_path, resume_environment, monkeypatch, selection, expectation
):
    training = short_training(3)
    run = replace(short_run(), validation_interval_steps=1, milestone_interval_checkpoints=1)
    architecture = selection or "vanilla"

    def create_selected_model(selected):
        assert torch.initial_seed() == training.seed
        model = tiny_model(selected)
        if selected == "rope":
            model.config.rope_theta = 625.0
        return model

    monkeypatch.setattr(training_run_module, "create_model", create_selected_model)
    options = {} if selection is None else {"architecture": selection}
    uninterrupted = train_module.train(training_conf=training, run_conf=run, **options)
    expected_state, expected_model = load_checkpoint(uninterrupted, torch.device("cpu"), training)
    milestone = uninterrupted.parent / "step-000002"
    saved_state, saved_model = load_checkpoint(milestone, torch.device("cpu"), training)
    destination = tmp_path / "resumed"
    restored = TrainingState.initialize(
        training,
        checkpoint_dir=destination,
        resume=milestone,
        target=torch.device("cpu"),
        architecture=expectation,
    )
    assert restored.completed_steps == 2
    assert restored.training_seconds_elapsed == saved_state["training_seconds_elapsed"]
    assert restored.best_validation_loss == saved_state["best_validation_loss"]
    assert restored.best_validation_step == saved_state["best_validation_step"]
    assert restored.model.config.shape_dict() == saved_model.config.shape_dict()
    assert restored.model.config.model_type == f"ventris-{architecture}-v1"
    if architecture == "rope":
        assert restored.model.config.rope_theta == 625.0
    assert restored.optimizer.state
    for moments, saved in zip(
        restored.optimizer.state.values(), saved_state["optimizer"]["state"].values(), strict=True
    ):
        assert moments["step"].item() == 2
        assert torch.count_nonzero(moments["exp_avg"]) > 0
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(moments[key], saved[key], rtol=0, atol=0)

    monkeypatch.setattr(train_module, "_checkpoint_directory", lambda distributed: destination)
    resumed = train_module.train(
        training_conf=training,
        run_conf=run,
        resume=milestone,
        architecture=expectation,
    )
    actual_state, actual_model = load_checkpoint(resumed, torch.device("cpu"), training)
    assert actual_state["step"] == expected_state["step"] == 3
    for name, expected in expected_model.state_dict().items():
        torch.testing.assert_close(actual_model.state_dict()[name], expected, rtol=0, atol=0)
    for actual, expected in zip(
        actual_state["optimizer"]["state"].values(),
        expected_state["optimizer"]["state"].values(),
        strict=True,
    ):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("saved_architecture", ["vanilla", "rope"])
def test_resume_architecture_conflict_precedes_weight_and_optimizer_loading(
    tmp_path, resume_environment, saved_architecture
):
    checkpoint = tmp_path / "config-only"
    tiny_model(saved_architecture).config.save_pretrained(checkpoint)
    requested = "rope" if saved_architecture == "vanilla" else "vanilla"

    with pytest.raises(ValueError, match=f"requested.*{requested}.*saved.*{saved_architecture}"):
        train_module.train(
            training_conf=short_training(3),
            run_conf=short_run(),
            resume=checkpoint,
            architecture=requested,
        )


@pytest.mark.parametrize("resume", [False, True])
def test_train_rejects_unsupported_programmatic_architecture(
    tmp_path, resume_environment, monkeypatch, resume
):
    from ventris.models import create_model

    monkeypatch.setattr(training_run_module, "create_model", create_model)
    checkpoint = tmp_path / "config-only"
    tiny_model().config.save_pretrained(checkpoint)
    with pytest.raises(ValueError, match="unsupported.*architecture.*mla"):
        train_module.train(
            training_conf=short_training(3),
            run_conf=short_run(),
            resume=checkpoint if resume else None,
            architecture="mla",
        )
