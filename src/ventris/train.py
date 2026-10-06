"""Optimize Ventris and measure validation loss."""

import math
import os
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from ventris.common import (
    DEFAULT_RUN_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    RunConfig,
    TrainingConfig,
    default_device,
    evaluation_mode,
    mixed_precision,
)
from ventris.data import DATA_DIR, load_prepared_data, load_tokenizer
from ventris.models import Model
from ventris.training_results import StepResult, ValidationResult
from ventris.training_run import TrainingRun, TrainingState


@dataclass(frozen=True)
class DistributedContext:
    """The current process's place in a distributed training run."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


PROMPT_TESTS = (
    "The capital of France is",
    "Once upon a time",
    "The meaning of life is",
    "In a surprising scientific discovery, researchers found",
    "def fibonacci(n):",
)


def train(
    training_conf: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    run_conf: RunConfig = DEFAULT_RUN_CONFIG,
    resume: Path | None = None,
    *,
    continue_run: bool = False,
    architecture: str | None = None,
) -> Path:
    """Train through ``training_conf.steps`` and return the latest checkpoint."""
    if architecture is None and resume is None:
        raise ValueError("architecture is required for fresh training")
    distributed, initialized_here = _initialize_distributed()
    try:
        return _train(training_conf, run_conf, resume, distributed, continue_run, architecture)
    finally:
        if initialized_here:
            dist.destroy_process_group()


def _train(
    training_conf: TrainingConfig,
    run_conf: RunConfig,
    resume: Path | None,
    distributed: DistributedContext,
    continue_run: bool,
    architecture: str | None,
) -> Path:
    # Every process follows this training path. The branches only select
    # multi-process mechanics and primary-process side effects.
    _validate_training_config(training_conf)
    _validate_run_config(training_conf, run_conf, distributed.world_size)
    if continue_run and resume is None:
        raise ValueError("continue_run requires a resume checkpoint")
    data = load_prepared_data()
    target = _training_device()
    # This permits TF32 for FP32 matmuls; autocast separately selects BF16 operations.
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(training_conf.seed)
    checkpoint_dir = (
        resume.parent if continue_run and resume is not None else _checkpoint_directory(distributed)
    )
    state = TrainingState.initialize(
        training_conf,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        target=target,
        architecture=architecture,
    )

    batches, accumulation_steps = _prepare_training_batches(
        data["train"],
        training_conf,
        run_conf,
        distributed,
        state.completed_steps,
        target,
    )

    training_model = _prepare_training_model(state.model, run_conf, distributed)
    run = TrainingRun(
        state,
        training_conf,
        run_conf,
        accumulation_steps=accumulation_steps,
        world_size=distributed.world_size,
        is_primary=distributed.is_primary,
        tokenizer_dir=DATA_DIR,
    )

    def validate() -> ValidationResult | None:
        run.start_validation()
        return _validate_model(
            state.model,
            data["validation"],
            device_batch_size=run_conf.device_batch_size,
            target=target,
            max_tokens=run_conf.validation_tokens,
            distributed=distributed,
            on_batch=run.validation_batch_completed,
        )

    with run:
        run.initial_validation_completed(validate())
        if distributed.is_distributed:
            dist.barrier()

        for step in range(state.completed_steps, training_conf.steps):
            run.start_step()
            started = time.perf_counter()
            learning_rate = _learning_rate(step, training_conf)
            loss, gradient_norm, gradient_clipped = _optimizer_step(
                training_model,
                state.optimizer,
                batches,
                accumulation_steps,
                learning_rate,
                target,
                gradient_clip_norm=training_conf.gradient_clip_norm,
                on_device_batch=run.device_batch_completed,
            )
            if target.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            step_result = StepResult(
                completed=step + 1,
                loss=loss,
                learning_rate=learning_rate,
                elapsed=elapsed,
                gradient_norm=gradient_norm,
                gradient_clipped=gradient_clipped,
            )

            should_validate = _should_validate(
                step_result.completed,
                interval_steps=run_conf.validation_interval_steps,
                total_steps=training_conf.steps,
            )
            validation = validate() if should_validate else None
            run.step_completed(step_result, validation)

            if distributed.is_distributed and should_validate:
                dist.barrier()

    if distributed.is_distributed:
        dist.barrier()
    return run.latest_checkpoint


def _initialize_distributed() -> tuple[DistributedContext, bool]:
    """Join a torchrun process group when the environment requests one."""
    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", str(torch.cuda.current_device())))
        torch.cuda.set_device(local_rank)
        return DistributedContext(dist.get_rank(), dist.get_world_size(), local_rank), False

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistributedContext(), False

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    return DistributedContext(dist.get_rank(), dist.get_world_size(), local_rank), True


def _validate_training_config(training_conf: TrainingConfig) -> None:
    if training_conf.effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    if not 0 < training_conf.warmup_steps < training_conf.steps:
        raise ValueError("warmup_steps must be between zero and steps")
    if not 0 <= training_conf.minimum_learning_rate <= training_conf.peak_learning_rate:
        raise ValueError(
            "learning rates must satisfy 0 <= minimum_learning_rate <= peak_learning_rate"
        )
    if training_conf.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")


def _validate_run_config(
    training_conf: TrainingConfig,
    run_conf: RunConfig,
    world_size: int = 1,
) -> None:
    if run_conf.device_batch_size <= 0:
        raise ValueError("device_batch_size must be positive")
    distributed_device_batch_size = run_conf.device_batch_size * world_size
    if training_conf.effective_batch_size % distributed_device_batch_size:
        if world_size == 1:
            raise ValueError("device_batch_size must divide effective_batch_size")
        raise ValueError(
            "device_batch_size * world_size must divide effective_batch_size "
            f"({run_conf.device_batch_size} * {world_size} does not divide "
            f"{training_conf.effective_batch_size})"
        )
    if run_conf.validation_interval_steps <= 0:
        raise ValueError("validation_interval_steps must be positive")
    if run_conf.milestone_interval_checkpoints <= 0:
        raise ValueError("milestone_interval_checkpoints must be positive")
    if run_conf.validation_tokens <= 0:
        raise ValueError("validation_tokens must be positive")


def _training_device() -> torch.device:
    """Return a CUDA device with native BF16 support for pretraining."""
    target = default_device()
    if target.type != "cuda":
        raise RuntimeError("training requires a CUDA GPU with native BF16 support")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("training requires native BF16 hardware; emulation is disabled")
    return target


def _checkpoint_directory(distributed: DistributedContext) -> Path:
    """Choose one checkpoint directory and share it with every rank."""
    path = None
    if distributed.is_primary:
        path = str(DATA_DIR / "checkpoints" / time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()))
    if distributed.is_distributed:
        paths = [path]
        dist.broadcast_object_list(paths)
        path = paths[0]
    assert path is not None
    return Path(path)


def _prepare_training_batches(
    training_data: Dataset,
    training_conf: TrainingConfig,
    run_conf: RunConfig,
    distributed: DistributedContext,
    start_step: int,
    target: torch.device,
) -> tuple[Iterator[dict[str, torch.Tensor]], int]:
    """Build this process's deterministic batch stream and accumulation count."""
    required_sequences = training_conf.steps * training_conf.effective_batch_size
    if len(training_data) < required_sequences:
        raise ValueError(
            f"{training_conf.steps} steps need {required_sequences} sequences; "
            f"the prepared data has {len(training_data)}"
        )

    first_sequence = start_step * training_conf.effective_batch_size
    stop_sequence = training_conf.steps * training_conf.effective_batch_size
    training_data = training_data.shuffle(seed=training_conf.seed).select(
        range(first_sequence, stop_sequence)
    )
    if distributed.is_distributed:
        training_data = training_data.shard(
            num_shards=distributed.world_size,
            index=distributed.rank,
            contiguous=False,
        )
    loader = DataLoader(
        cast(TorchDataset[dict[str, torch.Tensor]], training_data),
        batch_size=run_conf.device_batch_size,
        pin_memory=target.type == "cuda",
    )
    accumulation_steps = training_conf.effective_batch_size // (
        run_conf.device_batch_size * distributed.world_size
    )
    return iter(loader), accumulation_steps


def _prepare_training_model(
    model: Model, run_conf: RunConfig, distributed: DistributedContext
) -> torch.nn.Module:
    """Compile and wrap the model used for optimizer steps."""
    if run_conf.compile_model:
        model.compile()
    if distributed.is_distributed:
        return DistributedDataParallel(
            model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            gradient_as_bucket_view=True,
        )
    return model


def _learning_rate(step: int, training_conf: TrainingConfig = DEFAULT_TRAINING_CONFIG) -> float:
    """Compute the learning rate using linear warmup followed by cosine decay."""
    if step < training_conf.warmup_steps:
        return training_conf.peak_learning_rate * (step + 1) / training_conf.warmup_steps
    progress = min(
        (step - training_conf.warmup_steps) / (training_conf.steps - training_conf.warmup_steps),
        1.0,
    )
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return training_conf.minimum_learning_rate + cosine * (
        training_conf.peak_learning_rate - training_conf.minimum_learning_rate
    )


def _optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterator[dict[str, torch.Tensor]],
    accumulation_steps: int,
    learning_rate: float,
    target: torch.device,
    gradient_clip_norm: float = DEFAULT_TRAINING_CONFIG.gradient_clip_norm,
    on_device_batch: Callable[[], None] | None = None,
) -> tuple[float, float, bool]:
    """Accumulate one effective batch and update the model once."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad()
    total_loss = torch.zeros((), device=target)

    for device_batch_index in range(accumulation_steps):
        input_ids, labels = _prepare_next_token_batch(next(batches), target)
        suppress_sync = device_batch_index < accumulation_steps - 1
        if isinstance(model, DistributedDataParallel) and suppress_sync:
            synchronization_context = model.no_sync()
        else:
            synchronization_context = nullcontext()
        with synchronization_context:
            with mixed_precision(target):
                loss = _next_token_loss(model, input_ids, labels)
            # BF16 has FP32's exponent range, so gradient scaling is unnecessary.
            (loss / accumulation_steps).backward()
        total_loss += loss.detach() / accumulation_steps
        if on_device_batch is not None:
            on_device_batch()

    if dist.is_initialized():
        dist.all_reduce(total_loss)
        total_loss /= dist.get_world_size()
    loss_value = total_loss.item()
    if not math.isfinite(loss_value):
        raise FloatingPointError(f"training loss is not finite: {loss_value}")

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        gradient_clip_norm,
    ).item()
    if not math.isfinite(gradient_norm):
        raise FloatingPointError(f"gradient norm is not finite: {gradient_norm}")

    gradient_clipped = gradient_norm > gradient_clip_norm
    optimizer.step()
    return loss_value, gradient_norm, gradient_clipped


def _should_validate(completed: int, *, interval_steps: int, total_steps: int) -> bool:
    """Return whether a completed step should run validation."""
    return completed % interval_steps == 0 or completed == total_steps


def _validate_model(
    model: Model,
    validation: Dataset,
    *,
    device_batch_size: int,
    target: torch.device,
    max_tokens: int,
    distributed: DistributedContext | None = None,
    on_batch: Callable[[int], None] | None = None,
) -> ValidationResult | None:
    """Measure validation on every rank and report samples on the primary rank."""
    distributed = distributed or DistributedContext()
    started = time.perf_counter()
    loss, tokens = _measure_validation_loss(
        model,
        validation,
        device_batch_size=device_batch_size,
        target=target,
        max_tokens=max_tokens,
        distributed=distributed,
        on_batch=on_batch,
    )
    if target.type == "cuda":
        torch.cuda.synchronize()
    if not distributed.is_primary:
        return None
    elapsed = time.perf_counter() - started
    return ValidationResult(
        loss=loss,
        elapsed=elapsed,
        tokens=tokens,
        samples=_generate_prompt_samples(model),
    )


def _measure_validation_loss(
    model: torch.nn.Module,
    validation: Dataset,
    *,
    device_batch_size: int,
    target: torch.device,
    max_tokens: int | None = None,
    distributed: DistributedContext | None = None,
    on_batch: Callable[[int], None] | None = None,
) -> tuple[float, int]:
    """Return measurements from next-token loss over a validation dataset."""
    distributed = distributed or DistributedContext()
    validation, total_progress_tokens = _prepare_validation_data(
        validation,
        device_batch_size=device_batch_size,
        max_tokens=max_tokens,
        distributed=distributed,
    )

    loader = DataLoader(
        cast(TorchDataset[dict[str, torch.Tensor]], validation),
        batch_size=device_batch_size,
        pin_memory=target.type == "cuda",
    )
    with evaluation_mode(model):
        total_loss = 0.0
        tokens_seen = 0
        progress_reported = 0
        for batch in loader:
            input_ids, labels = _prepare_next_token_batch(batch, target)
            with mixed_precision(target):
                loss = _next_token_loss(model, input_ids, labels)
            total_loss += loss.item() * labels.numel()
            tokens_seen += labels.numel()
            if on_batch is not None:
                # Estimate global progress from this rank, capped at the selected prefix.
                progress = min(tokens_seen * distributed.world_size, total_progress_tokens)
                on_batch(progress - progress_reported)
                progress_reported = progress
        if distributed.is_distributed:
            totals = torch.tensor([total_loss, tokens_seen], dtype=torch.float64, device=target)
            dist.all_reduce(totals)
            total_loss, tokens_seen = totals.tolist()
            tokens_seen = int(tokens_seen)
        if max_tokens is not None and tokens_seen < max_tokens:
            raise ValueError(
                f"validation split contains {tokens_seen:,} tokens, "
                f"fewer than the requested {max_tokens:,}"
            )
        if tokens_seen == 0:
            raise ValueError("validation split is empty")
        return total_loss / tokens_seen, tokens_seen


def _prepare_validation_data(
    validation: Dataset,
    *,
    device_batch_size: int,
    max_tokens: int | None,
    distributed: DistributedContext,
) -> tuple[Dataset, int]:
    """Select a complete-device-batch prefix, then share it across validation ranks."""
    if len(validation) == 0:
        raise ValueError("validation split is empty")
    # Prepared sequences have a fixed length, including one extra token for labels.
    sequence_tokens = len(cast(dict[str, torch.Tensor], validation[0])["input_ids"]) - 1
    sequence_count = len(validation)
    if max_tokens is not None:
        batch_tokens = device_batch_size * sequence_tokens
        requested_sequences = math.ceil(max_tokens / batch_tokens) * device_batch_size
        sequence_count = min(sequence_count, requested_sequences)
    validation = validation.select(range(sequence_count))
    if distributed.is_distributed:
        validation = validation.shard(
            num_shards=distributed.world_size,
            index=distributed.rank,
            contiguous=False,
        )
    return validation, sequence_count * sequence_tokens


def _generate_prompt_samples(model: Model) -> list[tuple[str, str]]:
    """Generate stable qualitative comparisons for a checkpoint report."""
    from ventris.generate import generate_from_model

    tokenizer = load_tokenizer()
    return [
        (
            prompt,
            generate_from_model(
                model,
                tokenizer,
                prompt,
                max_new_tokens=80,
                temperature=0.8,
                top_k=50,
                seed=index,
            ),
        )
        for index, prompt in enumerate(PROMPT_TESTS)
    ]


def _prepare_next_token_batch(
    batch: dict[str, torch.Tensor],
    target: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move packed tokens to the device and shift labels one token ahead of inputs."""
    non_blocking = target.type == "cuda"
    token_ids = batch["input_ids"].to(target, non_blocking=non_blocking)
    return token_ids[:, :-1], token_ids[:, 1:]


def _next_token_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids).logits
    return F.cross_entropy(logits.float().flatten(end_dim=1), labels.flatten())
