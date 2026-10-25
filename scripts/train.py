#!/usr/bin/env python3
"""Train a Ventris Base model."""

import argparse
import os
from dataclasses import replace

from ventris.common import DEFAULT_RUN_CONFIG, DEFAULT_TRAINING_CONFIG
from ventris.train import train


def main() -> None:
    arguments = _parse_arguments()

    training_conf = replace(
        DEFAULT_TRAINING_CONFIG,
        steps=arguments.steps,
        seed=arguments.seed,
        warmup_steps=arguments.warmup_steps,
        effective_batch_size=arguments.effective_batch_size,
    )
    run_conf = replace(
        DEFAULT_RUN_CONFIG,
        device_batch_size=arguments.device_batch_size,
        validation_interval_steps=arguments.validation_interval,
        milestone_interval_checkpoints=arguments.milestone_interval,
        compile_model=arguments.compile,
        wandb_project=arguments.wandb_project,
        mlflow_root=arguments.mlflow_root,
        validation_tokens=arguments.validation_tokens,
    )
    checkpoint = train(
        training_conf=training_conf,
        run_conf=run_conf,
        resume=arguments.fork_checkpoint,
        continue_run=arguments.continue_run,
        architecture=arguments.architecture,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(checkpoint)


def _parse_arguments() -> argparse.Namespace:
    training_defaults = DEFAULT_TRAINING_CONFIG
    run_defaults = DEFAULT_RUN_CONFIG
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--architecture",
        choices=("vanilla", "rope"),
        default=None,
        help="model architecture; required for fresh training",
    )
    parser.add_argument(
        "--device-batch-size",
        type=int,
        default=run_defaults.device_batch_size,
        help="sequences processed in one forward and backward pass",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=training_defaults.steps,
        help="total optimizer steps, including steps from a resumed checkpoint",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=run_defaults.validation_interval_steps,
        help="optimizer steps between validation passes, prompt testing, and checkpoints",
    )
    parser.add_argument(
        "--milestone-interval",
        type=int,
        default=run_defaults.milestone_interval_checkpoints,
        metavar="CHECKPOINTS",
        help="scheduled validation checkpoints between retained, resumable milestones",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=training_defaults.seed,
        help="model initialization and training data order seed",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--fork-checkpoint",
        metavar="CHECKPOINT",
        help="fork a models:/ reference, registry alias, or local checkpoint into a new run",
    )
    source.add_argument(
        "--continue-run",
        metavar="RUN_ID",
        help="continue an interrupted MLflow run automatically from its latest checkpoint",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=training_defaults.warmup_steps,
        help="optimizer steps of learning-rate warmup",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=training_defaults.effective_batch_size,
        help="sequences per optimizer step across all devices",
    )
    parser.add_argument(
        "--validation-tokens",
        type=int,
        default=run_defaults.validation_tokens,
        help="token budget per validation pass",
    )
    tracking = parser.add_mutually_exclusive_group()
    tracking.add_argument(
        "--mlflow-root",
        default=run_defaults.mlflow_root,
        help="local SQLite and artifact directory",
    )
    tracking.add_argument(
        "--no-mlflow",
        dest="mlflow_root",
        action="store_const",
        const=None,
        default=argparse.SUPPRESS,
        help="disable MLflow tracking and save local checkpoints",
    )
    parser.add_argument("--compile", action="store_true", help="use torch.compile")
    wandb = parser.add_mutually_exclusive_group()
    wandb.add_argument(
        "--wandb-project",
        default=run_defaults.wandb_project,
        metavar="PROJECT",
        help="Weights & Biases project for training metrics",
    )
    wandb.add_argument(
        "--no-wandb",
        action="store_const",
        const=None,
        default=argparse.SUPPRESS,
        dest="wandb_project",
        help="disable Weights & Biases reporting",
    )
    arguments = parser.parse_args()
    if (
        arguments.architecture is None
        and arguments.fork_checkpoint is None
        and arguments.continue_run is None
    ):
        parser.error("--architecture, --fork-checkpoint, or --continue-run is required")

    return arguments


if __name__ == "__main__":
    from torch.distributed.elastic.multiprocessing.errors import record

    record(main)()
