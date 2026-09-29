#!/usr/bin/env python3
"""Train Ventris Vanilla 124M Base."""

import argparse
import os
from dataclasses import replace
from pathlib import Path

from ventris.common import DEFAULT_RUN_CONFIG, DEFAULT_TRAINING_CONFIG
from ventris.train import train


def main() -> None:
    arguments = _parse_arguments()

    training_conf = replace(
        DEFAULT_TRAINING_CONFIG,
        steps=arguments.steps,
        seed=arguments.seed,
    )
    run_conf = replace(
        DEFAULT_RUN_CONFIG,
        device_batch_size=arguments.device_batch_size,
        validation_interval_steps=arguments.validation_interval,
        milestone_interval_checkpoints=arguments.milestone_interval,
        compile_model=arguments.compile,
        wandb_project=arguments.wandb_project,
    )
    checkpoint = train(
        training_conf=training_conf,
        run_conf=run_conf,
        resume=arguments.resume_checkpoint,
        continue_run=arguments.continue_run,
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
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        metavar="CHECKPOINT",
        help=(
            "checkpoint from which to resume training, for example "
            "data/checkpoints/2026-09-20_14-30-00/latest"
        ),
    )
    parser.add_argument(
        "--continue-run",
        action="store_true",
        help="write resumed checkpoints into the source run directory",
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
    if arguments.continue_run and arguments.resume_checkpoint is None:
        parser.error("--continue-run requires --resume-checkpoint")

    return arguments


if __name__ == "__main__":
    from torch.distributed.elastic.multiprocessing.errors import record

    record(main)()
