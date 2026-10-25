#!/usr/bin/env python3
"""Evaluate one model on all PIQA validation questions and save results in MLflow."""

import argparse

from ventris.evaluate import GPT2_REPOSITORY, evaluate
from ventris.tracking import DEFAULT_MLFLOW_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint", help="Tracked models:/MODEL_ID or registry alias")
    selection.add_argument("--hf-model", help=f"Hugging Face repository, e.g. {GPT2_REPOSITORY}")
    parser.add_argument("--hf-revision", help="Full repository commit for model and tokenizer")
    parser.add_argument("--mlflow-root", default=DEFAULT_MLFLOW_ROOT)
    parser.add_argument("--device", help="Torch device; defaults to CUDA when available")
    parser.add_argument(
        "--precision", choices=("float32", "bfloat16", "float16"), default="float32"
    )
    parser.add_argument("--device-batch-size", type=int, default=4)
    arguments = parser.parse_args()
    if bool(arguments.hf_model) != bool(arguments.hf_revision):
        parser.error("--hf-model and --hf-revision must be supplied together")
    run_id = evaluate(
        arguments.checkpoint,
        hf_model=arguments.hf_model,
        hf_revision=arguments.hf_revision,
        mlflow_root=arguments.mlflow_root,
        device=arguments.device,
        precision=arguments.precision,
        device_batch_size=arguments.device_batch_size,
    )
    print(f"Evaluation run: {run_id}")


if __name__ == "__main__":
    main()
