#!/usr/bin/env python3
"""Measure fixed-text next-token loss and record it against exact tracked weights."""

import argparse

from ventris.evaluate import evaluate
from ventris.tracking import DEFAULT_MLFLOW_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="models:/<model-id> or native registry alias"
    )
    parser.add_argument("--mlflow-root", default=DEFAULT_MLFLOW_ROOT)
    arguments = parser.parse_args()
    print(evaluate(arguments.checkpoint, mlflow_root=arguments.mlflow_root))


if __name__ == "__main__":
    main()
