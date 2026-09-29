#!/usr/bin/env python3
"""Generate one continuation from a checkpoint."""

import argparse
from pathlib import Path

from ventris.generate import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    print(
        generate(
            arguments.checkpoint,
            arguments.prompt,
            max_new_tokens=arguments.max_new_tokens,
            temperature=arguments.temperature,
            top_k=arguments.top_k,
            seed=arguments.seed,
        )
    )


if __name__ == "__main__":
    main()
