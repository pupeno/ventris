#!/usr/bin/env python3
"""Train the byte-level BPE tokenizer."""

import argparse
from pathlib import Path

from ventris.data import train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="directory for tokenizer files (default: data/)",
    )
    args = parser.parse_args()

    print(train_tokenizer(args.output_dir))


if __name__ == "__main__":
    main()
