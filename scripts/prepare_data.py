#!/usr/bin/env python3
"""Tokenize FineWeb-Edu into cached training and validation sequences."""

import argparse

from ventris.data import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    data = prepare_data()
    print(f"training sequences:   {len(data['train']):,}")
    print(f"validation sequences: {len(data['validation']):,}")


if __name__ == "__main__":
    main()
