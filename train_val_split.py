"""Compatibility script for splitting datasets via autolabel-detect."""
from __future__ import annotations

import argparse

from autolabel_detect import split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a dataset into training and validation sets.")
    parser.add_argument("dataset", help="Directory containing images/ and labels/ folders")
    parser.add_argument("output", help="Destination directory for the split dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of images for training")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible splits")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dataset(
        args.dataset,
        args.output,
        train_ratio=args.train_ratio,
        seed=args.seed,
        copy=not args.move,
    )


if __name__ == "__main__":
    main()
