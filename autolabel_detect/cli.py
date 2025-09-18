"""Command line interface for the autolabel-detect toolkit."""
from __future__ import annotations

import argparse
from typing import Sequence

from .autolabel import YOLOAutoLabeler
from .dataset import split_dataset
from .trainer import TrainingConfig, YOLOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for training and using YOLO models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    label_parser = subparsers.add_parser("autolabel", help="Generate YOLO labels for a folder of images.")
    label_parser.add_argument("model", help="Path to the trained YOLO model.")
    label_parser.add_argument("input", help="Directory containing source images.")
    label_parser.add_argument("output", help="Destination directory for images and labels.")
    label_parser.add_argument("--classes", nargs="*", help="Override class names exposed by the model.")
    label_parser.add_argument("--no-overwrite", action="store_true", help="Skip files that already have labels.")

    train_parser = subparsers.add_parser("train", help="Train a YOLO model using ultralytics.")
    train_parser.add_argument("project", help="Name of the training project (stored under train_results/).")
    train_parser.add_argument("data", help="Path to the dataset YAML file.")
    train_parser.add_argument("--model", default="yolo11n.pt", help="Model checkpoint to fine-tune.")
    train_parser.add_argument("--batch", type=float, default=0.9, help="Batch size ratio.")
    train_parser.add_argument("--cache", action="store_true", help="Cache dataset in memory for faster training.")
    train_parser.add_argument("--no-cache", dest="cache", action="store_false")
    train_parser.set_defaults(cache=True)
    train_parser.add_argument("--time", type=float, required=True, help="Maximum training time (hours).")
    train_parser.add_argument("--name", help="Optional run name (defaults to timestamp from Ultralytics).")
    train_parser.add_argument("--resume", default=False, help="Resume training from a checkpoint path.")

    split_parser = subparsers.add_parser("split", help="Split a dataset into train/validation subsets.")
    split_parser.add_argument("dataset", help="Directory containing images/ and labels/ folders.")
    split_parser.add_argument("output", help="Destination directory for the split dataset.")
    split_parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of images for training.")
    split_parser.add_argument("--seed", type=int, help="Random seed for deterministic splits.")
    split_parser.add_argument("--move", action="store_true", help="Move files instead of copying.")

    return parser


def _cmd_autolabel(args: argparse.Namespace) -> None:
    labeler = YOLOAutoLabeler(args.model, class_names=args.classes)
    processed = labeler.process_folder(args.input, args.output, overwrite=not args.no_overwrite)
    print(f"Processed {processed} images from {args.input} -> {args.output}")


def _cmd_train(args: argparse.Namespace) -> None:
    trainer = YOLOTrainer(args.model)
    config = TrainingConfig(
        data=args.data,
        project=args.project,
        model=args.model,
        batch=args.batch,
        cache=args.cache,
        time=args.time,
        name=args.name,
        resume=args.resume,
    )
    trainer.train(config)


def _cmd_split(args: argparse.Namespace) -> None:
    train, val = split_dataset(
        args.dataset,
        args.output,
        train_ratio=args.train_ratio,
        seed=args.seed,
        copy=not args.move,
    )
    print(f"Created train split with {len(train)} images and validation split with {len(val)} images.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "autolabel":
        _cmd_autolabel(args)
    elif args.command == "train":
        _cmd_train(args)
    elif args.command == "split":
        _cmd_split(args)
    else:  # pragma: no cover - defensive fallback
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
