"""Compatibility wrapper around the new autolabel-detect training package."""
from __future__ import annotations

import argparse

from autolabel_detect import TrainingConfig, YOLOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model with autolabel-detect.")
    parser.add_argument("project", help="Project name (stored under train_results/)")
    parser.add_argument("data", help="Path to the dataset YAML file")
    parser.add_argument("--model", default="yolo11n.pt", help="Model weights to fine-tune")
    parser.add_argument("--batch", type=float, default=0.9, help="Batch size ratio")
    parser.add_argument("--cache", dest="cache", action="store_true", default=True, help="Cache dataset in memory")
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.add_argument("--time", type=float, required=True, help="Maximum training time (hours)")
    parser.add_argument("--name", help="Optional run name")
    parser.add_argument("--resume", default=False, help="Resume from a checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
