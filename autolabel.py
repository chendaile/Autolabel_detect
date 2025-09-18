"""Minimal wrapper to keep backwards compatibility with the old autolabel script."""
from __future__ import annotations

import argparse

from autolabel_detect import YOLOAutoLabeler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLO labels using a trained model.")
    parser.add_argument("--model", "-m", required=True, help="Path to the trained YOLO model")
    parser.add_argument("--input", "-i", required=True, help="Directory containing source images")
    parser.add_argument("--output", "-o", default="./data", help="Destination directory")
    parser.add_argument("--classes", "-c", nargs="*", help="Override class names")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip files that already have labels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labeler = YOLOAutoLabeler(args.model, class_names=args.classes)
    count = labeler.process_folder(args.input, args.output, overwrite=not args.no_overwrite)
    print(f"Processed {count} images from {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
