"""Command line entry points for the toolkit."""

from __future__ import annotations

import argparse
from typing import Sequence

from .detector import DetectorConfig, YOLODetector
from .labeler import SUPPORTED_IMAGE_EXTENSIONS, YOLOLabeler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO auto-labelling and detection toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    label_parser = subparsers.add_parser("label", help="Generate YOLO annotations for images")
    label_parser.add_argument("model", help="Path to the YOLO *.pt model file")
    label_parser.add_argument("input", help="Directory of images to label")
    label_parser.add_argument("output", help="Destination directory for the generated dataset")
    label_parser.add_argument(
        "--classes",
        nargs="+",
        help="Optional custom list of class names overriding the model metadata",
    )
    label_parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(SUPPORTED_IMAGE_EXTENSIONS),
        help="Image file extensions to consider",
    )

    detect_parser = subparsers.add_parser("detect", help="Run the interactive YOLO detector")
    detect_parser.add_argument("model", help="Path to the YOLO *.pt model file")
    detect_parser.add_argument(
        "--source",
        default="0",
        help="Input source (camera index, image path or video path)",
    )
    detect_parser.add_argument(
        "--duty",
        default="detect",
        choices=["detect", "segment", "classify", "pose"],
        help="Task to execute",
    )
    detect_parser.add_argument("--imgsz", type=int, default=640, help="Target image size")
    detect_parser.add_argument(
        "--flip",
        type=int,
        default=1,
        choices=[1, -1, 0, 2],
        help="Flip mode: 1=horizontal, -1=vertical, 0=both, 2=disable",
    )
    detect_parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold",
    )
    detect_parser.add_argument("--save", action="store_true", help="Persist the processed output")
    detect_parser.add_argument(
        "--output",
        default="./test_result",
        help="Directory for storing processed media",
    )
    detect_parser.add_argument("--jetson", action="store_true", help="Use Jetson CSI camera pipeline")
    detect_parser.add_argument("--cam-width", type=int, default=1280, help="Camera capture width")
    detect_parser.add_argument("--cam-height", type=int, default=720, help="Camera capture height")
    detect_parser.add_argument("--cam-fps", type=int, default=30, help="Camera capture FPS")

    return parser


def run_label_flow(args: argparse.Namespace) -> None:
    labeler = YOLOLabeler(args.model, args.classes)
    processed = labeler.label_directory(args.input, args.output, image_extensions=args.extensions)
    print(f"Processed {processed} image(s).")


def run_detection_flow(args: argparse.Namespace) -> None:
    flip_mode = None if args.flip == 2 else args.flip
    config = DetectorConfig(
        model_path=args.model,
        duty=args.duty,
        source=args.source,
        imgsz=args.imgsz,
        flip_mode=flip_mode,
        conf_threshold=args.conf,
        save_output=args.save,
        output_dir=args.output,
        jetson=args.jetson,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
        cam_fps=args.cam_fps,
    )
    detector = YOLODetector(config)
    detector.run()


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "label":
        run_label_flow(args)
    elif args.command == "detect":
        run_detection_flow(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
