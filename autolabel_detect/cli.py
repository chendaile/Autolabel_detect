"""Command line interface for the autolabel-detect toolkit."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .autolabel import AutoLabelConfig, YOLOAutoLabeler
from .capture import CameraConfig, UniversalCamera
from .datasets import SplitConfig, split_dataset
from .detection import DetectorConfig, YOLODetector
from .training import TrainingConfig, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autolabel-detect",
        description="Utilities for managing the lifecycle of YOLO vision models",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    detect = subparsers.add_parser("detect", help="Run inference on an image, video or camera")
    detect.add_argument("--model", "-m", required=True, type=Path, help="Path to the YOLO model (.pt)")
    detect.add_argument("--source", "-s", default="0", help="Input source (camera id or file path)")
    detect.add_argument("--imgsz", "-i", type=int, default=640, help="Image size used for inference")
    detect.add_argument("--flip", "-f", type=int, default=1, choices=[-1, 0, 1, 2], help="Flip mode: 2 disables flipping")
    detect.add_argument("--conf", "-c", type=float, default=0.3, help="Confidence threshold")
    detect.add_argument("--save", action="store_true", help="Persist the annotated output")
    detect.add_argument("--output", "-o", type=Path, default=Path("test_result"), help="Directory to save results")
    detect.add_argument("--jetson", action="store_true", help="Use Jetson CSI camera pipeline")
    detect.add_argument("--cam-width", type=int, default=1280, help="Camera capture width")
    detect.add_argument("--cam-height", type=int, default=720, help="Camera capture height")
    detect.add_argument("--cam-fps", type=int, default=30, help="Camera capture FPS")

    # ------------------------------------------------------------------
    auto = subparsers.add_parser("autolabel", help="Generate YOLO labels for an image directory")
    auto.add_argument("--model", "-m", required=True, type=Path, help="Path to trained YOLO model (.pt)")
    auto.add_argument("--input", "-i", required=True, type=Path, help="Directory containing raw images")
    auto.add_argument("--output", "-o", type=Path, default=Path("data"), help="Output directory")
    auto.add_argument("--classes", nargs="*", help="Optional list of class names overriding the model metadata")

    # ------------------------------------------------------------------
    train = subparsers.add_parser("train", help="Fine-tune or train a YOLO model")
    train.add_argument("--model", "-m", type=Path, default=Path("yolo11n.pt"), help="Path to base model")
    train.add_argument("--data", type=Path, default=Path("data.yaml"), help="Path to dataset YAML")
    train.add_argument("--batch", type=float, default=0.9, help="Training batch size or ratio")
    train.add_argument("--cache", action="store_true", help="Cache images during training")
    train.add_argument("--time", "-t", type=float, required=True, help="Training time limit in hours")
    train.add_argument("--name", type=str, help="Run name used for logging")
    train.add_argument("--project", "-P", type=Path, default=Path("train_results"), help="Output root directory")
    train.add_argument("--resume", action="store_true", help="Resume from the last training checkpoint")

    # ------------------------------------------------------------------
    split = subparsers.add_parser("split", help="Split a labelled dataset into train/validation")
    split.add_argument("--dataset", "-d", required=True, type=Path, help="Dataset root containing images/ and labels/")
    split.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of images to allocate to the train set")
    split.add_argument("--output", "-o", type=Path, default=Path("data"), help="Directory for the split dataset")

    # ------------------------------------------------------------------
    capture = subparsers.add_parser("capture", help="Capture frames from a camera")
    capture.add_argument("--frames", "-f", type=int, default=10, help="Number of frames to capture")
    capture.add_argument("--width", type=int, default=640, help="Capture width")
    capture.add_argument("--height", type=int, default=480, help="Capture height")
    capture.add_argument("--fps", type=int, default=30, help="Capture FPS")
    capture.add_argument("--interval", type=float, default=0.5, help="Interval between frames")
    capture.add_argument("--output", "-o", type=Path, default=Path("captured_frames"), help="Output directory")
    capture.add_argument("--jetson", action="store_true", help="Use Jetson CSI camera pipeline")
    capture.add_argument("--camera-id", type=int, default=0, help="Camera identifier")
    capture.add_argument("--continuous", action="store_true", help="Continuously capture frames until interrupted")
    capture.add_argument("--no-preview", action="store_true", help="Disable the live preview window")

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "detect":
        config = DetectorConfig(
            model_path=args.model,
            source=args.source,
            imgsz=args.imgsz,
            flip_mode=None if args.flip == 2 else args.flip,
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
    elif args.command == "autolabel":
        config = AutoLabelConfig(
            model_path=args.model,
            input_dir=args.input,
            output_dir=args.output,
            class_names=args.classes,
        )
        labeler = YOLOAutoLabeler(config)
        labeler.process()
    elif args.command == "train":
        config = TrainingConfig(
            model=args.model,
            data=args.data,
            batch=args.batch,
            cache=args.cache,
            time=args.time,
            name=args.name,
            project=args.project,
            resume=args.resume,
        )
        train_model(config)
    elif args.command == "split":
        config = SplitConfig(dataset_root=args.dataset, train_ratio=args.train_ratio, output_root=args.output)
        train_count, val_count = split_dataset(config)
        print(f"Train images: {train_count}, validation images: {val_count}")
    elif args.command == "capture":
        camera_config = CameraConfig(
            width=args.width,
            height=args.height,
            fps=args.fps,
            use_jetcam=args.jetson,
            output_dir=args.output,
            camera_id=args.camera_id,
        )
        camera = UniversalCamera(camera_config)
        if not camera.initialize():
            raise RuntimeError("Failed to initialise camera")
        try:
            if args.continuous:
                camera.capture_continuous(interval=args.interval, show_preview=not args.no_preview)
            else:
                camera.capture_frames(args.frames, interval=args.interval, show_preview=not args.no_preview)
        finally:
            camera.release()
    else:  # pragma: no cover - defensive
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
