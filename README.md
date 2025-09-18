# Autolabel Detect

A lightweight toolkit that packages the original training, auto-labelling and
inference scripts into a reusable Python package. The focus of the project is to
provide an end-to-end workflow for computer-vision experiments built on top of
Ultralytics YOLO, regardless of whether you run it on Jetson, a workstation or a
cloud VM.

## Key features

- **Single CLI** – manage data capture, dataset splitting, auto-labelling,
  model training and inference from one command.
- **Python package** – installable via `pip` for reuse in other projects.
- **YOLO-first design** – Jetson-specific optimisations are supported but
  optional so the codebase remains portable.

## Installation

Create a virtual environment with Python 3.8+ and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The installation pulls in `opencv-python`, `numpy` and `ultralytics`.

## Command line usage

All functionality is exposed through the `autolabel-detect` CLI. Run
`autolabel-detect --help` to see the available sub-commands.

### 1. Capture frames for dataset bootstrapping

```bash
autolabel-detect capture \
  --frames 200 \
  --output data/raw_frames
```

Use `--continuous` for endless capture and `--jetson` to switch to the CSI
camera pipeline when running on a Jetson device.

### 2. Split an existing YOLO-format dataset

```bash
autolabel-detect split \
  --dataset /path/to/dataset \
  --train-ratio 0.85 \
  --output data
```

The command copies files into `data/train` and `data/validation` with the
expected `images/` and `labels/` sub-directories.

### 3. Automatically label a folder of images

```bash
autolabel-detect autolabel \
  --model runs/detect/train/weights/best.pt \
  --input data/raw_frames \
  --output data/auto_labelled
```

Optionally specify new class names with `--classes` to override the model
metadata.

### 4. Train or fine-tune a YOLO model

```bash
autolabel-detect train \
  --model yolo11n.pt \
  --data data.yaml \
  --time 2.0 \
  --project train_results \
  --name experiment_001
```

This mirrors the behaviour of `yolo train` while keeping runs inside the
repository.

### 5. Run inference on images, videos or live camera feeds

```bash
autolabel-detect detect \
  --model train_results/experiment_001/weights/best.pt \
  --source test/sample.mp4 \
  --save
```

Specify `--source 0` to run on a webcam, or provide an image path for single
image inference.

## Python API

The package exposes the main building blocks for programmatic use:

```python
from autolabel_detect import YOLOAutoLabeler, YOLODetector
from autolabel_detect.autolabel import AutoLabelConfig
from autolabel_detect.detection import DetectorConfig

labeler = YOLOAutoLabeler(AutoLabelConfig(model_path="best.pt", input_dir="images"))
labeler.process()

config = DetectorConfig(model_path="best.pt", source="video.mp4", save_output=True)
detector = YOLODetector(config)
detector.run()
```

## Testing

Tests are implemented with `pytest` and focus on validating the CLI interface
and dataset utilities. Execute them with:

```bash
pytest
```

## License

MIT
