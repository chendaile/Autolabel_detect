# Autolabel Detect

Autolabel Detect is a lightweight toolkit that packages the training and automatic labelling
workflow for Ultralytics YOLO models.  It focuses on reproducible visual model development on any
Linux workstation, while still remaining compatible with Jetson devices when required.

## Installation

```bash
pip install -e .[full]
```

The optional `full` extra installs `ultralytics` and `opencv-python`.  If you already have these
packages installed or would like to provide custom implementations, omit the extra.

## Command Line Interface

The package exposes a single CLI entry point called `autolabel-detect` with three subcommands that
cover the end-to-end workflow.

### 1. Automatic labelling

```bash
autolabel-detect autolabel path/to/model.pt path/to/images path/to/output \
  --classes class_a class_b class_c
```

This command copies images to the output directory and generates YOLO-format label files using a
trained model.  Existing label files are overwritten unless `--no-overwrite` is supplied.

### 2. Training

```bash
autolabel-detect train my_project configs/data.yaml --model yolov8n.pt --time 1.5
```

This wraps `ultralytics.YOLO.train` while automatically storing results under
`train_results/<project>`.  Any compatible YOLO checkpoint can be provided via `--model` and the
dataset configuration via `--data`.

### 3. Dataset split utility

```bash
autolabel-detect split path/to/dataset path/to/output --train-ratio 0.75 --seed 123
```

Split an existing dataset containing `images/` and `labels/` folders into training and validation
subdirectories.  Files are copied by default; pass `--move` to relocate instead.

## Python API

The same features are available as importable helpers:

```python
from autolabel_detect import YOLOAutoLabeler, YOLOTrainer, TrainingConfig, split_dataset

labeler = YOLOAutoLabeler("weights/best.pt")
labeler.process_folder("./raw_images", "./prepared")

trainer = YOLOTrainer("yolo11n.pt")
trainer.train(TrainingConfig(data="data.yaml", project="demo", time=1.0))

split_dataset("./dataset", "./dataset_split", train_ratio=0.8, seed=2024)
```

The high-level classes accept already-instantiated model objects, making it easy to plug in mocks
or custom backends when running tests or deploying to constrained hardware.

## Running tests

```bash
pytest
```

The test-suite stubs the Ultralytics interface, so it can be executed without GPU hardware or
pretrained weights.  Use it to verify that the packaging and CLI behave as expected before sharing
the project with collaborators.
