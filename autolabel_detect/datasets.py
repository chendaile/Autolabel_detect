"""Dataset preparation utilities."""
from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


@dataclass
class SplitConfig:
    dataset_root: Path
    train_ratio: float = 0.8
    output_root: Path = Path("data")


def split_dataset(config: SplitConfig) -> Tuple[int, int]:
    if not config.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {config.dataset_root}")
    if not 0.01 <= config.train_ratio <= 0.99:
        raise ValueError("train_ratio must be between 0.01 and 0.99")
    image_dir = config.dataset_root / "images"
    label_dir = config.dataset_root / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory missing: {image_dir}")
    train_img_dir = config.output_root / "train" / "images"
    train_lbl_dir = config.output_root / "train" / "labels"
    val_img_dir = config.output_root / "validation" / "images"
    val_lbl_dir = config.output_root / "validation" / "labels"
    for directory in (train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir):
        directory.mkdir(parents=True, exist_ok=True)
    image_files = list(_iter_files(image_dir))
    random.shuffle(image_files)
    train_count = int(len(image_files) * config.train_ratio)
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]
    _copy_files(train_files, label_dir, train_img_dir, train_lbl_dir)
    _copy_files(val_files, label_dir, val_img_dir, val_lbl_dir)
    return len(train_files), len(val_files)


def _iter_files(directory: Path) -> Iterable[Path]:
    for path in directory.rglob("*"):
        if path.is_file():
            yield path


def _copy_files(files: Iterable[Path], label_dir: Path, image_target: Path, label_target: Path) -> None:
    for img_path in files:
        shutil.copy(img_path, image_target / img_path.name)
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, label_target / label_path.name)
