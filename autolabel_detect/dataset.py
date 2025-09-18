"""Dataset utilities for splitting YOLO image/label sets."""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple


def _gather_files(folder: Path) -> List[Path]:
    files = [path for path in folder.rglob("*") if path.is_file()]
    files.sort()
    return files


def split_dataset(
    dataset_dir: Path | str,
    output_dir: Path | str,
    train_ratio: float = 0.8,
    seed: int | None = None,
    copy: bool = True,
) -> Tuple[List[Path], List[Path]]:
    """Split a YOLO dataset into train/validation folders.

    Args:
        dataset_dir: Directory containing ``images`` and ``labels`` subfolders.
        output_dir: Base directory where ``train`` and ``validation`` will be created.
        train_ratio: Fraction of images that should end up in the training split.
        seed: Optional random seed to make the process deterministic.
        copy: Whether to copy files (default) or move them.

    Returns:
        A tuple with the list of training image paths and validation image paths
        within the destination directory.
    """

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    dataset_path = Path(dataset_dir)
    images_src = dataset_path / "images"
    labels_src = dataset_path / "labels"
    if not images_src.is_dir() or not labels_src.is_dir():
        raise FileNotFoundError("Dataset directory must contain 'images' and 'labels'.")

    destination = Path(output_dir)
    train_img_dst = destination / "train" / "images"
    train_lbl_dst = destination / "train" / "labels"
    val_img_dst = destination / "validation" / "images"
    val_lbl_dst = destination / "validation" / "labels"
    for folder in (train_img_dst, train_lbl_dst, val_img_dst, val_lbl_dst):
        folder.mkdir(parents=True, exist_ok=True)

    images = _gather_files(images_src)
    if seed is not None:
        random.Random(seed).shuffle(images)
    else:
        random.shuffle(images)

    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    def _transfer(files: Sequence[Path], img_dst: Path, lbl_dst: Path) -> List[Path]:
        transferred: List[Path] = []
        for image_file in files:
            label_file = labels_src / f"{image_file.stem}.txt"
            if copy:
                shutil.copy2(image_file, img_dst / image_file.name)
                if label_file.exists():
                    shutil.copy2(label_file, lbl_dst / label_file.name)
            else:
                shutil.move(str(image_file), str(img_dst / image_file.name))
                if label_file.exists():
                    shutil.move(str(label_file), str(lbl_dst / label_file.name))
            transferred.append(img_dst / image_file.name)
        return transferred

    train_result = _transfer(train_images, train_img_dst, train_lbl_dst)
    val_result = _transfer(val_images, val_img_dst, val_lbl_dst)
    return train_result, val_result


__all__ = ["split_dataset"]
