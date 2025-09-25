"""High level utilities for running YOLO models to auto generate labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


@dataclass
class LabelResult:
    """Container returned for every processed image."""

    image: np.ndarray
    annotations: List[str] = field(default_factory=list)

    def as_yolo_txt(self) -> str:
        """Serialise the annotations to the YOLO plain text format."""

        return "\n".join(self.annotations)


class YOLOLabeler:
    """Run inference with a YOLO model and export annotations in YOLO format."""

    def __init__(self, model_path: str | Path, class_names: Optional[Sequence[str]] = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")

        self.model = YOLO(str(self.model_path))

        if class_names is None:
            self.class_names = self.model.names
        else:
            self.class_names = {i: name for i, name in enumerate(class_names)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def label_image(self, image_path: str | Path) -> LabelResult:
        """Run inference on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            A :class:`LabelResult` containing the original image array and the
            annotations in YOLO text format.
        """

        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {path}")

        results = self.model(image)
        height, width = image.shape[:2]

        annotations: List[str] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                centre_x = (x1 + x2) / 2 / width
                centre_y = (y1 + y2) / 2 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                class_id = int(box.cls[0].cpu().numpy())
                annotations.append(
                    f"{class_id} {centre_x:.6f} {centre_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                )

        return LabelResult(image=image, annotations=annotations)

    def label_directory(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        *,
        image_extensions: Iterable[str] = SUPPORTED_IMAGE_EXTENSIONS,
    ) -> int:
        """Label all supported images in *image_dir* and persist the outputs.

        Args:
            image_dir: Directory containing input images.
            output_dir: Directory where the ``images`` and ``labels`` folders
                will be created.
            image_extensions: Optional custom set of file extensions.

        Returns:
            The number of images processed successfully.
        """

        input_dir = Path(image_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {input_dir}")

        output_dir = Path(output_dir)
        images_output = output_dir / "images"
        labels_output = output_dir / "labels"
        images_output.mkdir(parents=True, exist_ok=True)
        labels_output.mkdir(parents=True, exist_ok=True)

        extensions = {ext.lower() for ext in image_extensions}
        image_files = sorted(
            file for file in input_dir.iterdir() if file.suffix.lower() in extensions
        )
        if not image_files:
            raise FileNotFoundError(
                f"No images with supported extensions found in {input_dir}. "
                f"Supported: {', '.join(sorted(extensions))}."
            )

        processed = 0
        for image_path in image_files:
            result = self.label_image(image_path)

            cv2.imwrite(str(images_output / image_path.name), result.image)
            (labels_output / f"{image_path.stem}.txt").write_text(result.as_yolo_txt())
            processed += 1

        return processed


__all__ = ["YOLOLabeler", "LabelResult", "SUPPORTED_IMAGE_EXTENSIONS"]
