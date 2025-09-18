"""Automatic dataset labelling helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - handled gracefully at runtime
    YOLO = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


@dataclass
class AutoLabelConfig:
    model_path: Path
    input_dir: Path
    output_dir: Path = Path("data")
    class_names: Optional[Sequence[str]] = None


class YOLOAutoLabeler:
    """Batch process images using a YOLO model and export YOLO-format labels."""

    def __init__(self, config: AutoLabelConfig) -> None:
        self.config = config
        self._require_cv2()
        self._ensure_model()
        self.output_images = self.config.output_dir / "images"
        self.output_labels = self.config.output_dir / "labels"
        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_labels.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is required for auto labelling") from _IMPORT_ERROR
        self.model = YOLO(str(self.config.model_path))
        if self.config.class_names is None:
            self.class_names = self.model.names
        else:
            self.class_names = {i: name for i, name in enumerate(self.config.class_names)}

    @staticmethod
    def _require_cv2() -> None:
        if cv2 is None:  # pragma: no cover - informative failure
            raise RuntimeError("OpenCV is required for auto labelling") from _CV2_IMPORT_ERROR

    # ------------------------------------------------------------------
    def process(self) -> List[Path]:
        image_files = self._find_images(self.config.input_dir)
        if not image_files:
            raise FileNotFoundError(f"No supported images found in {self.config.input_dir}")
        written = []
        for image_path in image_files:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            results = self.model(image)
            annotations = self._results_to_yolo_format(results, image.shape[:2])
            cv2.imwrite(str(self.output_images / image_path.name), image)
            label_path = self.output_labels / f"{image_path.stem}.txt"
            with label_path.open("w", encoding="utf-8") as fh:
                fh.write("\n".join(annotations))
            written.append(label_path)
        return written

    @staticmethod
    def _find_images(directory: Path) -> List[Path]:
        return [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]

    @staticmethod
    def _results_to_yolo_format(results, shape: Iterable[int]) -> List[str]:
        height, width = shape
        annotations: List[str] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2 / width
                cy = (y1 + y2) / 2 / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height
                class_id = int(box.cls[0].cpu().numpy())
                annotations.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return annotations
