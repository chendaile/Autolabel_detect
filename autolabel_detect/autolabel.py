"""Utilities for generating YOLO-format labels with a trained model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO as _UltralyticsYOLO
except Exception:  # pragma: no cover - optional dependency
    _UltralyticsYOLO = None  # type: ignore


_ImageLoader = Callable[[Path], Any]
_ImageWriter = Callable[[Path, Any], None]


def _coerce_to_float_tuple(value: Any) -> Tuple[float, ...]:
    """Convert tensors/arrays/lists to a tuple of floats."""

    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    try:
        return (float(value),)
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise TypeError(f"Cannot convert value to float tuple: {value!r}") from exc


@dataclass
class DetectionResult:
    """Container representing a single detection in YOLO format."""

    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    def to_annotation_line(self) -> str:
        return f"{self.class_id} {self.center_x:.6f} {self.center_y:.6f} {self.width:.6f} {self.height:.6f}"


class YOLOAutoLabeler:
    """Generate YOLO-format annotations from images using a trained model."""

    def __init__(
        self,
        model: Union[str, Path, object],
        class_names: Optional[Sequence[str]] = None,
        image_loader: Optional[_ImageLoader] = None,
        image_writer: Optional[_ImageWriter] = None,
    ) -> None:
        self.model = self._load_model(model)
        self.class_names = self._resolve_class_names(class_names)
        self.image_loader = image_loader or self._default_loader
        self.image_writer = image_writer or self._default_writer

    def _load_model(self, model: Union[str, Path, object]) -> object:
        if isinstance(model, (str, Path)):
            if _UltralyticsYOLO is None:
                raise ImportError("Ultralytics is required to load YOLO models from disk.")
            return _UltralyticsYOLO(str(model))
        return model

    def _resolve_class_names(self, class_names: Optional[Sequence[str]]) -> dict[int, str]:
        if class_names is not None:
            return {i: name for i, name in enumerate(class_names)}

        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            return {int(i): str(name) for i, name in names.items()}
        if isinstance(names, Sequence):
            return {i: str(name) for i, name in enumerate(names)}
        return {}

    # --- image IO helpers -------------------------------------------------
    @staticmethod
    def _default_loader(path: Path) -> Any:
        import cv2  # Imported lazily to keep dependency optional.

        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image at {path}")
        return image

    @staticmethod
    def _default_writer(path: Path, image: Any) -> None:
        import cv2  # Imported lazily to keep dependency optional.

        if not cv2.imwrite(str(path), image):
            raise IOError(f"Failed to write image to {path}")

    # --- detection helpers ------------------------------------------------
    @staticmethod
    def _image_shape(image: Any) -> Sequence[int]:
        shape = getattr(image, "shape", None)
        if shape is None:
            raise AttributeError("Loaded images must expose a 'shape' attribute with height/width.")
        return shape

    @staticmethod
    def _extract_detections(model_output: Iterable) -> List[Tuple[Tuple[float, float, float, float], int]]:
        detections: List[Tuple[Tuple[float, float, float, float], int]] = []
        for result in model_output:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = _coerce_to_float_tuple(getattr(box, "xyxy")[0])
                cls_value = getattr(box, "cls")[0]
                cls = int(_coerce_to_float_tuple([cls_value])[0])
                detections.append((xyxy, cls))
        return detections

    @staticmethod
    def _convert_to_yolo(
        xyxy: Tuple[float, float, float, float],
        cls: int,
        image_shape: Sequence[int],
    ) -> DetectionResult:
        x1, y1, x2, y2 = map(float, xyxy)
        height, width = image_shape[:2]
        center_x = (x1 + x2) / 2 / width
        center_y = (y1 + y2) / 2 / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        return DetectionResult(cls, center_x, center_y, box_width, box_height)

    # --- public API -------------------------------------------------------
    def process_image(
        self,
        image_path: Union[str, Path],
        image: Optional[Any] = None,
    ) -> List[DetectionResult]:
        path = Path(image_path)
        if image is None:
            image = self.image_loader(path)
        output = self.model(image)
        detections: List[DetectionResult] = []
        shape = self._image_shape(image)
        for xyxy, cls in self._extract_detections(output):
            detections.append(self._convert_to_yolo(xyxy, cls, shape))
        return detections

    def process_folder(
        self,
        input_folder: Union[str, Path],
        output_dir: Union[str, Path],
        overwrite: bool = True,
    ) -> int:
        input_path = Path(input_folder)
        output_path = Path(output_dir)
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        processed = 0
        for image_file in sorted(input_path.iterdir()):
            if image_file.suffix.lower() not in image_extensions:
                continue
            label_file = labels_dir / f"{image_file.stem}.txt"
            if not overwrite and label_file.exists():
                continue
            image = self.image_loader(image_file)
            detections = self.process_image(image_file, image=image)
            self.image_writer(images_dir / image_file.name, image)
            label_lines = [d.to_annotation_line() for d in detections]
            label_file.write_text("\n".join(label_lines))
            processed += 1
        return processed


__all__ = ["YOLOAutoLabeler", "DetectionResult"]
