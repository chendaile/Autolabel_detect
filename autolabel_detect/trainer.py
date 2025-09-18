"""Training helpers built around the Ultralytics YOLO interface."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO as _UltralyticsYOLO
except Exception:  # pragma: no cover - optional dependency
    _UltralyticsYOLO = None  # type: ignore


@dataclass
class TrainingConfig:
    """Configuration for training a YOLO model."""

    data: Union[str, Path]
    project: Union[str, Path]
    model: Union[str, Path, object] = "yolo11n.pt"
    batch: float = 0.9
    cache: bool = True
    time: float = 1.0
    name: Optional[str] = None
    resume: Union[bool, str] = False


class YOLOTrainer:
    """Wraps the Ultralytics training API with higher level ergonomics."""

    def __init__(self, model: Union[str, Path, object]):
        self.model = self._load_model(model)

    @staticmethod
    def _load_model(model: Union[str, Path, object]) -> object:
        if isinstance(model, (str, Path)):
            if _UltralyticsYOLO is None:
                raise ImportError("Ultralytics is required to load YOLO models from disk.")
            return _UltralyticsYOLO(str(model))
        return model

    def train(self, config: TrainingConfig) -> None:
        train_args = {
            "data": str(config.data),
            "batch": config.batch,
            "cache": config.cache,
            "time": config.time,
            "name": config.name,
            "resume": config.resume,
            "profile": True,
            "exist_ok": True,
            "project": str(Path("train_results") / config.project),
        }
        # Remove None values for cleanliness
        train_args = {k: v for k, v in train_args.items() if v is not None}
        self.model.train(**train_args)


__all__ = ["TrainingConfig", "YOLOTrainer"]
