"""Model training helpers built on top of Ultralytics YOLO."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency in tests
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class TrainingConfig:
    model: Path = Path("yolo11n.pt")
    data: Path = Path("data.yaml")
    batch: float = 0.9
    cache: bool = True
    time: float = 1.0
    name: Optional[str] = None
    project: Path = Path("train_results")
    resume: bool = False


def train_model(config: TrainingConfig) -> None:
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is required for training") from _IMPORT_ERROR
    model = YOLO(model=str(config.model), verbose=False)
    output_project = config.project / (config.name or "run")
    model.train(
        data=str(config.data),
        batch=config.batch,
        cache=config.cache,
        time=config.time,
        name=config.name,
        resume=config.resume,
        profile=True,
        exist_ok=True,
        project=str(output_project)
    )
