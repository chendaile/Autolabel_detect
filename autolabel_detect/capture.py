"""Camera capture utilities supporting Jetson and standard webcams."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    use_jetcam: bool = False
    output_dir: Path = Path("captured_frames")
    camera_id: int = 0


class UniversalCamera:
    """Provide a unified camera capture interface."""

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self.config = config or CameraConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        self._require_cv2()
        try:
            if self.config.use_jetcam:
                pipeline = (
                    "nvarguscamerasrc ! "
                    f"video/x-raw(memory:NVMM), width={self.config.width}, height={self.config.height}, format=NV12, "
                    f"framerate={self.config.fps}/1 ! "
                    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink"
                )
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                self.cap = cv2.VideoCapture(self.config.camera_id)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        except Exception:
            self.cap = None
        return bool(self.cap and self.cap.isOpened())

    @staticmethod
    def _require_cv2() -> None:
        if cv2 is None:  # pragma: no cover - informative failure
            raise RuntimeError("OpenCV is required for camera capture") from _CV2_IMPORT_ERROR

    # ------------------------------------------------------------------
    def capture_frames(self, total_frames: int, interval: float = 0.1, show_preview: bool = True) -> int:
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera not initialised. Call initialize() first.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        captured = 0
        try:
            while captured < total_frames:
                ret, frame = self.cap.read()
                if not ret:
                    break
                filename = f"frame_{timestamp}_{captured:06d}.jpg"
                cv2.imwrite(str(self.output_dir / filename), frame)
                captured += 1
                if show_preview:
                    cv2.imshow("Camera Preview", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if interval > 0:
                    time.sleep(interval)
        finally:
            if show_preview:
                cv2.destroyAllWindows()
        return captured

    def capture_continuous(self, interval: float = 1.0, show_preview: bool = True) -> int:
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera not initialised. Call initialize() first.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        captured = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                filename = f"continuous_{timestamp}_{captured:06d}.jpg"
                cv2.imwrite(str(self.output_dir / filename), frame)
                captured += 1
                if show_preview:
                    cv2.imshow("Camera Preview", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                time.sleep(interval)
        finally:
            if show_preview:
                cv2.destroyAllWindows()
        return captured

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            self.cap = None
