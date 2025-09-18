"""Detection utilities built around the Ultralytics YOLO API."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception as exc:  # pragma: no cover
    np = None  # type: ignore
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency during tests
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - handled at runtime
    YOLO = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


Number = Union[int, float]


@dataclass
class DetectorConfig:
    """Configuration options for :class:`YOLODetector`."""

    model_path: Union[str, Path]
    source: Union[int, str] = 0
    duty: str = "detect"
    imgsz: Union[int, Tuple[int, int]] = 640
    flip_mode: Optional[int] = 1
    conf_threshold: float = 0.3
    save_output: bool = False
    output_dir: Union[str, Path] = Path("test_result")
    jetson: bool = False
    cam_width: int = 1280
    cam_height: int = 720
    cam_fps: int = 30


class YOLODetector:
    """Wrapper that encapsulates video/image/camera processing with YOLO."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._require_cv2()
        self._require_numpy()
        self._ensure_model()
        self.labels = self.model.names if hasattr(self.model, "names") else {}
        self.colors = self._generate_colors()
        if self.config.save_output:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if YOLO is None:
            raise RuntimeError(
                "Ultralytics YOLO is required for detection tasks"  # pragma: no cover - informative
            ) from _IMPORT_ERROR
        self.model = YOLO(self.config.model_path, verbose=False)
        self.model.overrides["verbose"] = False

    @staticmethod
    def _require_cv2() -> None:
        if cv2 is None:  # pragma: no cover - informative failure
            raise RuntimeError("OpenCV is required for detection") from _CV2_IMPORT_ERROR

    @staticmethod
    def _require_numpy() -> None:
        if np is None:  # pragma: no cover - informative failure
            raise RuntimeError("NumPy is required for detection") from _NUMPY_IMPORT_ERROR

    @staticmethod
    def _generate_colors() -> Iterable[Tuple[int, int, int]]:
        return (
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (128, 0, 128),
            (255, 165, 0),
            (0, 128, 128),
            (128, 128, 0),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        input_type = self._get_input_type(self.config.source)
        if input_type == "image":
            self._detect_image(str(self.config.source))
        elif input_type == "video":
            self._detect_video(str(self.config.source))
        elif input_type == "camera":
            self._detect_camera()
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported input source: {self.config.source}")

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_input_type(source: Union[int, str]) -> str:
        if isinstance(source, int):
            return "camera"
        if isinstance(source, str) and source.isdigit():
            return "camera"
        lower = str(source).lower()
        if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
            return "image"
        if lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")):
            return "video"
        return "unknown"

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        size = self.config.imgsz
        if isinstance(size, int):
            h, w = frame.shape[:2]
            if not h or not w:  # pragma: no cover - defensive
                return frame
            if h > w:
                new_h, new_w = size, int(w * size / h)
            else:
                new_w, new_h = size, int(h * size / w)
        else:
            new_w, new_h = size
        return cv2.resize(frame, (new_w, new_h))

    def _flip_frame(self, frame: np.ndarray) -> np.ndarray:
        mode = self.config.flip_mode
        if mode is None:
            return frame
        mapping = {1: 1, -1: 0, 0: -1}
        if mode not in mapping:
            return frame
        return cv2.flip(frame, mapping[mode])

    def _draw_detections(self, frame: np.ndarray, results) -> np.ndarray:
        if not hasattr(results, "boxes") or results.boxes is None:
            return frame
        for idx, box in enumerate(results.boxes):
            xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
            if xyxy.shape != (4,):  # pragma: no cover - defensive
                continue
            cls_idx = int(box.cls.item())
            conf = float(box.conf.item())
            if conf < self.config.conf_threshold:
                continue
            color = self.colors[cls_idx % len(self.colors)]
            label = self.labels.get(cls_idx, f"Class{cls_idx}")
            x1, y1, x2, y2 = xyxy
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label}: {int(conf * 100)}%"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1), (x1 + text_w, y1 - text_h - 6), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def _detect_image(self, image_path: str) -> None:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        frame = self._flip_frame(self._resize_frame(frame))
        results = self.model(frame)[0]
        output_frame = self._draw_detections(frame.copy(), results)
        cv2.imshow("YOLO Detection", output_frame)
        if self.config.save_output:
            output_path = Path(self.config.output_dir) / f"result_{Path(image_path).stem}.jpg"
            cv2.imwrite(str(output_path), output_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _detect_video(self, video_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")
        writer = self._create_video_writer(cap) if self.config.save_output else None
        prev_time = cv2.getTickCount()
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self._flip_frame(self._resize_frame(frame))
            results = self.model(frame)[0]
            output_frame = self._draw_detections(frame.copy(), results)
            curr_time = cv2.getTickCount()
            fps_display = cv2.getTickFrequency() / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(output_frame, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("YOLO Detection", output_frame)
            if writer is not None:
                writer.write(output_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    def _create_video_writer(self, cap: cv2.VideoCapture) -> Optional[cv2.VideoWriter]:
        fps = cap.get(cv2.CAP_PROP_FPS) or self.config.cam_fps
        ret, test_frame = cap.read()
        if not ret:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        processed = self._flip_frame(self._resize_frame(test_frame))
        height, width = processed.shape[:2]
        output_path = Path(self.config.output_dir) / f"result_{Path(str(self.config.source)).stem}.mp4"
        for codec in ("mp4v", "XVID", "MJPG"):
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
            if writer.isOpened():
                return writer
        return None

    def _detect_camera(self) -> None:
        if self.config.jetson:
            pipeline = (
                "nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM), width={self.config.cam_width}, height={self.config.cam_height}, format=NV12, "
                f"framerate={self.config.cam_fps}/1 ! "
                "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            source = int(self.config.source) if str(self.config.source).isdigit() else self.config.source
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.cam_height)
            cap.set(cv2.CAP_PROP_FPS, self.config.cam_fps)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera source: {self.config.source}")
        writer = None
        recording = False
        start_time = cv2.getTickCount()
        prev_time = cv2.getTickCount()
        recording_start = None
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self._flip_frame(frame)
                results = self.model(frame)[0]
                output_frame = self._draw_detections(frame.copy(), results)
                curr_time = cv2.getTickCount()
                fps_display = cv2.getTickFrequency() / (curr_time - prev_time)
                prev_time = curr_time
                elapsed = (curr_time - start_time) / cv2.getTickFrequency()
                time_text = self._format_duration(elapsed)
                cv2.putText(output_frame, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(output_frame, f"Time: {time_text}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if recording:
                    cv2.putText(output_frame, "REC", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if recording_start is not None:
                        rec_elapsed = (curr_time - recording_start) / cv2.getTickFrequency()
                        cv2.putText(output_frame, self._format_duration(rec_elapsed), (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("YOLO Detection", output_frame)
                if recording and writer is not None and writer.isOpened():
                    writer.write(output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s") and self.config.save_output:
                    filename = f"camera_{time_text.replace(':', '-')}.jpg"
                    cv2.imwrite(str(Path(self.config.output_dir) / filename), output_frame)
                if key == ord("r") and self.config.save_output:
                    if not recording:
                        writer = self._start_recording(output_frame, fps_display)
                        recording = writer is not None
                        recording_start = curr_time if recording else None
                    else:
                        if writer is not None:
                            writer.release()
                        writer = None
                        recording = False
                        recording_start = None
        finally:
            cap.release()
            if writer is not None and writer.isOpened():
                writer.release()
            cv2.destroyAllWindows()

    def _start_recording(self, frame: np.ndarray, fps: Number) -> Optional[cv2.VideoWriter]:
        height, width = frame.shape[:2]
        timestamp = self._format_duration(cv2.getTickCount() / cv2.getTickFrequency()).replace(":", "-")
        prefix = "cameraJETCAM" if self.config.jetson else "cameraUSB"
        output_path = Path(self.config.output_dir) / f"{prefix}_record_{timestamp}.mp4"
        for codec in ("mp4v", "XVID", "MJPG"):
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
            if writer.isOpened():
                return writer
        return None

    @staticmethod
    def _format_duration(seconds: Number) -> str:
        total = int(seconds)
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
