"""Reusable detector utilities built on top of the Ultralytics YOLO models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectorConfig:
    """Configuration values required to construct a :class:`YOLODetector`."""

    model_path: str | Path
    duty: str = "detect"
    source: str = "0"
    imgsz: int | tuple[int, int] = 640
    flip_mode: Optional[int] = 1
    conf_threshold: float = 0.3
    save_output: bool = False
    output_dir: str | Path = "./test_result"
    jetson: bool = False
    cam_width: int = 1280
    cam_height: int = 720
    cam_fps: int = 30


class YOLODetector:
    """Wrapper providing handy helpers to run YOLO detection pipelines."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.model_path = Path(config.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = YOLO(str(self.model_path), verbose=False)
        self.model.overrides["verbose"] = False
        self.labels = self.model.names

        if self.config.save_output:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # A palette of colours for drawing bounding boxes
        self.palette = [
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
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _input_type(self) -> str:
        source = self.config.source
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            return "camera"
        source = str(source).lower()
        if source.endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")):
            return "video"
        if source.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
            return "image"
        return "unknown"

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        target = self.config.imgsz
        if isinstance(target, tuple):
            new_w, new_h = target
        else:
            height, width = frame.shape[:2]
            if height > width:
                new_h = target
                new_w = int(width * target / height)
            else:
                new_w = target
                new_h = int(height * target / width)
        try:
            return cv2.resize(frame, (new_w, new_h))
        except cv2.error:
            return frame

    def _flip_frame(self, frame: np.ndarray) -> np.ndarray:
        flip_mode = self.config.flip_mode
        if flip_mode is None:
            return frame
        if flip_mode == 1:
            return cv2.flip(frame, 1)
        if flip_mode == -1:
            return cv2.flip(frame, 0)
        if flip_mode == 0:
            return cv2.flip(frame, -1)
        return frame

    def _draw_detections(self, frame: np.ndarray, results) -> np.ndarray:
        if not hasattr(results, "boxes") or results.boxes is None:
            return frame

        output = frame.copy()
        for i, box in enumerate(results.boxes):
            xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
            if xyxy.shape[-1] != 4:
                continue

            cls_idx = int(box.cls.item())
            conf = float(box.conf.item())
            if conf < self.config.conf_threshold:
                continue

            colour = self.palette[cls_idx % len(self.palette)]
            label = self.labels.get(cls_idx, f"Class{cls_idx}")
            x1, y1, x2, y2 = xyxy
            h, w = output.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cv2.rectangle(output, (x1, y1), (x2, y2), colour, 2)

            caption = f"{label}: {int(conf * 100)}%"
            (text_w, text_h), baseline = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_y = max(y1, text_h + baseline + 5)
            cv2.rectangle(
                output,
                (x1, text_y - text_h - baseline - 5),
                (x1 + text_w, text_y + baseline),
                colour,
                -1,
            )
            cv2.putText(
                output,
                caption,
                (x1, text_y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return output

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        input_type = self._input_type()
        print("-" * 60)
        print(f"Model      : {self.model_path}")
        print(f"Task       : {self.config.duty}")
        print(f"Source     : {self.config.source}")
        print(f"Input type : {input_type}")
        print(f"Confidence : {self.config.conf_threshold}")
        print(f"Image size : {self.config.imgsz}")
        print("-" * 60)

        if input_type == "image":
            self._run_on_image()
        elif input_type == "video":
            self._run_on_video()
        elif input_type == "camera":
            self._run_on_camera()
        else:
            raise ValueError(f"Unsupported source: {self.config.source}")

    # Individual modes -------------------------------------------------
    def _run_on_image(self) -> None:
        frame = cv2.imread(str(self.config.source))
        if frame is None:
            raise FileNotFoundError(f"Unable to read image: {self.config.source}")

        frame = self._resize_frame(frame)
        frame = self._flip_frame(frame)
        results = self.model(frame)[0]
        output = self._draw_detections(frame, results)
        cv2.imshow("YOLO Detection", output)
        if self.config.save_output:
            output_path = Path(self.config.output_dir) / f"result_{Path(self.config.source).stem}.jpg"
            cv2.imwrite(str(output_path), output)
            print(f"Saved: {output_path}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _run_on_video(self) -> None:
        source = str(self.config.source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video info: {total_frames} frames @ {fps:.2f} fps")

        writer: Optional[cv2.VideoWriter] = None
        save_frames = False
        frames_dir = Path(self.config.output_dir) / f"{Path(source).stem}_frames"
        output_path = Path(self.config.output_dir) / f"result_{Path(source).stem}.mp4"

        if self.config.save_output:
            tmp_ret, tmp_frame = cap.read()
            if not tmp_ret:
                raise RuntimeError("Failed to read first frame for sizing information")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            processed = self._flip_frame(self._resize_frame(tmp_frame))
            height, width = processed.shape[:2]
            for codec in ("mp4v", "XVID", "MJPG"):
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if writer.isOpened():
                    print(f"Using codec: {codec}")
                    break
                writer.release()
                writer = None
            if writer is None:
                save_frames = True
                frames_dir.mkdir(parents=True, exist_ok=True)
                print("Falling back to saving individual frames.")

        frame_count = 0
        prev_time = cv2.getTickCount()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self._flip_frame(self._resize_frame(frame))
            results = self.model(frame)[0]
            output = self._draw_detections(frame, results)

            now = cv2.getTickCount()
            fps_display = cv2.getTickFrequency() / (now - prev_time)
            prev_time = now
            cv2.putText(
                output,
                f"FPS: {fps_display:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            frame_count += 1
            if frame_count % 30 == 0:
                percent = frame_count / max(total_frames, 1) * 100
                print(f"Progress: {frame_count}/{total_frames} ({percent:.1f}%)")

            cv2.imshow("YOLO Detection", output)
            if writer is not None and writer.isOpened():
                writer.write(output)
            elif save_frames:
                cv2.imwrite(str(frames_dir / f"frame_{frame_count:06d}.jpg"), output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer is not None and writer.isOpened():
            writer.release()
            print(f"Video saved to: {output_path}")
        cv2.destroyAllWindows()

    def _run_on_camera(self) -> None:
        source = self.config.source
        if self.config.jetson:
            pipeline = (
                "nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM), width={self.config.cam_width}, height={self.config.cam_height}, "
                f"format=NV12, framerate={self.config.cam_fps}/1 ! "
                "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            camera_index = int(source) if isinstance(source, str) and source.isdigit() else int(source)
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.cam_height)
            cap.set(cv2.CAP_PROP_FPS, self.config.cam_fps)

        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera: {source}")

        print("Camera ready. Press 'q' to quit, 's' to save a frame, 'r' to toggle recording.")
        writer: Optional[cv2.VideoWriter] = None
        recording = False
        recording_start = 0

        start_time = cv2.getTickCount()
        prev_time = start_time

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Unable to read frame from camera")
                break

            frame = self._flip_frame(frame)
            results = self.model(frame)[0]
            output = self._draw_detections(frame, results)

            now = cv2.getTickCount()
            fps_display = cv2.getTickFrequency() / (now - prev_time)
            prev_time = now
            elapsed = (now - start_time) / cv2.getTickFrequency()
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)

            cv2.putText(output, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(
                output,
                f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if recording:
                cv2.putText(output, "REC", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                duration = (now - recording_start) / cv2.getTickFrequency()
                rec_minutes, rec_seconds = divmod(int(duration), 60)
                cv2.putText(
                    output,
                    f"{rec_minutes:02d}:{rec_seconds:02d}",
                    (60, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("YOLO Camera", output)

            if recording and writer is not None and writer.isOpened():
                writer.write(output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and self.config.save_output:
                timestamp = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
                filename = f"camera_{timestamp}.jpg"
                cv2.imwrite(str(Path(self.config.output_dir) / filename), output)
                print(f"Frame saved: {filename}")
            if key == ord("r") and self.config.save_output:
                if not recording:
                    h, w = output.shape[:2]
                    timestamp = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
                    prefix = "JETCAM" if self.config.jetson else "USBCAM"
                    video_path = Path(self.config.output_dir) / f"camera_{prefix}_{timestamp}.mp4"
                    for codec in ("mp4v", "XVID", "MJPG"):
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        writer = cv2.VideoWriter(str(video_path), fourcc, fps_display, (w, h))
                        if writer.isOpened():
                            recording = True
                            recording_start = now
                            print(f"Recording to: {video_path}")
                            break
                        writer.release()
                        writer = None
                    if not recording:
                        print("Unable to start recording")
                else:
                    if writer is not None:
                        writer.release()
                        writer = None
                    recording = False
                    recording_start = 0
                    print("Recording stopped")

        cap.release()
        if writer is not None and writer.isOpened():
            writer.release()
            print("Recording saved")
        cv2.destroyAllWindows()


__all__ = ["YOLODetector", "DetectorConfig"]
