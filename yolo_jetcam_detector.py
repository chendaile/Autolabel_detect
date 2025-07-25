import cv2
import argparse
import os
import sys
import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# JetCam 支持
try:
    from jetcam.csi_camera import CSICamera
    from jetcam.usb_camera import USBCamera
    JETCAM_AVAILABLE = True
    print("JetCam 库已加载")
except ImportError:
    JETCAM_AVAILABLE = False
    print("JetCam 库未找到，将使用标准OpenCV摄像头")


class CameraInterface:
    """摄像头接口基类"""
    def __init__(self):
        self.is_opened = False
    
    def open(self):
        raise NotImplementedError
    
    def read(self):
        raise NotImplementedError
    
    def release(self):
        raise NotImplementedError
    
    def set_resolution(self, width, height):
        pass


class OpenCVCamera(CameraInterface):
    """OpenCV标准摄像头接口"""
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.cap = None
    
    def open(self):
        try:
            self.cap = cv2.VideoCapture(int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source)
            self.is_opened = self.cap.isOpened()
            return self.is_opened
        except Exception as e:
            print(f"OpenCV摄像头打开失败: {e}")
            return False
    
    def read(self):
        if self.cap and self.is_opened:
            ret, frame = self.cap.read()
            return ret, frame
        return False, None
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.is_opened = False
    
    def set_resolution(self, width, height):
        if self.cap and self.is_opened:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


class JetCamCSI(CameraInterface):
    """JetCam CSI摄像头接口"""
    def __init__(self, width=1280, height=720, fps=30):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
    
    def open(self):
        if not JETCAM_AVAILABLE:
            return False
        try:
            self.camera = CSICamera(width=self.width, height=self.height, capture_fps=self.fps)
            self.camera.running = True
            time.sleep(1)  # 等待摄像头初始化
            self.is_opened = True
            return True
        except Exception as e:
            print(f"JetCam CSI摄像头打开失败: {e}")
            return False
    
    def read(self):
        if self.camera and self.is_opened:
            try:
                frame = self.camera.value
                if frame is not None:
                    # JetCam返回RGB格式，转换为BGR供OpenCV使用
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    return True, frame
            except Exception as e:
                print(f"读取JetCam CSI帧失败: {e}")
        return False, None
    
    def release(self):
        if self.camera:
            try:
                self.camera.running = False
                self.is_opened = False
            except Exception as e:
                print(f"释放JetCam CSI资源失败: {e}")


class JetCamUSB(CameraInterface):
    """JetCam USB摄像头接口"""
    def __init__(self, device_id=0, width=1280, height=720):
        super().__init__()
        self.device_id = device_id
        self.width = width
        self.height = height
        self.camera = None
    
    def open(self):
        if not JETCAM_AVAILABLE:
            return False
        try:
            self.camera = USBCamera(width=self.width, height=self.height, capture_device=self.device_id)
            self.camera.running = True
            time.sleep(1)  # 等待摄像头初始化
            self.is_opened = True
            return True
        except Exception as e:
            print(f"JetCam USB摄像头打开失败: {e}")
            return False
    
    def read(self):
        if self.camera and self.is_opened:
            try:
                frame = self.camera.value
                if frame is not None:
                    # JetCam返回RGB格式，转换为BGR供OpenCV使用
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    return True, frame
            except Exception as e:
                print(f"读取JetCam USB帧失败: {e}")
        return False, None
    
    def release(self):
        if self.camera:
            try:
                self.camera.running = False
                self.is_opened = False
            except Exception as e:
                print(f"释放JetCam USB资源失败: {e}")


class CameraFactory:
    """摄像头工厂类"""
    @staticmethod
    def create_camera(camera_type, source=0, **kwargs):
        """
        创建摄像头接口
        
        Args:
            camera_type: 摄像头类型 ('opencv', 'jetcam_csi', 'jetcam_usb')
            source: 摄像头源
            **kwargs: 其他参数
        """
        if camera_type == 'opencv':
            return OpenCVCamera(source)
        elif camera_type == 'jetcam_csi':
            return JetCamCSI(**kwargs)
        elif camera_type == 'jetcam_usb':
            device_id = int(source) if isinstance(source, str) and source.isdigit() else source
            return JetCamUSB(device_id=device_id, **kwargs)
        else:
            raise ValueError(f"不支持的摄像头类型: {camera_type}")


class VideoRecorder:
    """视频录制管理器"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.video_writer = None
        self.is_recording = False
        self.recording_start_time = None
        self.current_path = None
    
    def start_recording(self, frame_width, frame_height, fps=20.0, timestamp=""):
        """开始录制"""
        if self.is_recording:
            return False
        
        filename = f"record_{timestamp}.mp4" if timestamp else "record.mp4"
        self.current_path = os.path.join(self.output_dir, filename)
        
        # 尝试不同的编码器
        for codec in ['mp4v', 'XVID', 'MJPG']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self.video_writer = cv2.VideoWriter(self.current_path, fourcc, fps, (frame_width, frame_height))
                if self.video_writer.isOpened():
                    self.is_recording = True
                    self.recording_start_time = cv2.getTickCount()
                    print(f"开始录制到: {self.current_path} (编码器: {codec})")
                    return True
                else:
                    self.video_writer.release()
                    self.video_writer = None
            except Exception as e:
                print(f"编码器 {codec} 初始化失败: {e}")
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
        
        print("警告: 无法开始录制")
        return False
    
    def write_frame(self, frame):
        """写入帧"""
        if self.is_recording and self.video_writer and self.video_writer.isOpened():
            self.video_writer.write(frame)
    
    def stop_recording(self):
        """停止录制"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        if self.is_recording:
            print(f"录制已停止并保存到: {self.current_path}")
        
        self.is_recording = False
        self.recording_start_time = None
        self.current_path = None
    
    def get_recording_duration(self):
        """获取录制时长"""
        if self.is_recording and self.recording_start_time:
            current_time = cv2.getTickCount()
            duration = (current_time - self.recording_start_time) / cv2.getTickFrequency()
            return duration
        return 0


class YOLODetector:
    def __init__(self, model_path, duty='detect', source=0, imgsz=640, flip_mode=1, 
                 conf_threshold=0.3, save_output=False, output_dir='./test_result',
                 camera_type='opencv', camera_width=1280, camera_height=720, camera_fps=30):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型路径
            duty: 任务类型 ('detect', 'segment', 'classify', 'pose')
            source: 输入源 (0为摄像头, 或视频/图片路径)
            imgsz: 输出图像尺寸
            flip_mode: 翻转模式 (0=不翻转, 1=水平翻转, -1=垂直翻转)
            conf_threshold: 置信度阈值
            save_output: 是否保存输出
            output_dir: 输出目录
            camera_type: 摄像头类型 ('opencv', 'jetcam_csi', 'jetcam_usb')
            camera_width: 摄像头宽度
            camera_height: 摄像头高度
            camera_fps: 摄像头帧率
        """
        self.model_path = model_path
        self.duty = duty
        self.source = source
        self.imgsz = imgsz
        self.flip_mode = flip_mode
        self.conf_threshold = conf_threshold
        self.save_output = save_output
        self.output_dir = output_dir
        self.camera_type = camera_type
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        
        # 加载模型
        self._load_model()
        
        # 创建输出目录
        if self.save_output:
            os.makedirs(output_dir, exist_ok=True)
        
        # 视频录制器
        self.recorder = VideoRecorder(output_dir) if save_output else None
        
        # 预定义颜色
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
            (255, 255, 0), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0),
        ]
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            self.model = YOLO(self.model_path, verbose=False)
            self.labels = self.model.names
            print(f"模型加载成功: {self.model_path}")
            print(f"检测类别: {list(self.labels.values())}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)
    
    def get_input_type(self):
        """判断输入类型"""
        if self.camera_type != 'opencv':
            return 'jetcam'
        elif isinstance(self.source, int) or self.source.isdigit():
            return 'camera'
        elif self.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            return 'video'
        elif self.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
            return 'image'
        else:
            return 'unknown'
    
    def resize_frame(self, frame):
        """调整帧大小"""
        if isinstance(self.imgsz, int):
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                print(f"警告: 帧尺寸异常 {w}x{h}")
                return frame
            
            # 按长边缩放，保持宽高比
            if h > w:
                new_h = self.imgsz
                new_w = int(w * self.imgsz / h)
            else:
                new_w = self.imgsz
                new_h = int(h * self.imgsz / w)
            
            new_w = max(1, new_w)
            new_h = max(1, new_h)
        else:  # tuple (width, height)
            new_w, new_h = self.imgsz
            new_w = max(1, new_w)
            new_h = max(1, new_h)
        
        try:
            return cv2.resize(frame, (new_w, new_h))
        except cv2.error as e:
            print(f"调整帧尺寸失败: {e}")
            return frame
    
    def flip_frame(self, frame):
        """翻转帧"""
        if self.flip_mode == 1:
            return cv2.flip(frame, 1)  # 水平翻转
        elif self.flip_mode == -1:
            return cv2.flip(frame, 0)  # 垂直翻转
        elif self.flip_mode == 0:
            return cv2.flip(frame, -1)  # 同时翻转
        else:
            return frame
    
    def draw_detections(self, frame, results):
        """绘制检测结果"""
        if not results or not hasattr(results, 'boxes') or results.boxes is None:
            return frame
        
        for box in results.boxes:
            # 获取边界框坐标
            xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
            if len(xyxy.shape) == 0 or len(xyxy) != 4:
                continue
                
            # 获取类别和置信度
            cls_idx = int(box.cls.item())
            conf = box.conf.item()
            
            if conf > self.conf_threshold:
                # 获取类别名称和颜色
                class_name = self.labels.get(cls_idx, f'Class{cls_idx}')
                color = self.colors[cls_idx % len(self.colors)]
                
                # 确保坐标在帧范围内
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = xyxy
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{class_name}: {int(conf*100)}%"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # 标签位置
                label_y1 = max(label_size[1] + 10, y1)
                label_x2 = min(w, x1 + label_size[0])
                label_y2 = max(0, label_y1 - label_size[1] - 10)
                
                # 绘制标签背景和文字
                cv2.rectangle(frame, (x1, label_y2), (label_x2, label_y1), color, -1)
                cv2.putText(frame, label, (x1, label_y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def add_info_overlay(self, frame, fps, duration=None, camera_info="", recording_duration=0):
        """添加信息覆盖层"""
        # FPS显示
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 时长显示
        if duration is not None:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            cv2.putText(frame, f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}", (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 摄像头信息
        if camera_info:
            cv2.putText(frame, camera_info, (10, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 录制状态
        if self.recorder and self.recorder.is_recording:
            cv2.putText(frame, "REC", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if recording_duration > 0:
                rec_minutes = int(recording_duration // 60)
                rec_seconds = int(recording_duration % 60)
                cv2.putText(frame, f"{rec_minutes:02d}:{rec_seconds:02d}", (60, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def detect_image(self):
        """处理单张图像"""
        print(f"处理图像: {self.source}")
        
        frame = cv2.imread(self.source)
        if frame is None:
            print(f"无法读取图像: {self.source}")
            return
        
        # 预处理
        frame = self.resize_frame(frame)
        frame = self.flip_frame(frame)
        
        # 检测
        results = self.model(frame)[0]
        output_frame = self.draw_detections(frame.copy(), results)
        
        # 显示和保存
        cv2.imshow("YOLO Detection - Press 'q' to quit", output_frame)
        
        if self.save_output:
            output_path = os.path.join(self.output_dir, f"result_{Path(self.source).stem}.jpg")
            cv2.imwrite(output_path, output_frame)
            print(f"结果已保存到: {output_path}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def detect_video(self):
        """处理视频文件"""
        print(f"处理视频: {self.source}")
        
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"无法打开视频: {self.source}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
        
        # 视频写入器设置
        video_writer = None
        if self.save_output:
            video_writer = self._setup_video_writer(cap, fps)
        
        frame_count = 0
        prev_time = cv2.getTickCount()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                frame = self.resize_frame(frame)
                frame = self.flip_frame(frame)
                results = self.model(frame)[0]
                output_frame = self.draw_detections(frame.copy(), results)
                
                # 计算FPS
                curr_time = cv2.getTickCount()
                fps_display = cv2.getTickFrequency() / (curr_time - prev_time)
                prev_time = curr_time
                
                # 添加信息覆盖
                output_frame = self.add_info_overlay(output_frame, fps_display)
                
                # 显示进度
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"处理进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
                
                cv2.imshow("YOLO Detection - Press 'q' to quit", output_frame)
                
                # 保存帧
                if video_writer and video_writer.isOpened():
                    video_writer.write(output_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
                print("视频处理完成")
            cv2.destroyAllWindows()
    
    def detect_camera(self):
        """处理摄像头输入（包括JetCam）"""
        camera_info = f"{self.camera_type.upper()}"
        if self.camera_type == 'jetcam_csi':
            camera_info = "JetCam-CSI"
        elif self.camera_type == 'jetcam_usb':
            camera_info = "JetCam-USB"
        
        print(f"使用摄像头: {camera_info}")
        
        # 创建摄像头接口
        camera_kwargs = {
            'width': self.camera_width,
            'height': self.camera_height
        }
        if self.camera_type.startswith('jetcam'):
            camera_kwargs['fps'] = self.camera_fps
        
        camera = CameraFactory.create_camera(self.camera_type, self.source, **camera_kwargs)
        
        if not camera.open():
            print(f"无法打开摄像头，尝试使用标准OpenCV摄像头")
            camera = CameraFactory.create_camera('opencv', self.source)
            if not camera.open():
                print("所有摄像头接口都无法打开")
                return
            camera_info = "OpenCV (fallback)"
        
        # 设置分辨率（仅对OpenCV有效）
        camera.set_resolution(self.camera_width, self.camera_height)
        
        print(f"摄像头已就绪 ({camera_info})，按 'q' 退出，按 's' 保存当前帧，按 'r' 开始/停止录制")
        
        start_time = cv2.getTickCount()
        prev_time = cv2.getTickCount()
        
        try:
            while True:
                ret, frame = camera.read()
                if not ret or frame is None:
                    print("无法从摄像头读取帧")
                    time.sleep(0.01)
                    continue
                
                # 处理帧
                frame = self.resize_frame(frame)
                frame = self.flip_frame(frame)
                results = self.model(frame)[0]
                output_frame = self.draw_detections(frame.copy(), results)
                
                # 计算时间信息
                curr_time = cv2.getTickCount()
                fps_display = cv2.getTickFrequency() / (curr_time - prev_time)
                total_duration = (curr_time - start_time) / cv2.getTickFrequency()
                recording_duration = self.recorder.get_recording_duration() if self.recorder else 0
                prev_time = curr_time
                
                # 添加信息覆盖
                output_frame = self.add_info_overlay(output_frame, fps_display, 
                                                   total_duration, camera_info, recording_duration)
                
                # 显示结果
                window_title = f"YOLO Detection ({camera_info}) - Press 'q' to quit, 's' to save, 'r' to record"
                cv2.imshow(window_title, output_frame)
                
                # 录制帧
                if self.recorder and self.recorder.is_recording:
                    self.recorder.write_frame(output_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and self.save_output:
                    self._save_frame(output_frame, total_duration, camera_info.lower())
                elif key == ord('r') and self.save_output and self.recorder:
                    self._toggle_recording(output_frame, total_duration)
        
        finally:
            camera.release()
            if self.recorder:
                self.recorder.stop_recording()
            cv2.destroyAllWindows()
    
    def _setup_video_writer(self, cap, fps):
        """设置视频写入器"""
        ret, test_frame = cap.read()
        if not ret:
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开始
        
        processed_frame = self.resize_frame(test_frame)
        processed_frame = self.flip_frame(processed_frame)
        height, width = processed_frame.shape[:2]
        
        output_path = os.path.join(self.output_dir, f"result_{Path(self.source).stem}.mp4")
        
        for codec in ['mp4v', 'XVID', 'MJPG']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if writer.isOpened():
                    print(f"视频将保存到: {output_path} (编码器: {codec})")
                    return writer
                writer.release()
            except Exception as e:
                print(f"编码器 {codec} 初始化失败: {e}")
        
        return None
    
    def _save_frame(self, frame, duration, prefix):
        """保存当前帧"""
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        save_path = os.path.join(self.output_dir, f"{prefix}_{hours:02d}h{minutes:02d}m{seconds:02d}s.jpg")
        cv2.imwrite(save_path, frame)
        print(f"帧已保存: {save_path}")
    
    def _toggle_recording(self, frame, duration):
        """切换录制状态"""
        if not self.recorder.is_recording:
            # 开始录制
            frame_height, frame_width = frame.shape[:2]
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            timestamp = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
            self.recorder.start_recording(frame_width, frame_height, 20.0, timestamp)
        else:
            # 停止录制
            self.recorder.stop_recording()
    
    def run(self):
        """运行检测"""
        input_type = self.get_input_type()
        
        print(f"任务类型: {self.duty}")
        print(f"输入类型: {input_type}")
        print(f"输出尺寸: {self.imgsz}")
        print(f"置信度阈值: {self.conf_threshold}")
        if input_type in ['camera', 'jetcam']:
            print(f"摄像头类型: {self.camera_type}")
        print("-" * 50)
        
        if input_type == 'image':
            self.detect_image()
        elif input_type == 'video':
            self.detect_video()
        elif input_type in ['camera', 'jetcam']:
            self.detect_camera()
        else:
            print(f"不支持的输入类型: {self.source}")


def main():
    parser = argparse.ArgumentParser(description='Universal YOLO Detection System with JetCam Support')

    # 基础参数
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='YOLO模型路径')
    parser.add_argument('--source', '-s', type=str, default='0',
                        help='输入源: 摄像头ID(如0), 视频文件路径, 或图像文件路径 (默认: 0)')
    parser.add_argument('--duty', '-d', type=str, default='detect',
                        choices=['detect', 'segment', 'classify', 'pose'],
                        help='任务类型 (默认: detect)')
    parser.add_argument('--imgsz', '-i', type=int, default=640,
                        help='输出图像尺寸 (默认: 640)')
    parser.add_argument('--flip', '-f', type=int, default=1,
                        choices=[1, -1, 0, 2],
                        help='翻转模式: 1=水平翻转, -1=垂直翻转, 0=双向翻转, 2=不翻转')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='置信度阈值 (默认: 0.3)')
    parser.add_argument('--save', action='store_true',
                        help='是否保存输出结果')
    parser.add_argument('--output', type=str, default='./test_result',
                        help='输出目录 (默认: ./test_result)')
    parser.add_argument('--camera_type', type=str, default='opencv',
                        choices=['opencv', 'jetcam_csi', 'jetcam_usb'],
                        help='摄像头类型 (默认: opencv)')
    parser.add_argument('--camera_width', type=int, default=1280,
                        help='摄像头宽度 (默认: 1280)')
    parser.add_argument('--camera_height', type=int, default=720,
                        help='摄像头高度 (默认: 720)')
    parser.add_argument('--camera_fps', type=int, default=30,
                        help='摄像头帧率 (默认: 30)')

    args = parser.parse_args()

    # 创建检测器并运行
    detector = YOLODetector(
        model_path=args.model,
        duty=args.duty,
        source=args.source,
        imgsz=args.imgsz,
        flip_mode=(args.flip if args.flip != 2 else None),  # 若选择2，则不翻转
        conf_threshold=args.conf,
        save_output=args.save,
        output_dir=args.output,
        camera_type=args.camera_type,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps
    )
    detector.run()


if __name__ == '__main__':
    main()
