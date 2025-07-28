#!/usr/bin/env python3
import cv2
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

class UniversalCamera:
    def __init__(self, cam_width=640, cam_height=480, cam_fps=30, use_jetcam=False, output_dir="captured_frames"):
        """
        通用摄像头类，支持JetCam和普通OpenCV摄像头
        
        Args:
            cam_width (int): 摄像头宽度
            cam_height (int): 摄像头高度
            cam_fps (int): 帧率
            use_jetcam (bool): 是否使用JetCam (Jetson平台)
            output_dir (str): 输出文件夹路径
        """
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_fps = cam_fps
        self.use_jetcam = use_jetcam
        self.output_dir = output_dir
        self.cap = None
        
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def _setup_jetcam(self):
        """设置JetCam摄像头管道"""
        pipeline = (
            "nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={self.cam_width}, height={self.cam_height}, format=NV12, framerate={self.cam_fps}/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "appsink"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    def _setup_opencv_cam(self, camera_id=0):
        """设置普通OpenCV摄像头"""
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
            cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
        return cap
    
    def initialize_camera(self, camera_id=0):
        """初始化摄像头"""
        try:
            if self.use_jetcam:
                print("正在初始化JetCam摄像头...")
                self.cap = self._setup_jetcam()
            else:
                print(f"正在初始化OpenCV摄像头 (ID: {camera_id})...")
                self.cap = self._setup_opencv_cam(camera_id)
            
            if not self.cap.isOpened():
                raise RuntimeError("无法打开摄像头")
            
            print(f"摄像头初始化成功! 分辨率: {self.cam_width}x{self.cam_height}, 帧率: {self.cam_fps}")
            return True
            
        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            return False
    
    def capture_frames(self, total_frames, interval=0.1, show_preview=True):
        """
        连续拍摄指定数量的帧
        
        Args:
            total_frames (int): 总拍摄帧数
            interval (float): 拍摄间隔(秒)
            show_preview (bool): 是否显示预览窗口
        """
        if self.cap is None or not self.cap.isOpened():
            print("摄像头未初始化或已关闭!")
            return False
        
        print(f"开始拍摄 {total_frames} 帧，间隔 {interval} 秒")
        print(f"图片将保存到: {self.output_dir}")
        
        captured_count = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            while captured_count < total_frames:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头帧!")
                    break
                
                # 保存图片
                filename = f"frame_{timestamp}_{captured_count:06d}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                success = cv2.imwrite(filepath, frame)
                if success:
                    captured_count += 1
                    print(f"已拍摄: {captured_count}/{total_frames} - {filename}")
                else:
                    print(f"保存失败: {filename}")
                
                # 显示预览窗口
                if show_preview:
                    cv2.imshow('Camera Preview', frame)
                    # 按 'q' 键退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户中断拍摄")
                        break
                
                # 等待指定间隔
                if interval > 0:
                    time.sleep(interval)
            
            print(f"拍摄完成! 共保存 {captured_count} 张图片")
            return True
            
        except KeyboardInterrupt:
            print(f"\n拍摄被中断! 已保存 {captured_count} 张图片")
            return False
        
        finally:
            if show_preview:
                cv2.destroyAllWindows()
    
    def capture_continuous(self, interval=1.0, show_preview=True):
        """
        连续拍摄模式，按Ctrl+C停止
        
        Args:
            interval (float): 拍摄间隔(秒)
            show_preview (bool): 是否显示预览窗口
        """
        if self.cap is None or not self.cap.isOpened():
            print("摄像头未初始化或已关闭!")
            return False
        
        print(f"连续拍摄模式，间隔 {interval} 秒")
        print("按 Ctrl+C 停止拍摄")
        print(f"图片将保存到: {self.output_dir}")
        
        captured_count = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头帧!")
                    break
                
                # 保存图片
                filename = f"continuous_{timestamp}_{captured_count:06d}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                success = cv2.imwrite(filepath, frame)
                if success:
                    captured_count += 1
                    print(f"已拍摄: {captured_count} - {filename}")
                else:
                    print(f"保存失败: {filename}")
                
                # 显示预览窗口
                if show_preview:
                    cv2.imshow('Camera Preview', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户退出拍摄")
                        break
                
                # 等待指定间隔
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n拍摄停止! 共保存 {captured_count} 张图片")
            
        finally:
            if show_preview:
                cv2.destroyAllWindows()
    
    def release(self):
        """释放摄像头资源"""
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            print("摄像头资源已释放")

def main():
    parser = argparse.ArgumentParser(description="通用摄像头拍摄工具")
    parser.add_argument("--frames", "-f", type=int, default=10, help="拍摄帧数 (默认: 10)")
    parser.add_argument("--width", "-W", type=int, default=640, help="图像宽度 (默认: 640)")
    parser.add_argument("--height", "-H", type=int, default=480, help="图像高度 (默认: 480)")
    parser.add_argument("--fps", type=int, default=30, help="帧率 (默认: 30)")
    parser.add_argument("--interval", "-i", type=float, default=0.5, help="拍摄间隔秒数 (默认: 0.5)")
    parser.add_argument("--output", "-o", type=str, default="captured_frames", help="输出文件夹 (默认: captured_frames)")
    parser.add_argument("--jetcam", action="store_true", help="使用JetCam (Jetson平台)")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头ID (默认: 0)")
    parser.add_argument("--continuous", "-c", action="store_true", help="连续拍摄模式")
    parser.add_argument("--no-preview", action="store_true", help="不显示预览窗口")
    
    args = parser.parse_args()
    
    # 创建摄像头实例
    camera = UniversalCamera(
        cam_width=args.width,
        cam_height=args.height,
        cam_fps=args.fps,
        use_jetcam=args.jetcam,
        output_dir=args.output
    )
    
    # 初始化摄像头
    if not camera.initialize_camera(args.camera_id):
        return
    
    try:
        if args.continuous:
            # 连续拍摄模式
            camera.capture_continuous(
                interval=args.interval,
                show_preview=not args.no_preview
            )
        else:
            # 固定帧数拍摄模式
            camera.capture_frames(
                total_frames=args.frames,
                interval=args.interval,
                show_preview=not args.no_preview
            )
    
    finally:
        camera.release()

if __name__ == "__main__":
    main()
