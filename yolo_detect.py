import cv2
import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_path, duty='detect', source=0, imgsz=640,
                flip_mode=1, conf_threshold=0.3, save_output=False,
                output_dir='./test_result', jetson=False, cam_width = 1280,
                cam_height=720, cam_fps=30):
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
        """
        self.model_path = model_path
        self.duty = duty
        self.source = source
        self.imgsz = imgsz
        self.flip_mode = flip_mode
        self.conf_threshold = conf_threshold
        self.save_output = save_output
        self.output_dir = output_dir
        self.jetson = jetson
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_fps = cam_fps
        
        # 加载模型
        try:
            self.model = YOLO(model_path, verbose=False)
            self.labels = self.model.names
            print(f"模型加载成功: {model_path}")
            print(f"检测类别: {list(self.labels.values())}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)
        
        # 创建输出目录
        if self.save_output:
            os.makedirs(output_dir, exist_ok=True)
        
        # 预定义颜色
        self.colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (0, 255, 255),  # 青色
            (255, 0, 255),  # 洋红色
            (255, 255, 0),  # 黄色
            (128, 0, 128),  # 紫色
            (255, 165, 0),  # 橙色
            (0, 128, 128),  # 蓝绿色
            (128, 128, 0),  # 橄榄色
        ]
    
    def get_input_type(self):
        """判断输入类型"""
        if isinstance(self.source, int) or self.source.isdigit():
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
            
        else:  # tuple (width, height)
            new_w, new_h = self.imgsz
        
        try:
            return cv2.resize(frame, (new_w, new_h))
        except cv2.error as e:
            print(f"调整帧尺寸失败: {e}")
            print(f"原始尺寸: {frame.shape}, 目标尺寸: ({new_w}, {new_h})")
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
        
        for i, box in enumerate(results.boxes):
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
                
                # 确保标签位置在帧范围内
                label_y1 = max(label_size[1] + 10, y1)
                label_x2 = min(w, x1 + label_size[0])
                label_y2 = max(0, label_y1 - label_size[1] - 10)
                
                # 绘制标签背景
                cv2.rectangle(frame, 
                            (x1, label_y2),
                            (label_x2, label_y1), 
                            color, -1)
                
                # 绘制标签文字
                cv2.putText(frame, label, 
                          (x1, label_y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def detect_image(self):
        """处理单张图像"""
        print(f"处理图像: {self.source}")
        
        # 读取图像
        frame = cv2.imread(self.source)
        if frame is None:
            print(f"无法读取图像: {self.source}")
            return
        
        # 调整大小和翻转
        frame = self.resize_frame(frame)
        frame = self.flip_frame(frame)
        
        # 进行检测
        results = self.model(frame)[0]
        
        # 绘制结果
        output_frame = self.draw_detections(frame.copy(), results)
        
        # 显示结果
        cv2.imshow("YOLO Detection - Press 'q' to quit", output_frame)
        
        # 保存结果
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
        
        # 视频写入器
        video_writer = None
        expected_width, expected_height = 0, 0
        
        if self.save_output:
            # 获取第一帧来确定输出尺寸
            ret, test_frame = cap.read()
            if not ret:
                print("无法读取视频第一帧")
                return
            
            # 重置视频到开始位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # 处理第一帧以获得实际输出尺寸
            try:
                processed_frame = self.resize_frame(test_frame)
                processed_frame = self.flip_frame(processed_frame)
                expected_height, expected_width = processed_frame.shape[:2]
                
                print(f"输出帧尺寸: {expected_width}x{expected_height}")
                
                # 尝试不同的编码器
                output_path = os.path.join(self.output_dir, f"result_{Path(self.source).stem}.mp4")
                
                for codec in ['mp4v', 'XVID', 'MJPG']:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (expected_width, expected_height))
                        
                        if video_writer.isOpened():
                            print(f"使用编码器: {codec}")
                            break
                        else:
                            video_writer.release()
                            video_writer = None
                    except Exception as e:
                        print(f"编码器 {codec} 初始化失败: {e}")
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                        continue
                
                if video_writer is None:
                    print("警告: 无法创建视频写入器，将保存为图像序列")
                    self.save_as_images = True
                    self.image_save_dir = os.path.join(self.output_dir, f"{Path(self.source).stem}_frames")
                    os.makedirs(self.image_save_dir, exist_ok=True)
                else:
                    self.save_as_images = False
                    
            except Exception as e:
                print(f"处理第一帧时出错: {e}")
                self.save_output = False
        
        frame_count = 0
        prev_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 调整大小和翻转
            frame = self.resize_frame(frame)
            frame = self.flip_frame(frame)
            
            # 进行检测
            results = self.model(frame)[0]
            
            # 绘制结果
            output_frame = self.draw_detections(frame.copy(), results)
            
            # 计算并显示帧率
            curr_time = cv2.getTickCount()
            fps_display = cv2.getTickFrequency() / (curr_time - prev_time)
            prev_time = curr_time
            
            # 添加帧率显示
            cv2.putText(output_frame, f"FPS: {fps_display:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示进度
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"处理进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            # 显示结果
            cv2.imshow("YOLO Detection - Press 'q' to quit", output_frame)
            
            # 保存视频帧
            if self.save_output:
                if video_writer and video_writer.isOpened():
                    current_height, current_width = output_frame.shape[:2]
                    
                    # 检查尺寸是否匹配
                    if current_height == expected_height and current_width == expected_width:
                        video_writer.write(output_frame)
                    else:
                        # 只有在尺寸有效时才调整
                        if expected_width > 0 and expected_height > 0:
                            output_frame_resized = cv2.resize(output_frame, (expected_width, expected_height))
                            video_writer.write(output_frame_resized)
                        else:
                            # 如果期望尺寸无效，使用当前帧尺寸重新创建写入器
                            video_writer.release()
                            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (current_width, current_height))
                            expected_width, expected_height = current_width, current_height
                            if video_writer.isOpened():
                                video_writer.write(output_frame)
                            
                elif hasattr(self, 'save_as_images') and self.save_as_images:
                    img_path = os.path.join(self.image_save_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(img_path, output_frame)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if video_writer and video_writer.isOpened():
            video_writer.release()
            print(f"视频结果已保存到: {output_path}")
        elif hasattr(self, 'save_as_images') and self.save_as_images:
            print(f"图像序列已保存到: {self.image_save_dir}")
        cv2.destroyAllWindows()
    
    def detect_camera(self):
        """处理摄像头输入"""
        if self.jetson:
            print(f"打开jetson摄像头: {self.source}")
        else:
            print(f"打开普通USB摄像头: {self.source}")
        
        # 打开摄像头
        if not self.jetson:
            cap = cv2.VideoCapture(int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source)
            # 设置摄像头分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        else:
            pipeline = (
                "nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM), width={self.cam_width}, height={self.cam_height}, format=NV12, framerate={self.cam_fps}/1 ! "
                "nvvidconv ! "
                "video/x-raw, format=BGRx ! "
                "videoconvert ! "
                "appsink"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print(f"无法打开摄像头: {self.source}")
            return
        
        print("摄像头已就绪，按 'q' 退出，按 's' 保存当前帧，按 'r' 开始/停止录制")
        
        start_time = cv2.getTickCount()
        prev_time = cv2.getTickCount()
        
        # 视频录制相关变量
        video_writer = None
        is_recording = False
        recording_start_time = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法从摄像头读取帧")
                break
            
            # 调整大小和翻转
            frame = self.resize_frame(frame)
            frame = self.flip_frame(frame)
            
            # 进行检测
            results = self.model(frame)[0]
            
            # 绘制结果
            output_frame = self.draw_detections(frame.copy(), results)
            
            # 计算帧率和总时长
            curr_time = cv2.getTickCount()
            fps_display = cv2.getTickFrequency() / (curr_time - prev_time)
            total_duration = (curr_time - start_time) / cv2.getTickFrequency()
            prev_time = curr_time
            
            # 格式化时长显示（时:分:秒）
            hours = int(total_duration // 3600)
            minutes = int((total_duration % 3600) // 60)
            seconds = int(total_duration % 60)
            
            # 添加帧率和时长信息
            cv2.putText(output_frame, f"FPS: {fps_display:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}", (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示录制状态
            if is_recording:
                cv2.putText(output_frame, "REC", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # 录制时长
                if recording_start_time:
                    rec_duration = (curr_time - recording_start_time) / cv2.getTickFrequency()
                    rec_minutes = int(rec_duration // 60)
                    rec_seconds = int(rec_duration % 60)
                    cv2.putText(output_frame, f"{rec_minutes:02d}:{rec_seconds:02d}", (60, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 显示结果
            cv2.imshow("YOLO Detection - Press 'q' to quit, 's' to save, 'r' to record", output_frame)
            
            # 如果正在录制，保存帧
            if is_recording and video_writer and video_writer.isOpened():
                video_writer.write(output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and self.save_output:
                # 保存当前帧
                save_path = os.path.join(self.output_dir, f"camera_{hours:02d}h{minutes:02d}m{seconds:02d}s.jpg")
                cv2.imwrite(save_path, output_frame)
                print(f"帧已保存: {save_path}")
            elif key == ord('r') and self.save_output:
                # 开始/停止录制
                if not is_recording:
                    # 开始录制
                    frame_height, frame_width = output_frame.shape[:2]
                    timestamp = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
                    video_path = os.path.join(self.output_dir, f"camera_record_{timestamp}.mp4")
                    
                    # 尝试创建视频写入器
                    for codec in ['mp4v', 'XVID', 'MJPG']:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*codec)
                            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
                            if video_writer.isOpened():
                                is_recording = True
                                recording_start_time = curr_time
                                print(f"开始录制到: {video_path}")
                                break
                            else:
                                video_writer.release()
                                video_writer = None
                        except:
                            if video_writer:
                                video_writer.release()
                                video_writer = None
                            continue
                    
                    if not is_recording:
                        print("警告: 无法开始录制")
                else:
                    # 停止录制
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    is_recording = False
                    recording_start_time = None
                    print("录制已停止")
        
        cap.release()
        if video_writer and video_writer.isOpened():
            video_writer.release()
            print("录制已结束并保存")
        cv2.destroyAllWindows()
    
    def run(self):
        """运行检测"""
        input_type = self.get_input_type()
        
        print(f"任务类型: {self.duty}")
        print(f"输入类型: {input_type}")
        print(f"输出尺寸: {self.imgsz}")
        print(f"置信度阈值: {self.conf_threshold}")
        print("-" * 50)
        
        if input_type == 'image':
            self.detect_image()
        elif input_type == 'video':
            self.detect_video()
        elif input_type == 'camera':
            self.detect_camera()
        else:
            print(f"不支持的输入类型: {self.source}")

def main():
    parser = argparse.ArgumentParser(description='Universal YOLO Detection System')
    
    # 必需参数
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='YOLO模型路径')
    
    # 可选参数
    parser.add_argument('--source', '-s', type=str, default='0',
                       help='输入源: 摄像头ID(如0), 视频文件路径, 或图像文件路径 (默认: 0)')
    
    parser.add_argument('--duty', '-d', type=str, default='detect',
                       choices=['detect', 'segment', 'classify', 'pose'],
                       help='任务类型 (默认: detect)(暂时detect这一个任务可用)')
    
    parser.add_argument('--imgsz', '-i', type=int, default=640,
                       help='输出图像尺寸 (默认: 640)')
    
    parser.add_argument('--flip', '-f', type=int, default=1,
                       choices=[1, -1, 0, 2],  # 2表示不翻转
                       help='翻转模式: 1=水平翻转, -1=垂直翻转, 0=同时翻转, 2=不翻转 (默认: 1)')
    
    parser.add_argument('--conf', '-c', type=float, default=0.3,
                       help='置信度阈值 (默认: 0.3)')
    
    parser.add_argument('--save', action='store_true',
                       help='保存检测结果')
    
    parser.add_argument('--output', '-o', type=str, default='./test_result',
                       help='输出目录 (默认: ./test_result)')
    
    parser.add_argument('--jetson', '-j', action='store_true',
                       help='使用Jetson CSI摄像头')
    
    parser.add_argument('--cam_width', '-w', type=int, default=1280,
                       help='CSI摄像头宽度 (默认: 3264)')
    
    parser.add_argument('--cam_height', '-h', type=int, default=720,
                       help='CSI摄像头高度 (默认: 2464)')
    
    parser.add_argument('--cam_fps', '-fps', type=int, default=30,
                       help='CSI摄像头帧率 (默认: 21)')
    
    args = parser.parse_args()
    
    # 处理翻转模式
    flip_mode = None if args.flip == 2 else args.flip
    
    # 创建检测器
    detector = YOLODetector(
        model_path=args.model,
        duty=args.duty,
        source=args.source,
        imgsz=args.imgsz,
        flip_mode=flip_mode,
        conf_threshold=args.conf,
        save_output=args.save,
        output_dir=args.output,
        jetson = args.jetson,
        cam_width = args.cam_width,
        cam_height = args.cam_height,
        cam_fps = args.cam_fps
    )
    
    # 运行检测
    try:
        detector.run()
    except KeyboardInterrupt:
        print("\n检测被用户中断")
    except Exception as e:
        print(f"运行时错误: {e}")

if __name__ == "__main__":
    main()
