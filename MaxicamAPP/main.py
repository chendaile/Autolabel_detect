from maix import camera, display, image, nn, app, pinmap
import time
import struct
import signal
import sys
import math

class MaixYOLODetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45, 
                 use_serial=True, serial_device="/dev/ttyS1", baudrate=115200):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_serial = use_serial

        self.detector = nn.YOLOv8(model=model_path, dual_buff=False)
        
        print(f"模型加载成功: {model_path}")
        print(f"检测类别: {self.detector.labels}")

        self.cam = camera.Camera(
            self.detector.input_width(), 
            self.detector.input_height(), 
            self.detector.input_format(),
            buff_num=1
        )
        self.cam_width = self.detector.input_width()
        self.cam_height = self.detector.input_height()
        self.center_x = self.cam_width // 2
        self.center_y = self.cam_height // 2
        print(f"摄像头初始化: {self.cam_width}x{self.cam_height}")
        
        self.disp = display.Display()
        self.ser = None
        if self.use_serial:
            try:
                import serial
                self.ser = serial.Serial(serial_device, baudrate, timeout=0.1)
                print(f"串口初始化成功: {serial_device}, 波特率: {baudrate}")
            except Exception as e:
                print(f"串口初始化失败: {e}")
                self.use_serial = False

        self.colors = [
            image.COLOR_RED,
            image.COLOR_GREEN, 
            image.COLOR_BLUE,
            image.COLOR_YELLOW,
            image.COLOR_GRAY,
            image.COLOR_WHITE,
            image.COLOR_ORANGE,
        ]
        
        # 中心目标框的特殊颜色
        self.center_target_color = image.COLOR_ORANGE
    
    def calculate_distance_to_center(self, obj):
        """计算检测框中心到画面中心的距离"""
        obj_center_x = obj.x + obj.w / 2
        obj_center_y = obj.y + obj.h / 2
        
        distance = math.sqrt((obj_center_x - self.center_x)**2 + (obj_center_y - self.center_y)**2)
        return distance
    
    def find_closest_to_center(self, objs):
        """找到最接近画面中心的检测框"""
        if not objs:
            return None
        
        closest_obj = None
        min_distance = float('inf')
        
        for obj in objs:
            distance = self.calculate_distance_to_center(obj)
            if distance < min_distance:
                min_distance = distance
                closest_obj = obj
        
        return closest_obj
    
    def draw_detections(self, img, objs):
        """在图像上绘制检测结果"""
        if not objs:
            return img
        
        # 找到最接近中心的检测框
        closest_obj = self.find_closest_to_center(objs)
        
        for obj in objs:
            # 如果是最接近中心的框，使用特殊颜色，否则使用普通颜色
            if obj == closest_obj:
                color = self.center_target_color
                thickness = 3  # 加粗显示
            else:
                color = self.colors[obj.class_id % len(self.colors)]
                thickness = 2
            
            img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=color, thickness=thickness)
            label = f'{self.detector.labels[obj.class_id]}: {obj.score:.2f}'
            
            # 如果是中心目标，添加特殊标识
            if obj == closest_obj:
                label = f'[TARGET] {label}'
            
            img.draw_string(obj.x, obj.y - 20, label, color=color)
        
        return img
    
    def send_serial(self, objs):
        """发送串口数据 - 发送最接近中心的人脸位置"""
        if not objs:
            # 没有检测到目标
            data_list = [0, 0, 0]
        else:
            # 找到最接近中心的检测框
            closest_obj = self.find_closest_to_center(objs)
            
            if closest_obj:
                # 计算目标中心坐标并归一化到0-255范围
                center_x = int((closest_obj.x + closest_obj.w/2) / self.cam_width * 255)
                center_y = int((closest_obj.y + closest_obj.h/2) / self.cam_height * 255)
                detect_flag = 1  # 检测到目标
                
                data_list = [center_x, center_y, detect_flag]
            else:
                data_list = [0, 0, 0]

        # 计算校验和
        last_num = sum([0xAA, 0x06, *data_list]) % (0xFF + 1)
        data = struct.pack("<" + "B" * 6, 0xAA, 0x06, *data_list, last_num)
        
        if self.use_serial and self.ser:
            self.ser.write(data)
            print(f"发送串口数据: {data_list}")

    def add_info_overlay(self, img, objs, fps):
        info_text = f"Objects: {len(objs)} | FPS: {fps:.1f}"
        
        img.draw_string(10, 10, info_text, color=image.COLOR_RED, scale=1)

        def draw_center(img):
            """绘制缩小的中心瞄准镜"""
            h, w = img.height(), img.width()
            center_x, center_y = w // 2, h // 2
            
            # 颜色定义
            cyan = image.COLOR_GRAY
            green = image.COLOR_GREEN
            white = image.COLOR_WHITE
            
            # 缩小的多层圆圈系统 (原来的一半大小)
            img.draw_circle(center_x, center_y, 30, color=cyan, thickness=1)      # 外圆
            img.draw_circle(center_x, center_y, 22, color=green, thickness=2)     # 主圆
            img.draw_circle(center_x, center_y, 15, color=cyan, thickness=1)      # 中圆
            img.draw_circle(center_x, center_y, 8, color=green, thickness=1)      # 内圆
            
            # 缩小的主十字线
            line_gap = 5   # 中心空隙
            line_length = 25  # 缩小线长
            
            # 水平主线
            img.draw_line(center_x - line_length, center_y, center_x - line_gap, center_y, color=green, thickness=2)
            img.draw_line(center_x + line_gap, center_y, center_x + line_length, center_y, color=green, thickness=2)
            
            # 垂直主线
            img.draw_line(center_x, center_y - line_length, center_x, center_y - line_gap, color=green, thickness=2)
            img.draw_line(center_x, center_y + line_gap, center_x, center_y + line_length, color=green, thickness=2)
            
            # 缩小的细十字线
            fine_length = 12
            img.draw_line(center_x - fine_length, center_y, center_x - line_gap, center_y, color=cyan, thickness=1)
            img.draw_line(center_x + line_gap, center_y, center_x + fine_length, center_y, color=cyan, thickness=1)
            img.draw_line(center_x, center_y - fine_length, center_x, center_y - line_gap, color=cyan, thickness=1)
            img.draw_line(center_x, center_y + line_gap, center_x, center_y + fine_length, color=cyan, thickness=1)
            
            # 缩小的距离刻度标记
            for distance in [10, 17, 25]:  # 缩小距离
                thickness = 2 if distance == 17 else 1
                mark_color = green if distance == 17 else cyan
                mark_length = 3 if distance == 17 else 2  # 缩小标记长度
                
                # 水平标记
                img.draw_line(center_x - distance, center_y - mark_length, center_x - distance, center_y + mark_length, color=mark_color, thickness=thickness)
                img.draw_line(center_x + distance, center_y - mark_length, center_x + distance, center_y + mark_length, color=mark_color, thickness=thickness)
                
                # 垂直标记
                img.draw_line(center_x - mark_length, center_y - distance, center_x + mark_length, center_y - distance, color=mark_color, thickness=thickness)
                img.draw_line(center_x - mark_length, center_y + distance, center_x + mark_length, center_y + distance, color=mark_color, thickness=thickness)
            
            # 中心精确瞄准点
            img.draw_circle(center_x, center_y, 2, color=white, thickness=1)  # 缩小中心点
            img.draw_circle(center_x, center_y, 1, color=green, thickness=-1)
        
        draw_center(img)

        # 显示检测统计和目标信息
        if objs:
            class_counts = {}
            for obj in objs:
                class_name = self.detector.labels[obj.class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            y_offset = 25
            # for class_name, count in class_counts.items():
            #     text = f"{class_name}: {count}"
            #     img.draw_string(10, y_offset, text, color=image.COLOR_YELLOW)
            #     y_offset += 20
            
            # 显示中心目标信息
            closest_obj = self.find_closest_to_center(objs)
            if closest_obj:
                distance = self.calculate_distance_to_center(closest_obj)
                target_info = f"Target Distance: {distance:.1f}px"
                img.draw_string(10, y_offset, target_info, color=self.center_target_color)

        return img

    def cleanup(self):
        """清理资源"""
        print("Cleaning up resources...")
        try:
            if hasattr(self, 'cam') and self.cam:
                self.cam.close()
                print("Camera closed.")
        except Exception as e:
            print(f"Error closing camera: {e}")
        
        try:
            if hasattr(self, 'disp') and self.disp:
                self.disp.close()
                print("Display closed.")
        except Exception as e:
            print(f"Error closing display: {e}")
        
        try:
            if hasattr(self, 'detector') and self.detector:
                del self.detector  # 或使用 nn.destroy 如果可用
                print("NN detector destroyed.")
        except Exception as e:
            print(f"Error destroying detector: {e}")
        
        try:
            if self.use_serial and self.ser:
                self.ser.close()
                print("Serial closed.")
        except Exception as e:
            print(f"Error closing serial: {e}")
        
        # 如果有 PWM 或其他资源，在这里添加 unexport
        try: 
            with open("/sys/class/pwm/pwmchip0/unexport", "w") as f:
                f.write("10")
        except: 
            pass

    def run(self):
        """运行检测主循环"""
        print("开始检测，按Ctrl+C退出...")
        print(f"置信度阈值: {self.conf_threshold}")
        print(f"IOU阈值: {self.iou_threshold}")
        print(f"串口传输: {'启用' if self.use_serial else '禁用'}")
        print("串口数据格式: [中心目标X坐标, 中心目标Y坐标, 检测标志(1/0)]")
        print("-" * 50)
        
        # 设置信号处理
        def signal_handler(sig, frame):
            print("Received termination signal, cleaning up...")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)  # 也处理 Ctrl+C
        
        start_time = time.time()
        
        try:
            while not app.need_exit():
                img = self.cam.read()
                objs = self.detector.detect(
                    img, 
                    conf_th=self.conf_threshold, 
                    iou_th=self.iou_threshold
                )
                
                # 绘制检测结果
                img = self.draw_detections(img, objs)
                
                # 计算FPS
                current_time = time.time()
                fps = 1 / (current_time - start_time)
                start_time = current_time
                
                # 添加信息叠加
                img = self.add_info_overlay(img, objs, fps)
                
                # 发送串口数据 - 现在发送最接近中心的目标
                self.send_serial(objs)
                
                # 显示图像
                self.disp.show(img)
        finally:
            self.cleanup()
            print("检测结束")

def main():
    """主函数"""
    pinmap.set_pin_function("A18", "UART1_RX")
    pinmap.set_pin_function("A19", "UART1_TX")

    MODEL_PATH = "yolov8n_face.mud"  # 修改为你的模型路径
    CONF_THRESHOLD = 0.5      # 置信度阈值
    IOU_THRESHOLD = 0.45      # IOU阈值
    USE_SERIAL = True         # 是否使用串口
    SERIAL_DEVICE = "/dev/ttyS1"  # MaixCAM串口设备
    BAUDRATE = 115200         # 波特率
    
    # 创建检测器
    detector = MaixYOLODetector(
        model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        use_serial=USE_SERIAL,
        serial_device=SERIAL_DEVICE,
        baudrate=BAUDRATE
    )
    
    # 运行检测
    detector.run()

main()