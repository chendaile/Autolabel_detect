import os
import cv2
import argparse
from pathlib import Path
import numpy as np
from ultralytics import YOLO


class YOLOProcessor:
    """YOLO模型推理和数据处理类"""
    
    def __init__(self, model_path, class_names=None):
        """
        初始化YOLO处理器
        
        Args:
            model_path (str): YOLO模型文件路径
            class_names (list): 类别名称列表，如果为None则使用模型默认类别
        """
        self.model_path = model_path
        self.model = YOLO(model_path)
        
        # 获取类别名称
        if class_names is None:
            self.class_names = self.model.names
        else:
            self.class_names = {i: name for i, name in enumerate(class_names)}
        
        print(f"模型加载成功: {model_path}")
        print(f"检测类别: {list(self.class_names.values())}")
    
    def create_directories(self, base_dir):
        """创建输出目录结构"""
        base_path = Path(base_dir)
        self.images_dir = base_path / "images"
        self.labels_dir = base_path / "labels"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"输出目录创建完成:")
        print(f"  图片目录: {self.images_dir}")
        print(f"  标注目录: {self.labels_dir}")
    
    def process_image(self, image_path):
        """
        处理单张图片
        
        Args:
            image_path (str): 图片路径
            
        Returns:
            tuple: (原始图片, YOLO格式标注数据)
        """
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"警告: 无法读取图片 {image_path}")
            return None, None
        
        # 运行YOLO推理
        results = self.model(image)
        
        # 获取图片尺寸
        h, w = image.shape[:2]
        
        # 处理检测结果
        yolo_annotations = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标 (xyxy格式)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 转换为YOLO格式 (中心点坐标和相对尺寸)
                    center_x = (x1 + x2) / 2 / w
                    center_y = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # 获取类别
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # 保存YOLO格式标注
                    yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        return image, yolo_annotations
    
    def process_folder(self, input_folder, output_dir):
        """
        处理文件夹中的所有图片
        
        Args:
            input_folder (str): 输入图片文件夹路径
            output_dir (str): 输出目录路径（将直接在此目录下创建images和labels文件夹）
        """
        # 创建输出目录
        self.create_directories(output_dir)
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 获取所有图片文件
        input_path = Path(input_folder)
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"错误: 在 {input_folder} 中未找到支持的图片文件")
            return
        
        print(f"找到 {len(image_files)} 张图片，开始处理...")
        
        processed_count = 0
        for image_file in image_files:
            print(f"处理中: {image_file.name}")
            
            # 处理图片
            original_image, annotations = self.process_image(image_file)
            
            if original_image is not None:
                # 保存原始图片
                output_image_path = self.images_dir / image_file.name
                cv2.imwrite(str(output_image_path), original_image)
                
                # 保存YOLO格式标注文件
                label_file = self.labels_dir / f"{image_file.stem}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(annotations))
                
                processed_count += 1
                print(f"  已保存: {output_image_path}")
                print(f"  标注数: {len(annotations)}")
        
        print(f"\n处理完成! 共处理 {processed_count} 张图片")


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='YOLO模型推理和数据处理工具')
    parser.add_argument('--model', '-m', required=True, help='YOLO模型文件路径 (.pt文件)')
    parser.add_argument('--input', '-i', required=True, help='输入图片文件夹路径')
    parser.add_argument('--output', '-o', default='./data', help='输出目录路径 (默认: ./data)')
    parser.add_argument('--classes', '-c', nargs='+', help='自定义类别名称列表')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在 - {args.model}")
        return
    
    # 检查输入文件夹是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件夹不存在 - {args.input}")
        return
    
    try:
        # 创建YOLO处理器
        processor = YOLOProcessor(args.model, args.classes)
        
        # 处理图片
        processor.process_folder(args.input, args.output)
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()