# colorBlock_detect

本项目基于YOLO目标检测算法，实现了色块检测功能，适用于视频或图片中的色块识别与定位。

## 目录结构

```
yolo_detect.py                # 主检测脚本
results/
    detect_n/
        train/
            args.yaml         # 训练参数配置
            BoxF1_curve.png  # F1曲线
            BoxP_curve.png   # 精确率曲线
            BoxPR_curve.png  # 精确率-召回率曲线
            BoxR_curve.png   # 召回率曲线
            confusion_matrix_normalized.png # 归一化混淆矩阵
            confusion_matrix.png            # 混淆矩阵
            labels.jpg       # 标签分布
            results.csv      # 训练结果数据
            results.png      # 训练结果可视化
            train_batch*.jpg # 训练批次样本
            val_batch*_labels.jpg # 验证集标签
            val_batch*_pred.jpg   # 验证集预测
            weights/
                best.engine  # 最优TensorRT模型
                best.onnx    # 最优ONNX模型
                best.pt      # 最优PyTorch模型
                last.pt      # 最后一次训练模型

test/
    test.mp4                 # 测试视频
```

## 环境依赖
- Python 3.7+
- PyTorch
- OpenCV
- 其它依赖请参考`requirements.txt`或根据实际报错安装

## 快速开始

1. **模型训练**
   - 请根据实际需求准备数据集，并配置YOLO训练参数。
   - 训练完成后，模型权重保存在`results/detect_n/train/weights/`目录下。


2. **模型推理/检测**
   - 运行检测脚本：
     ```bash
     python yolo_detect.py --model results/detect_n/train/weights/best.pt --source test/test.mp4
     ```
   - 详细参数说明及用法见下方“yolo_detect.py使用手册”。

---

## yolo_detect.py 使用手册

`yolo_detect.py` 是本项目的通用推理脚本，支持图片、视频、摄像头等多种输入源，支持检测结果保存、帧率显示、摄像头录制等功能。

### 基本用法

```bash
python yolo_detect.py --model <模型权重路径> [其它参数]
```

### 常用参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model, -m | YOLO模型权重路径（必填） | - |
| --source, -s | 输入源：摄像头ID(如0)、图片/视频路径 | 0 |
| --duty, -d | 任务类型（detect/segment/classify/pose） | detect |
| --imgsz, -i | 输出图像尺寸 | 640 |
| --flip, -f | 翻转模式：1=水平，-1=垂直，0=同时，2=不翻转 | 1 |
| --conf, -c | 置信度阈值 | 0.3 |
| --save | 保存检测结果 | False |
| --output, -o | 输出目录 | ./test_result |
| --jetson, -j | 使用Jetson CSI摄像头 | False |
| --cam_width, -W | 摄像头宽度 | 1280 |
| --cam_height, -H | 摄像头高度 | 720 |
| --cam_fps, -fps | 摄像头帧率 | 30 |

### 输入源说明
- 摄像头：`--source 0`（或其它摄像头ID）
- 图片：`--source path/to/image.jpg`
- 视频：`--source path/to/video.mp4`

### 典型用法示例

1. **摄像头实时检测**
   ```bash
   python yolo_detect.py --model results/detect_n/train/weights/best.pt --source 0
   ```

2. **检测图片并保存结果**
   ```bash
   python yolo_detect.py --model results/detect_n/train/weights/best.pt --source path/to/image.jpg --save
   ```

3. **检测视频并保存结果视频**
   ```bash
   python yolo_detect.py --model results/detect_n/train/weights/best.pt --source path/to/video.mp4 --save
   ```

4. **设置检测阈值和输出尺寸**
   ```bash
   python yolo_detect.py --model results/detect_n/train/weights/best.pt --source 0 --conf 0.5 --imgsz 800
   ```

5. **不翻转画面**
   ```bash
   python yolo_detect.py --model results/detect_n/train/weights/best.pt --source 0 --flip 2
   ```

6. **Jetson平台CSI摄像头检测**
   ```bash
   python yolo_detect.py --model results/detect_n/train/weights/best.pt --jetson
   ```

### 运行时快捷键
- `q`：退出检测
- `s`：保存当前帧（需加 --save）
- `r`：摄像头模式下开始/停止录制（需加 --save）

### 检测结果
- 检测结果图片/视频/帧保存于 `--output` 指定目录。
- 检测窗口会实时显示检测框、类别、置信度、帧率等信息。

---

3. **结果查看**
   - 检测结果及训练过程可视化文件保存在`results/`目录下。

## 参考
- [YOLO官方文档](https://github.com/ultralytics/yolov5)

## 联系方式
如有问题欢迎提issue或联系作者。
