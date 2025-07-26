# ColorBlock Detect

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-supported-red) ![OpenCV](https://img.shields.io/badge/OpenCV-required-green) ![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-yellow)

</div>

本项目基于 YOLO 目标检测算法，实现了色块检测功能，适用于视频或图片中的色块识别与定位。

## 目录

- [特性](#特性)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本使用手册](#脚本使用手册)
- [参考资料](#参考资料)
- [联系方式](#联系方式)

## 特性

- 支持图片、视频、摄像头等多种输入源
- 实时检测，高性能推理
- 完整的训练和评估流程
- 支持检测结果保存
- 支持摄像头实时录制
- Jetson 平台支持

## 项目结构

```
yolo_detect.py                # 主检测脚本
train/
    train.py                 # 训练脚本
    train_val_split.py      # 数据集划分脚本
results/
    detect_n/
        train/              # 训练输出目录
            weights/        # 模型权重
                best.pt     # 最优模型
                last.pt     # 最新模型
            *.png          # 训练过程可视化图表
test/
    test.mp4               # 测试视频
```

## 环境要求

- Python 3.7+
- PyTorch
- OpenCV
- Ultralytics
- 其它依赖请参考 `requirements.txt` 或根据实际报错安装

## 快速开始

### 数据集准备与划分

```bash
python train/train_val_split.py --datapath ./dataset --train_pct 0.8
```

### 模型训练

```bash
python train/train.py
```

### 模型推理/检测

```bash
python yolo_detect.py --model results/detect_n/train/weights/best.pt --source test/test.mp4
```

## 脚本使用手册

### yolo_detect.py

通用推理脚本，支持多种输入源。

<details>
<summary>常用参数</summary>

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

</details>

### train_val_split.py

数据集划分脚本，自动创建所需目录结构并随机复制图片及标注文件。

```bash
python train/train_val_split.py --datapath <数据集路径> --train_pct <训练集比例>
```

### train.py

基于 Ultralytics YOLO 框架进行模型训练。

```bash
python train/train.py
```

## 参考资料

- [YOLO 官方文档](https://github.com/ultralytics/yolov5)
- [Ultralytics 文档](https://docs.ultralytics.com/)

## 联系方式

如有问题欢迎提 [Issue](https://github.com/chendaile/ColorBlock_detect/issues)。

<div align="center">
⭐️ 如果这个项目对你有帮助，欢迎 Star！
</div>
