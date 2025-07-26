# 🎯 ColorBlock Detect

<div align="center">

![Python Version](https### 3. 模型推理/检测

```bash
python yolo_detect.py --model results/detect_n/train/weights/best.pt --source test/test.mp4
```

## 📚 详细文档elds.io/badge/python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-supported-red)
![OpenCV](https://img.shields.io/badge/OpenCV-required-green)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-yellow)

</div>

本项目基于 YOLO 目标检测算法，实现了色块检测功能，适用于视频或图片中的色块识别与定位。

## 📑 目录

- [特性](#-特性)
- [环境要求](#-环境要求)
- [快速开始](#-快速开始)
- [详细文档](#-详细文档)
- [参考资料](#-参考资料)
- [联系方式](#-联系方式)

## ✨ 特性

- 🎯 支持图片、视频、摄像头等多种输入源
- 🚀 实时检测，高性能推理
- 📊 完整的训练和评估流程
- 💾 支持检测结果保存
- 🎥 支持摄像头实时录制
- 🖥️ Jetson 平台支持

## 📁 项目结构

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

## 📦 环境要求

- Python 3.7+
- PyTorch
- OpenCV
- Ultralytics
- 其它依赖请参考 `requirements.txt` 或根据实际报错安装

## 🚀 快速开始

### 1. 数据集准备与划分

```bash
# 将数据集按8:2的比例划分为训练集和验证集
python train/train_val_split.py --datapath ./dataset --train_pct 0.8
```

### 2. 模型训练

```bash
python train/train.py
```


2. **模型推理/检测**
   - 运行检测脚本：
     ```bash
     python yolo_detect.py --model results/detect_n/train/weights/best.pt --source test/test.mp4
     ```
   - 详细参数说明及用法见下方“yolo_detect.py使用手册”。

---

## yolo_detect.py 使用手册

### yolo_detect.py 使用说明

通用推理脚本，支持多种输入源。

<details>
<summary>📝 常用参数</summary>

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

---

## train_val_split.py 使用手册

`train/train_val_split.py` 用于将数据集随机划分为训练集和验证集。该脚本会自动创建所需的目录结构，并随机复制图片及其对应的标注文件到相应目录。

### 基本用法

```bash
python train/train_val_split.py --datapath <数据集路径> --train_pct <训练集比例>
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --datapath | 数据集根目录，需包含 images/ 和 labels/ 子文件夹（必填） | - |
| --train_pct | 训练集占比（0.01-0.99之间的浮点数） | 0.8 |

### 目录结构要求

输入数据集目录结构：
```
<datapath>/
    images/     # 存放所有图片
    labels/     # 存放所有标注文件
```

输出目录结构：
```
data/
    train/
        images/     # 训练集图片
        labels/     # 训练集标注
    validation/
        images/     # 验证集图片
        labels/     # 验证集标注
```

### 使用示例

```bash
# 将数据集按8:2的比例划分为训练集和验证集
python train/train_val_split.py --datapath ./dataset --train_pct 0.8
```

---

## train.py 使用手册

`train/train.py` 基于 Ultralytics YOLO 框架进行模型训练。该脚本会加载预训练模型，并在自定义数据集上进行训练。

### 基本配置

目前脚本使用了以下默认配置：
- 预训练模型：`premodel/yolo11n.pt`
- 数据集配置：`data.yaml`
- 批次大小：0.9（自动计算）
- 缓存：启用
- 训练时间：0.2（自动计算）
- 输出目录：`results/detect_n`

### 使用方法

1. 确保已准备好：
   - 预训练模型放置在 `premodel/` 目录下
   - 数据集配置文件 `data.yaml`
   - 已完成数据集划分

2. 运行训练：
```bash
python train/train.py
```

3. 训练过程将自动：
   - 加载预训练模型
   - 根据 data.yaml 配置加载数据集
   - 在 `results/detect_n` 目录下保存训练日志和结果
   - 在 `results/detect_n/train/weights/` 下保存模型权重

### 典型用例

```bash
# 1. 摄像头实时检测
python yolo_detect.py --model results/detect_n/train/weights/best.pt --source 0

# 2. 检测图片并保存
python yolo_detect.py --model results/detect_n/train/weights/best.pt --source path/to/image.jpg --save

# 3. 检测视频
python yolo_detect.py --model results/detect_n/train/weights/best.pt --source path/to/video.mp4 --save

# 4. Jetson CSI摄像头
python yolo_detect.py --model results/detect_n/train/weights/best.pt --jetson
```

<details>
<summary>⌨️ 快捷键</summary>

- `q`：退出检测
- `s`：保存当前帧（需加 --save）
- `r`：开始/停止录制（需加 --save）

</details>

## 📚 参考资料

- [YOLO 官方文档](https://github.com/ultralytics/yolov5)
- [Ultralytics 文档](https://docs.ultralytics.com/)

## 📮 联系方式

如有问题欢迎提 [Issue](https://github.com/chendaile/ColorBlock_detect/issues) 或通过以下方式联系作者：

<div align="center">
⭐️ 如果这个项目对你有帮助，欢迎 Star！
</div>
