# ColorBlock Detect

<div align="center">

![Python](https://img.shields.io/badge/python-3.7%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-supported-red) ![OpenCV](https://img.shields.io/badge/OpenCV-required-green) ![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-yellow)

**基于 YOLO 的色块检测系统**

支持图片、视频、实时摄像头检测

</div>

---

## 快速开始

### 1. 环境准备
```bash
pip install ultralytics opencv-python torch torchvision
```

### 2. 直接使用（推荐）
如果您有现成的 YOLO 模型，可以直接开始检测：

```bash
# 摄像头实时检测
python yolo_detect.py --model your_model.pt --source 0

# 视频文件检测
python yolo_detect.py --model your_model.pt --source test.mp4

# 图片检测
python yolo_detect.py --model your_model.pt --source image.jpg
```

### 3. 训练自己的模型（可选）
```bash
# 第一步：准备数据集并划分训练验证集
python train/train_val_split.py --datapath ./dataset --train_pct 0.8

# 第二步：开始训练
python train/train.py

# 第三步：使用训练好的模型
python yolo_detect.py --model results/detect_n/train/weights/best.pt --source 0
```

## 核心功能

| 功能 | 说明 |
|------|------|
| **多输入源** | 支持摄像头（ID如0,1）、视频文件（.mp4等）、图片文件（.jpg等） |
| **实时检测** | 摄像头模式下实时显示检测结果和FPS |
| **结果保存** | 可保存检测后的图片/视频，支持录制功能 |
| **灵活配置** | 可调节置信度、图像尺寸、翻转模式等 |
| **平台支持** | 支持普通USB摄像头和Jetson CSI摄像头 |

## 详细使用说明

### 基础检测命令

**必需参数：**
- `--model` / `-m`：YOLO 模型文件路径

**输入源参数：**
- `--source` / `-s`：输入源，默认为 0
  - 摄像头：使用数字 `0`, `1`, `2` 等
  - 视频：使用文件路径 `video.mp4`
  - 图片：使用文件路径 `image.jpg`

### 常用配置参数

```bash
python yolo_detect.py \
  --model model.pt \           # 模型路径
  --source 0 \                 # 输入源
  --conf 0.5 \                 # 置信度阈值（0-1）
  --imgsz 640 \                # 输出图像尺寸
  --save \                     # 保存检测结果
  --output ./results           # 输出目录
```

### 图像翻转设置

当使用前置摄像头时，通常需要水平翻转：

```bash
--flip 1    # 水平翻转（默认，适合前置摄像头）
--flip -1   # 垂直翻转
--flip 0    # 水平+垂直翻转
--flip 2    # 不翻转（适合后置摄像头）
```

### Jetson 平台使用

```bash
python yolo_detect.py \
  --model model.pt \
  --jetson \                   # 启用Jetson CSI摄像头
  --cam_width 1920 \           # 摄像头分辨率宽度
  --cam_height 1080 \          # 摄像头分辨率高度
  --cam_fps 30                 # 摄像头帧率
```

## 交互操作

在摄像头模式下，支持以下键盘操作：

| 按键 | 功能 |
|------|------|
| `q` | 退出程序 |
| `s` | 保存当前帧（需要 --save 参数） |
| `r` | 开始/停止视频录制（需要 --save 参数） |

## 使用示例

### 示例1：基础摄像头检测
```bash
python yolo_detect.py --model best.pt --source 0
```

### 示例2：高精度视频检测并保存
```bash
python yolo_detect.py --model best.pt --source video.mp4 --conf 0.7 --save
```

### 示例3：批量图片检测
```bash
python yolo_detect.py --model best.pt --source image.jpg --save --output ./detected_images
```

### 示例4：Jetson平台实时检测
```bash
python yolo_detect.py --model best.pt --jetson --cam_width 1280 --cam_height 720 --save
```

## 训练说明

### 数据集准备

您的数据集应该按以下结构组织：
```
dataset/
├── images/          # 所有图片文件
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── labels/          # 对应的标注文件
    ├── img1.txt
    ├── img2.txt
    └── ...
```

### 数据集自动划分

```bash
python train/train_val_split.py --datapath ./dataset --train_pct 0.8
```

执行后会在当前目录创建 `data/` 文件夹，包含训练集和验证集。

### 开始训练

```bash
python train/train.py
```

训练完成后，最佳模型保存在 `results/detect_n/train/weights/best.pt`

## 项目结构

```
ColorBlock_detect/
├── yolo_detect.py              # 主检测脚本
├── train/
│   ├── train.py               # 训练脚本
│   └── train_val_split.py     # 数据集划分工具
├── results/
│   └── detect_n/train/weights/
│       ├── best.pt            # 最佳训练模型
│       └── last.pt            # 最新训练模型
└── test/
    └── test.mp4               # 测试视频
```

## 常见问题

**Q: 摄像头无法打开？**
A: 检查摄像头ID是否正确，尝试 0, 1, 2 等不同数字。

**Q: 检测效果不好？**
A: 尝试调整 `--conf` 参数，降低置信度阈值。

**Q: 想要训练自己的模型？**
A: 准备好数据集后，依次运行数据划分和训练脚本。

**Q: 如何保存检测结果？**
A: 添加 `--save` 参数，结果会保存到 `--output` 指定的目录。

## 系统要求

- Python 3.7+
- 主要依赖：PyTorch, OpenCV, Ultralytics
- 支持 CPU/GPU 推理
- Jetson 平台兼容

---

<div align="center">

**如果这个项目对你有帮助，欢迎 Star ⭐**

</div>