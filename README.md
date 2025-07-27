# ColorBlock Detect

<div align="center">

![Python](https://img.shields.io/badge/python-3.7%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-supported-red) ![OpenCV](https://img.shields.io/badge/OpenCV-required-green) ![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-yellow)

**基于运行在Jetson Xavier Nx上YOLO的不同颜色的正方体目标检测系统**

支持图片、视频、实时摄像头检测

</div>

---

## 项目结构

```
ColorBlock_detect/
├── yolo_detect.py              # 主检测脚本
├── train/
│   ├── train.py               # 训练脚本
│   └── train_val_split.py     # 数据集划分工具
├── results/
│   └── detect_n/
│       └── train/
│           ├── weights/       # 训练权重目录
│           │   ├── best.pt    # 最佳模型权重
│           │   └── last.pt    # 最新模型权重
│           ├── train_batch*.jpg   # 训练过程可视化
│           ├── val_batch*.jpg     # 验证过程可视化
│           ├── confusion_matrix.png  # 混淆矩阵
│           └── results.png        # 训练曲线图
├── data/                      # 训练数据目录（由划分脚本生成）（此处文件过大不予上传）
│   ├── train/
│   │   ├── images/           # 训练图片
│   │   └── labels/           # 训练标签
│   └── validation/
│       ├── images/           # 验证图片
│       └── labels/           # 验证标签
├── test/
│   └── test.mp4              # 测试视频文件
└── premodel/
    └── yolo11n.pt            # 预训练模型
```

## 第一步：数据准备与划分

### 原始数据集结构

在使用数据划分工具之前，您的原始数据集应该按照以下结构组织：

```
your_dataset/
├── images/                    # 所有训练图片
│   ├── image001.jpg
│   ├── image002.jpg
│   ├── image003.png
│   └── ...
└── labels/                    # 对应的YOLO格式标注文件
    ├── image001.txt
    ├── image002.txt
    ├── image003.txt
    └── ...
```

**标注文件格式说明：**
每个 `.txt` 文件对应一张图片，格式为：
```
class_id center_x center_y width height
```
- `class_id`：目标类别ID（从0开始）
- `center_x, center_y`：目标中心点坐标（相对于图片宽高的比例，0-1之间）
- `width, height`：目标宽高（相对于图片宽高的比例，0-1之间）

### 使用数据划分工具

`train_val_split.py` 脚本用于将原始数据集随机划分为训练集和验证集。

#### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--datapath` | str | 是 | - | 原始数据集路径，包含images和labels文件夹 |
| `--train_pct` | float | 否 | 0.8 | 训练集比例，范围0.01-0.99，剩余部分为验证集 |

#### 使用示例

```bash
# 基本用法：80%训练集，20%验证集
python train/train_val_split.py --datapath ./your_dataset

# 自定义训练集比例：90%训练集，10%验证集
python train/train_val_split.py --datapath ./your_dataset --train_pct 0.9

# 指定具体路径
python train/train_val_split.py --datapath /path/to/your/dataset --train_pct 0.85
```

#### 脚本执行过程

1. **验证输入**：检查数据集路径是否存在，训练比例是否合理
2. **创建目录**：自动在当前工作目录下创建 `data/train/` 和 `data/validation/` 目录结构
3. **文件统计**：扫描并统计原始数据集中的图片和标注文件数量
4. **随机划分**：根据指定比例随机选择文件分配到训练集和验证集
5. **文件复制**：将选中的图片和对应标注文件复制到相应目录
6. **结果输出**：显示划分结果统计信息

- 生成的目录结构完全符合YOLO训练要求

## 第二步：模型训练

### 训练配置

`train.py` 脚本基于 Ultralytics YOLO 框架进行模型训练，使用预训练的 YOLOv11n 模型作为起点。

#### 训练参数详解

脚本中的关键参数及其作用：

| 参数 | 类型 | 当前值 | 说明 |
|------|------|--------|------|
| `model` | str | `premodel\yolo11n.pt` | 预训练模型路径，作为训练起点 |
| `data` | str | `data.yaml` | 数据集配置文件路径 |
| `batch` | float | 0.9 | 批次大小，0.9表示使用90%的可用GPU内存 |
| `cache` | bool | True | 是否缓存数据集到内存，加速训练 |
| `time` | float | 0.2 | 数据增强的时间比例 |
| `project` | str | `results\detect_n` | 训练结果保存的项目目录 |

#### 数据配置文件

您需要创建 `data.yaml` 文件来定义数据集信息：

```yaml
# data.yaml 示例内容
path: data          # 数据集根目录
train: train\images         # 训练图片路径（相对于path）         
val: validation\images          # 验证图片路径（相对于path）

nc: 4         # 类别数量

# 类别名称
names: ["blue block", "green block", "red block", "yellow block"]
```

#### 训练命令

```bash
# 开始训练
python train/train.py
```

#### 训练过程说明

1. **模型初始化**：加载预训练的YOLOv11n模型
2. **数据加载**：根据data.yaml配置加载训练和验证数据
3. **训练执行**：
   - 自动进行数据增强
   - 实时显示训练进度和损失
   - 定期在验证集上评估模型性能
   - 自动保存最佳模型（best.pt）和最新模型（last.pt）
4. **结果保存**：
   - 训练曲线图（results.png）
   - 混淆矩阵（confusion_matrix.png）
   - 训练批次可视化（train_batch*.jpg）
   - 验证批次可视化（val_batch*.jpg）

**训练技巧：**
- `batch=0.9` 会自动根据GPU内存调整批次大小
- `cache=True` 可以显著加速训练，但需要足够的内存
- 训练过程中可以通过Ctrl+C安全停止
- 更多设置可以在train.py脚本内调整

## 第三步：模型检测与推理

### 主检测脚本概述

`yolo_detect.py` 是核心检测脚本，支持多种输入源和丰富的配置选项。脚本采用面向对象设计，主要包含 `YOLODetector` 类来处理所有检测任务, 输入`python yolo_detect.py -h`以帮助写入参数。

### 完整参数列表

| 参数 | 简写 | 类型 | 默认值 | 必需 | 说明 |
|------|------|------|--------|------|------|
| `--model` | `-m` | str | - | 是 | YOLO模型文件路径（.pt格式） |
| `--source` | `-s` | str | `'0'` | 否 | 输入源：摄像头ID/视频路径/图片路径 |
| `--duty` | `-d` | str | `'detect'` | 否 | 任务类型，当前仅支持detect |
| `--imgsz` | `-i` | int | `640` | 否 | 输出图像尺寸（像素） |
| `--flip` | `-f` | int | `1` | 否 | 图像翻转模式 |
| `--conf` | `-c` | float | `0.3` | 否 | 置信度阈值（0.0-1.0） |
| `--save` | - | bool | `False` | 否 | 是否保存检测结果 |
| `--output` | `-o` | str | `'./test_result'` | 否 | 输出目录路径 |
| `--jetson` | `-j` | bool | `False` | 否 | 是否使用Jetson CSI摄像头 |
| `--cam_width` | `-W` | int | `1280` | 否 | CSI摄像头宽度 |
| `--cam_height` | `-H` | int | `720` | 否 | CSI摄像头高度 |
| `--cam_fps` | `-fps` | int | `30` | 否 | CSI摄像头帧率 |

### 参数详细说明

#### 核心参数

**`--model` (必需参数)**
- 指定训练好的YOLO模型文件路径
- 支持相对路径和绝对路径
- 示例：`--model results/detect_n/train/weights/best.pt`
- Jetson具有TensorRT模块，转化为FP16的Engine格式可以加速

**`--source` (输入源配置)**
输入源类型自动识别机制：
- **摄像头模式**：使用数字ID（如 `0`, `1`, `2`）
  - `0` 通常是默认摄像头
  - `1`, `2` 等为额外摄像头设备
- **视频模式**：支持格式 `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`
- **图片模式**：支持格式 `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

#### 图像处理参数

**`--imgsz` (图像尺寸)**
- 控制检测时的图像尺寸，影响检测精度和速度
- 较大尺寸：精度更高，速度较慢
- 较小尺寸：速度更快，精度可能降低
- 常用值：320, 640, 1280

**`--flip` (图像翻转)**
翻转模式详细说明：

| 值 | 模式 | 效果 | 适用场景 |
|---|------|------|----------|
| `1` | 水平翻转 | 左右镜像 | 前置摄像头（默认推荐） |
| `-1` | 垂直翻转 | 上下颠倒 | 特殊安装角度 |
| `0` | 双向翻转 | 水平+垂直翻转 | 180度旋转场景 |
| `2` | 不翻转 | 保持原样 | 后置摄像头、视频文件 |

**`--conf` (置信度阈值)**
- 控制检测结果的筛选严格程度
- 数值范围：0.0 - 1.0
- 较高值（如0.7）：只显示高置信度检测，减少误检
- 较低值（如0.3）：显示更多可能的检测，可能包含误检
- 建议根据实际应用场景调整

#### 输出控制参数

**`--save` (结果保存)**
启用后的保存行为：
- **图片模式**：保存检测后的图片到输出目录
- **视频模式**：保存检测后的视频文件，支持多种编码格式
- **摄像头模式**：支持手动保存帧和录制视频

**`--output` (输出目录)**
- 指定所有输出文件的保存位置
- 目录不存在时会自动创建
- 建议使用绝对路径避免路径混乱

#### Jetson平台参数

Jetson设备的CSI摄像头专用参数：

**`--jetson`**
- 启用Jetson Nano/Xavier等设备的CSI摄像头支持
- 使用GStreamer管道进行视频采集
- 默认不启动该选项，即默认使用USB摄像头

**`--cam_width` / `--cam_height`**
- 设置CSI摄像头的采集分辨率
- 常用分辨率：
  - 1280x720 (720p)
  - 1920x1080 (1080p)
  - 3264x2464 (5MP最大分辨率)

**`--cam_fps`**
- 设置摄像头帧率
- 建议值：15-30 FPS
- 过高帧率可能导致系统负载过大

### 使用示例

#### 基础检测示例

```bash
# 最简单的摄像头检测
python yolo_detect.py --model best.pt

# 指定摄像头ID
python yolo_detect.py --model best.pt --source 1

# 检测视频文件
python yolo_detect.py --model best.pt --source test_video.mp4

# 检测单张图片
python yolo_detect.py --model best.pt --source test_image.jpg
```

#### 高级配置示例

```bash
# 高精度检测并保存结果
python yolo_detect.py \
  --model results/detect_n/train/weights/best.pt \
  --source test_video.mp4 \
  --conf 0.7 \
  --imgsz 1280 \
  --save \
  --output ./high_quality_results

# 摄像头实时检测（适合监控场景）
python yolo_detect.py \
  --model best.pt \
  --source 0 \
  --conf 0.5 \
  --flip 1 \
  --save

# 不翻转的后置摄像头检测
python yolo_detect.py \
  --model best.pt \
  --source 0 \
  --flip 2 \
  --imgsz 640

# Jetson平台高分辨率检测
python yolo_detect.py \
  --model best.pt \
  --jetson \
  --cam_width 1920 \
  --cam_height 1080 \
  --cam_fps 30 \
  --save \
  --output ./jetson_results
```

#### 批量处理示例

```bash
# 处理多个视频文件（需要脚本循环）
for video in *.mp4; do
  python yolo_detect.py --model best.pt --source "$video" --save --output ./batch_results
done

# 处理多个图片文件
for image in *.jpg; do
  python yolo_detect.py --model best.pt --source "$image" --save
done
```

### 交互式操作

在摄像头模式下，脚本支持实时交互：

| 按键 | 功能 | 详细说明 |
|------|------|----------|
| `q` | 退出程序 | 安全退出，释放摄像头资源 |
| `s` | 保存当前帧 | 将当前检测结果保存为图片（需启用--save） |
| `r` | 录制切换 | 开始/停止视频录制（需启用--save） |

### 输出文件说明

启用 `--save` 参数后，不同模式的输出文件：

#### 图片模式输出
```
output_directory/
└── result_image_name.jpg      # 检测结果图片
```

#### 视频模式输出
```
output_directory/
└── result_video_name.mp4      # 检测结果视频
```

#### 摄像头模式输出
```
output_directory/
├── camera_00h15m30s.jpg       # 手动保存的帧（按's'键）
├── camera_record_00h15m45s.mp4 # 录制的视频（按'r'键）
└── ...
```

### 性能优化建议

#### 提高检测速度
1. 减小 `--imgsz` 参数值（如320或480）
2. 提高 `--conf` 阈值减少后处理时间
3. 使用GPU进行推理

#### 提高检测精度
1. 增大 `--imgsz` 参数值（如1280）
2. 降低 `--conf` 阈值捕获更多目标
3. 确保输入图像质量良好

#### 内存优化
1. 避免同时处理多个大尺寸视频
2. 定期清理输出目录
3. 监控系统内存使用情况

<div align="center">

**如果这个项目对你有帮助，欢迎 Star ⭐**

</div>