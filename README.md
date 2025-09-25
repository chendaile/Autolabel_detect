# Autolabel Detect

<div align="center">

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-supported-red)
![OpenCV](https://img.shields.io/badge/OpenCV-required-green)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-yellow)

**基于 Ultralytics YOLO 的自动标注与目标检测工具集**

</div>

> 该仓库经过整理后提供清晰的模块划分与统一的命令行入口，便于在本地或 Jetson 设备上快速完成数据标注与模型效果验证。

---

## 仓库结构概览

```
Autolabel_detect/
├── autolabel_toolkit/        # 可复用的核心库
│   ├── __init__.py
│   ├── cli.py                # 统一的命令行解析器
│   ├── detector.py           # YOLO 检测封装
│   └── labeler.py            # 自动标注封装
├── yolo_autolabel.py         # 兼容旧用法的自动标注入口
├── yolo_detect.py            # 兼容旧用法的检测入口
├── train.py                  # 模型训练脚本（原始版本保留）
├── train_val_split.py        # 数据集划分工具
├── train_results/            # 训练结果示例（可能为空）
├── test/                     # 测试素材目录
└── README.md
```

`autolabel_toolkit` 中的模块可以直接在其他项目中复用，而顶层的 `yolo_autolabel.py` 与 `yolo_detect.py` 仅作为兼容旧脚本名的薄包装。

---

## 环境准备

1. **安装 Python 依赖**

   ```bash
   pip install ultralytics opencv-python numpy
   ```

   如果在 Jetson 平台上部署，建议使用官方轮子或系统自带的 OpenCV。

2. **准备模型权重**

   - 支持任何 Ultralytics YOLO 系列的 `.pt` 模型。
   - 可将训练得到的 `best.pt` 或 `last.pt` 放在 `train_results/weights/` 等自定义位置。

3. **准备输入数据**
   - 自动标注支持 `.jpg/.jpeg/.png/.bmp/.tiff/.tif` 等常见图片格式。
   - 检测模块可以处理摄像头、单张图片或视频文件。

---

## 统一命令行

整理后的项目提供了一个统一的 CLI：

```bash
python -m autolabel_toolkit.cli --help
```

该命令会展示 `label` 与 `detect` 两个子命令。下面给出常见示例。

### 1. 自动标注图片数据集

```bash
python -m autolabel_toolkit.cli label \
  path/to/best.pt \
  ./raw_images \
  ./labeled_dataset \
  --classes boxA boxB boxC
```

- `path/to/best.pt`：YOLO 模型权重。
- `./raw_images`：待自动标注的图片目录。
- `./labeled_dataset`：输出的数据集目录，会自动创建 `images/` 与 `labels/` 子目录。
- `--classes`：可选，自定义类别名称（会覆盖模型内置标签）。
- `--extensions`：可选，手动指定要处理的图片后缀。

自动标注完成后，会在输出目录下写入与原图同名的图片及 YOLO 标注文本文件。

### 2. 运行检测

```bash
python -m autolabel_toolkit.cli detect path/to/best.pt --source 0 --save
```

常用参数说明：

| 参数                                 | 说明                                                      |
| ------------------------------------ | --------------------------------------------------------- |
| `--source`                           | 输入源，`0/1/...` 为摄像头索引，也可填写图片或视频路径    |
| `--imgsz`                            | 推理分辨率，整数表示最长边缩放到该值                      |
| `--flip`                             | 画面翻转模式（`1` 水平、`-1` 垂直、`0` 同时、`2` 不翻转） |
| `--conf`                             | 置信度阈值                                                |
| `--save`                             | 是否保存结果（图片/视频/摄像头录制）                      |
| `--output`                           | 保存目录（默认 `./test_result`）                          |
| `--jetson`                           | Jetson CSI 摄像头专用管线                                 |
| `--cam-width/--cam-height/--cam-fps` | 摄像头采集参数                                            |

摄像头模式下可用快捷键：`q` 退出、`s` 保存当前帧、`r` 开始/停止录制（需开启 `--save`）。

---

## 兼容旧脚本

为避免影响已有脚本，顶层仍保留原始文件名：

- `python yolo_autolabel.py --help` 等价于执行 `label` 子命令。
- `python yolo_detect.py --help` 等价于执行 `detect` 子命令。

因此旧有的调用方式仍然可用，同时也推荐迁移到新的统一 CLI 以获得更清晰的帮助信息。

---

## 后续工作

- 利用 `train.py` 与 `train_val_split.py` 继续完成模型训练与数据集划分。
- 根据需要扩展 `autolabel_toolkit`，例如增加日志记录或批处理脚本。

如果该项目对你有帮助，欢迎 Star ⭐ 支持！
