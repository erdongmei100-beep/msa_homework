# 车牌文本识别模块

这是一个基于 [HyperLPR3](https://github.com/szad670401/HyperLPR) 的车牌文本识别模块，用于从事件列表 CSV 和整帧截图中批量补全车牌号码。模块提供可复用函数和命令行入口，支持 GPU 优先运行并在不可用时回退到 CPU。

## 环境要求
- Python 3.10
- Linux/WSL
- 可选 GPU（安装 `onnxruntime-gpu` 后自动尝试使用 CUDAExecutionProvider，缺失时自动回退 CPU）

## 安装依赖
```bash
pip install -r requirements.txt
```
若需要 GPU 加速，请确保安装了 CUDA 相关驱动，并替换/安装 `onnxruntime-gpu`。

## 使用示例
```bash
python main.py \
  --csv-path data/events.csv \
  --images-dir data/images \
  --output-csv data/events_with_plate.csv \
  --log-file logs/plate_rec.log
```

常用参数说明：
- `--image-column`：CSV 中表示图片文件名的列名，默认 `image_name`。
- `--plate-column`：输出 CSV 新增车牌列名，默认 `plate_text`。
- `--device`：`auto` | `cpu` | `gpu`，默认自动选择（优先 GPU）。
- `--save-plate-images`：开启后保存车牌裁剪图像到 `--plate-image-dir`，并在 CSV 中写入 `--plate-image-column` 列。

## CSV 输入输出格式
- 输入 CSV 至少包含一列存放图片文件名，默认列名 `image_name`，可通过参数 `--image-column` 调整。
- 输出 CSV 会在原有列基础上追加车牌列（默认 `plate_text`）。若未检测到车牌则写入空字符串。
- 若启用车牌裁剪保存，将额外追加一列（默认 `plate_image_path`）记录裁剪图像路径。

## 模块函数
代码核心函数位于 `plate_recognition/pipeline.py`：
```python
from plate_recognition import recognize_plates_for_csv

recognize_plates_for_csv(
    csv_path="data/events.csv",
    images_dir="data/images",
    output_csv_path="data/events_with_plate.csv",
)
```
可在其他项目中直接导入调用。
