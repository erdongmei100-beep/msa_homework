"""Command line entrypoint for the HyperLPR3 plate recognition module."""

from __future__ import annotations

import argparse

from plate_recognition.pipeline import recognize_plates_for_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSV 批量车牌识别工具")
    parser.add_argument("--csv-path", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--images-dir", required=True, help="整帧图片所在目录")
    parser.add_argument("--output-csv", required=True, help="输出 CSV 文件路径")
    parser.add_argument("--image-column", default="image_name", help="CSV 中图片文件名的列名")
    parser.add_argument("--plate-column", default="plate_text", help="输出 CSV 车牌列名")
    parser.add_argument("--log-file", default=None, help="日志文件路径，可选")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="优先使用的设备类型",
    )
    parser.add_argument(
        "--save-plate-images",
        action="store_true",
        help="是否保存车牌裁剪图像",
    )
    parser.add_argument(
        "--plate-image-dir",
        default="outputs/plates",
        help="保存车牌裁剪图像的目录",
    )
    parser.add_argument(
        "--plate-image-column",
        default="plate_image_path",
        help="CSV 中保存车牌裁剪路径的列名",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    recognize_plates_for_csv(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        output_csv_path=args.output_csv,
        image_column=args.image_column,
        plate_column=args.plate_column,
        log_path=args.log_file,
        device=args.device,
        save_plate_images=args.save_plate_images,
        plate_image_dir=args.plate_image_dir,
        plate_image_column=args.plate_image_column,
    )


if __name__ == "__main__":
    main()

