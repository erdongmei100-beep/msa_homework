"""Core pipeline for running HyperLPR3 license plate recognition on CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import cv2

from .config import (
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_COLUMN,
    DEFAULT_PLATE_COLUMN,
    DEFAULT_PLATE_IMAGE_COLUMN,
    DEFAULT_PROGRESS_INTERVAL,
)
from .utils import (
    HyperLPRNotInstalledError,
    PlateCandidate,
    build_license_plate_catcher,
    crop_plate_image,
    ensure_directory,
    load_image,
    parse_candidates,
    select_best_candidate,
    setup_logger,
)


def recognize_plates_for_csv(
    csv_path: str,
    images_dir: str,
    output_csv_path: str,
    *,
    image_column: str = DEFAULT_IMAGE_COLUMN,
    plate_column: str = DEFAULT_PLATE_COLUMN,
    log_path: str | None = None,
    device: str = DEFAULT_DEVICE,
    save_plate_images: bool = False,
    plate_image_dir: str = "outputs/plates",
    plate_image_column: str = DEFAULT_PLATE_IMAGE_COLUMN,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> None:
    """读取 CSV 与图片目录，补全车牌文本并生成新的 CSV 文件。

    Parameters
    ----------
    csv_path:
        输入 CSV 文件路径。
    images_dir:
        整帧图片所在目录。
    output_csv_path:
        输出 CSV 文件路径。
    image_column:
        CSV 中表示图片文件名的列名，可配置。
    plate_column:
        输出 CSV 中新增的车牌文本列名。
    log_path:
        日志文件路径，若为 ``None`` 则只打印到控制台。
    device:
        设备偏好，"auto" | "cpu" | "gpu"。当前版本记录偏好并尝试自动选择。
    save_plate_images:
        是否将识别出的车牌区域裁剪并保存。
    plate_image_dir:
        保存车牌裁剪图像的目录路径。
    plate_image_column:
        在 CSV 中记录车牌裁剪图像路径的列名。
    progress_interval:
        处理进度日志的间隔行数。
    """

    logger = setup_logger(log_path)

    logger.info(
        "参数: csv_path=%s, images_dir=%s, output_csv_path=%s, image_column=%s, plate_column=%s, device=%s, save_plate_images=%s",
        csv_path,
        images_dir,
        output_csv_path,
        image_column,
        plate_column,
        device,
        save_plate_images,
    )

    try:
        catcher = build_license_plate_catcher(device=device, logger=logger)
    except HyperLPRNotInstalledError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("初始化 HyperLPR 失败")
        raise exc

    df = pd.read_csv(csv_path)
    if plate_column not in df.columns:
        df[plate_column] = ""
    if save_plate_images and plate_image_column not in df.columns:
        df[plate_image_column] = ""

    images_root = Path(images_dir)
    output_csv = Path(output_csv_path)
    ensure_directory(output_csv.parent)
    if save_plate_images:
        ensure_directory(plate_image_dir)

    total_rows = len(df)
    success_count = 0
    failure_count = 0

    for idx, row in df.iterrows():
        image_name = row.get(image_column)
        if not isinstance(image_name, str):
            logger.error("第 %d 行缺少图片文件名列 %s", idx, image_column)
            df.at[idx, plate_column] = ""
            failure_count += 1
            continue

        image_path = images_root / image_name
        if not image_path.exists():
            logger.error("图片文件不存在: %s (行 %d)", image_path, idx)
            df.at[idx, plate_column] = ""
            failure_count += 1
            continue

        image = load_image(str(image_path), logger=logger)
        if image is None:
            df.at[idx, plate_column] = ""
            failure_count += 1
            continue

        try:
            results = catcher(image)
        except Exception:
            logger.exception("HyperLPR 处理异常 (行 %d, 图片 %s)", idx, image_path)
            df.at[idx, plate_column] = ""
            failure_count += 1
            continue

        candidates = parse_candidates(results)
        best: Optional[PlateCandidate] = select_best_candidate(candidates)

        if best and best.text:
            df.at[idx, plate_column] = best.text
            success_count += 1

            if save_plate_images and best.bbox:
                # HyperLPR3 通常会返回 bbox，若缺失则跳过裁剪。
                try:
                    crop = crop_plate_image(image, best.bbox, padding=0.08)
                    crop_filename = f"{Path(image_name).stem}_plate.jpg"
                    crop_path = Path(plate_image_dir) / crop_filename
                    cv2.imwrite(str(crop_path), crop)
                    df.at[idx, plate_image_column] = str(crop_path)
                except Exception:
                    logger.exception("保存车牌裁剪失败 (行 %d, 图片 %s)", idx, image_path)
        else:
            df.at[idx, plate_column] = ""
            failure_count += 1
            logger.warning("未检测到车牌 (行 %d, 图片 %s)", idx, image_path)

        if progress_interval and (idx + 1) % progress_interval == 0:
            logger.info("已处理 %d/%d 行", idx + 1, total_rows)

    df.to_csv(output_csv, index=False)
    logger.info(
        "处理完成: 总行数=%d, 识别成功=%d, 识别失败=%d, 输出文件=%s",
        total_rows,
        success_count,
        failure_count,
        output_csv,
    )

