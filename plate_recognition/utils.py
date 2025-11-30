"""Utility helpers for the plate recognition pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None


@dataclass
class PlateCandidate:
    """Container for a single license plate prediction.

    Attributes
    ----------
    text:
        Predicted license plate string.
    score:
        Confidence score of the prediction.
    bbox:
        Optional bounding box represented as ``(x1, y1, x2, y2)`` in pixel
        coordinates.
    """

    text: Optional[str]
    score: Optional[float]
    bbox: Optional[Tuple[int, int, int, int]] = None


class HyperLPRNotInstalledError(ImportError):
    """Custom exception raised when hyperlpr3 is not available."""


def setup_logger(log_path: str | None = None) -> logging.Logger:
    """Configure the root logger with console and optional file handlers."""

    logger = logging.getLogger("plate_recognition")
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers in re-entrant calls.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_directory(path: str | os.PathLike[str]) -> None:
    """Create a directory if it does not exist."""

    Path(path).mkdir(parents=True, exist_ok=True)


def load_image(image_path: str, logger: logging.Logger) -> Optional[np.ndarray]:
    """Load an image from disk using OpenCV.

    Parameters
    ----------
    image_path:
        Absolute path to the image.
    logger:
        Logger instance for reporting errors.

    Returns
    -------
    numpy.ndarray | None
        The BGR image array if successfully loaded; otherwise ``None``.
    """

    image = cv2.imread(image_path)
    if image is None:
        logger.error("无法读取图片: %s", image_path)
        return None
    return image


def _extract_bbox(candidate: Sequence[int]) -> Optional[Tuple[int, int, int, int]]:
    """Normalize a bounding box sequence into (x1, y1, x2, y2)."""

    if len(candidate) == 8:
        xs = candidate[0::2]
        ys = candidate[1::2]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    if len(candidate) >= 4:
        x1, y1, x2, y2 = candidate[:4]
        return int(x1), int(y1), int(x2), int(y2)
    return None


def parse_candidates(result: Iterable) -> List[PlateCandidate]:
    """Normalize HyperLPR outputs into a list of :class:`PlateCandidate`.

    The HyperLPR3 API returns a list. Each element is often a dictionary with
    keys like ``text``, ``score`` and ``box``. To be defensive, tuple results are
    also supported where ``(text, score, bbox)`` are expected.
    """

    candidates: List[PlateCandidate] = []
    for entry in result or []:
        text: Optional[str] = None
        score: Optional[float] = None
        bbox: Optional[Tuple[int, int, int, int]] = None

        if isinstance(entry, dict):
            text = entry.get("text") or entry.get("plate")
            score = entry.get("score") or entry.get("confidence")
            raw_box = entry.get("box") or entry.get("bbox") or entry.get("position")
            if isinstance(raw_box, (list, tuple)):
                bbox = _extract_bbox(raw_box)
        elif isinstance(entry, (list, tuple)):
            if len(entry) >= 1:
                text = entry[0]
            if len(entry) >= 2:
                score = entry[1]
            if len(entry) >= 3 and isinstance(entry[2], (list, tuple)):
                bbox = _extract_bbox(entry[2])

        candidates.append(PlateCandidate(text=text, score=score, bbox=bbox))

    return candidates


def select_best_candidate(candidates: Sequence[PlateCandidate]) -> Optional[PlateCandidate]:
    """Pick the candidate with the highest confidence score."""

    if not candidates:
        return None

    scored_candidates = [c for c in candidates if c.text]
    if not scored_candidates:
        return None

    return max(scored_candidates, key=lambda c: c.score or 0.0)


def crop_plate_image(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.05,
) -> np.ndarray:
    """Crop the license plate region from an image with optional padding."""

    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return image[y1:y2, x1:x2]


def detect_available_execution_provider() -> str:
    """Return the preferred execution provider based on availability."""

    if ort is None:
        return "cpu"

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return "gpu"
    return "cpu"


def build_license_plate_catcher(device: str, logger: logging.Logger):
    """Instantiate HyperLPR3's LicensePlateCatcher with graceful fallbacks."""

    try:
        import hyperlpr3 as lpr3
    except ImportError as exc:  # pragma: no cover - library required at runtime
        raise HyperLPRNotInstalledError(
            "hyperlpr3 未安装，请先运行 pip install -r requirements.txt"
        ) from exc

    preferred_device = device
    available = detect_available_execution_provider()
    if device == "auto":
        preferred_device = available
    elif device == "gpu" and available != "gpu":
        logger.warning("未检测到可用 GPU，自动回退至 CPU 执行")
        preferred_device = available

    logger.info("优先使用设备: %s", preferred_device)

    catcher = lpr3.LicensePlateCatcher()

    if ort is not None:
        logger.info("ONNXRuntime 可用的执行提供者: %s", ort.get_available_providers())
    else:
        logger.info("未检测到 onnxruntime，按 CPU 运行")

    return catcher
