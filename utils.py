from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Tuple

import cv2
import numpy as np


def timestamp_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return -1.0
    return float(np.dot(a, b) / denom)


def cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query vector and a matrix of vectors."""
    q = np.asarray(query, dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    if m.size == 0:
        return np.empty((0,), dtype=np.float32)

    q_norm = np.linalg.norm(q)
    m_norm = np.linalg.norm(m, axis=1)
    denom = q_norm * m_norm
    denom = np.where(denom == 0.0, 1e-12, denom)
    return (m @ q) / denom


def save_thumbnail(path: str, image: np.ndarray, quality: int = 90) -> bool:
    """Save a JPEG thumbnail, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return bool(
        cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    )


def sanitize_ts_for_filename(ts: str) -> str:
    """Convert ISO timestamp into filename-safe text."""
    return ts.replace(":", "-").replace("+", "_").replace("T", "_")


def ensure_rtsp_tcp(url: str) -> str:
    """Append a TCP transport hint for RTSP URLs when missing."""
    if not url:
        return url
    lower = url.lower()
    if "rtsp://" not in lower:
        return url
    if "tcp" in lower:
        return url
    return f"{url}&tcp" if "?" in url else f"{url}?tcp"


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-Union for (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)
