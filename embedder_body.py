"""Body descriptor extraction from YOLO feature maps with fallback model."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from utils import l2_normalize

LOGGER = logging.getLogger(__name__)


class MobileNetBodyFallback:
    """Tiny body descriptor fallback using a MobileNetV2 ONNX model."""

    def __init__(self, model_path: Optional[str], input_size: Tuple[int, int] = (128, 256)) -> None:
        self.model_path = model_path
        self.input_w = int(input_size[0])
        self.input_h = int(input_size[1])
        self.session: Optional[ort.InferenceSession] = None
        self.active = False

        if not model_path:
            LOGGER.warning("Body fallback model path is empty, using color-hist fallback.")
            return

        try:
            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.session = ort.InferenceSession(
                model_path,
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            self.active = True
            LOGGER.info("Body descriptor path: MobileNetV2 fallback (%s)", model_path)
        except Exception as exc:
            LOGGER.warning("Could not load MobileNet fallback model: %s", exc)

    def embed(self, bgr_crop: np.ndarray) -> np.ndarray:
        if bgr_crop is None or bgr_crop.size == 0:
            return np.zeros((256,), dtype=np.float32)

        if self.session is None:
            # Low-cost fallback descriptor to preserve runtime behavior.
            hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
            flat = hist.reshape(-1).astype(np.float32)
            return l2_normalize(flat)

        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: tensor})[0]
        flat = np.asarray(output, dtype=np.float32).reshape(-1)

        if flat.size >= 256:
            flat = flat[:256]
        else:
            flat = np.pad(flat, (0, 256 - flat.size), mode="constant")
        return l2_normalize(flat)


def body_descriptor_from_backbone(
    backbone_feature: np.ndarray,
    det_bbox_input_xyxy: Tuple[int, int, int, int],
    model_input_size: Tuple[int, int],
    target_dim: int = 256,
) -> np.ndarray:
    """Extract detection-level descriptor by pooling a feature-map ROI."""
    feat = np.asarray(backbone_feature, dtype=np.float32)

    if feat.ndim == 3:
        # C,H,W
        c, fh, fw = feat.shape
    elif feat.ndim == 4:
        # N,C,H,W -> use first batch
        feat = feat[0]
        c, fh, fw = feat.shape
    else:
        raise ValueError(f"Unsupported feature shape: {feat.shape}")

    in_w, in_h = int(model_input_size[0]), int(model_input_size[1])
    x1, y1, x2, y2 = det_bbox_input_xyxy

    fx1 = max(0, min(fw - 1, int((x1 / max(1, in_w)) * fw)))
    fx2 = max(fx1 + 1, min(fw, int((x2 / max(1, in_w)) * fw)))
    fy1 = max(0, min(fh - 1, int((y1 / max(1, in_h)) * fh)))
    fy2 = max(fy1 + 1, min(fh, int((y2 / max(1, in_h)) * fh)))

    roi = feat[:, fy1:fy2, fx1:fx2]
    if roi.size == 0:
        pooled = np.zeros((c,), dtype=np.float32)
    else:
        pooled = roi.mean(axis=(1, 2)).astype(np.float32)

    if pooled.size >= target_dim:
        pooled = pooled[:target_dim]
    else:
        pooled = np.pad(pooled, (0, target_dim - pooled.size), mode="constant")

    return l2_normalize(pooled)
