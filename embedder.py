from __future__ import annotations

import io
import logging
from typing import Optional, Tuple

import cv2
import face_recognition
import numpy as np

from utils import normalize_embedding

logger = logging.getLogger("embedder")


def serialize_embedding(embedding: np.ndarray) -> bytes:
    arr = normalize_embedding(embedding)
    with io.BytesIO() as buf:
        np.save(buf, arr, allow_pickle=False)
        return buf.getvalue()


def deserialize_embedding(blob_bytes: bytes) -> np.ndarray:
    with io.BytesIO(blob_bytes) as buf:
        arr = np.load(buf, allow_pickle=False)
    return np.asarray(arr, dtype=np.float32)


class FaceEmbedder:
    """face_recognition wrapper for 128-d embeddings."""

    def __init__(
        self,
        padding_ratio: float = 0.2,
        num_jitters: int = 1,
        model: str = "small",
        debug: bool = False,
    ) -> None:
        self.padding_ratio = padding_ratio
        self.num_jitters = num_jitters
        self.model = model
        self.debug = debug

    def extract_face_crop(
        self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = box
        h, w = frame_bgr.shape[:2]

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        px = int(bw * self.padding_ratio)
        py = int(bh * self.padding_ratio)

        cx1 = max(0, x1 - px)
        cy1 = max(0, y1 - py)
        cx2 = min(w, x2 + px)
        cy2 = min(h, y2 + py)

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        return crop, (cx1, cy1, cx2, cy2)

    def compute_embedding(
        self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        _, (x1, y1, x2, y2) = self.extract_face_crop(frame_bgr, box)
        if x2 <= x1 or y2 <= y1:
            if self.debug:
                logger.info("Embedder rejected invalid crop box=%s", (x1, y1, x2, y2))
            return None

        if self.debug:
            logger.info("Embedder converting BGR->RGB for box=%s", (x1, y1, x2, y2))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        locations = [(y1, x2, y2, x1)]
        encodings = face_recognition.face_encodings(
            rgb,
            known_face_locations=locations,
            num_jitters=self.num_jitters,
            model=self.model,
        )
        if not encodings:
            if self.debug:
                logger.info("Embedder returned empty face_encodings for box=%s", (x1, y1, x2, y2))
            return None
        normalized = normalize_embedding(np.asarray(encodings[0], dtype=np.float32))
        if self.debug:
            logger.info(
                "Embedder embedding shape=%s dtype=%s norm=%.6f",
                tuple(normalized.shape),
                normalized.dtype,
                float(np.linalg.norm(normalized)),
            )
        return normalized
