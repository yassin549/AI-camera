from __future__ import annotations

import io
from typing import Optional, Tuple

import cv2
import face_recognition
import numpy as np


def serialize_embedding(embedding: np.ndarray) -> bytes:
    arr = np.asarray(embedding, dtype=np.float32)
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
    ) -> None:
        self.padding_ratio = padding_ratio
        self.num_jitters = num_jitters
        self.model = model

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
            return None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        locations = [(y1, x2, y2, x1)]
        encodings = face_recognition.face_encodings(
            rgb,
            known_face_locations=locations,
            num_jitters=self.num_jitters,
            model=self.model,
        )
        if not encodings:
            return None
        return np.asarray(encodings[0], dtype=np.float32)
