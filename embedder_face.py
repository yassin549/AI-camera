"""Face embedding helper using face_recognition / dlib."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from utils import l2_normalize

LOGGER = logging.getLogger(__name__)

try:
    import face_recognition
except Exception as exc:  # pragma: no cover - import guard
    face_recognition = None
    LOGGER.warning("face_recognition unavailable: %s", exc)


class FaceEmbedder:
    """Wrap face_recognition.face_encodings and return 128-D float32 vectors."""

    def __init__(self, num_jitters: int = 1, model: str = "small") -> None:
        self.num_jitters = int(num_jitters)
        self.model = model

    def embed_from_bgr(self, bgr_face: np.ndarray) -> Optional[np.ndarray]:
        if bgr_face is None or bgr_face.size == 0:
            return None

        # Keep minimum workable resolution for dlib encoder.
        face = bgr_face
        h, w = face.shape[:2]
        if min(h, w) < 64:
            scale = 64.0 / float(min(h, w))
            face = cv2.resize(face, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        if face_recognition is not None:
            encodings = face_recognition.face_encodings(
                rgb,
                known_face_locations=None,
                num_jitters=self.num_jitters,
                model=self.model,
            )
            if encodings:
                return l2_normalize(np.asarray(encodings[0], dtype=np.float32))

        # Deterministic fallback if dlib is missing or no face encoding produced.
        resized = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
        vec = resized.astype(np.float32).reshape(-1)
        if vec.size >= 128:
            vec = vec[:128]
        else:
            vec = np.pad(vec, (0, 128 - vec.size), mode="constant")
        return l2_normalize(vec)
