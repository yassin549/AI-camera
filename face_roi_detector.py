"""MediaPipe face detector restricted to person-top ROI."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover - import guard
    mp = None
    LOGGER.warning("MediaPipe unavailable: %s", exc)


class FaceROIDetector:
    """Detect faces in an RGB ROI and return local ROI-space bounding boxes."""

    def __init__(self, min_confidence: float = 0.5, model_selection: int = 0) -> None:
        self.min_confidence = float(min_confidence)
        self.model_selection = int(model_selection)
        self._detector = None
        if mp is not None:
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=self.model_selection,
                min_detection_confidence=self.min_confidence,
            )

    def detect(self, rgb_roi: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Return list of (x1, y1, x2, y2, score) inside the ROI."""
        if self._detector is None or rgb_roi is None or rgb_roi.size == 0:
            return []

        h, w = rgb_roi.shape[:2]
        result = self._detector.process(rgb_roi)
        if result.detections is None:
            return []

        out: List[Tuple[int, int, int, int, float]] = []
        for det in result.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))
            score = float(det.score[0]) if det.score else 0.0
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2, score))
        return out

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
