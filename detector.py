from __future__ import annotations

from typing import List, Tuple

import cv2
import mediapipe as mp


class FaceDetector:
    """MediaPipe face detector wrapper for CPU-friendly detection."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self._mp_face_detection = mp.solutions.face_detection
        self._detector = self._mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )

    def detect_faces(self, frame_bgr) -> List[Tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        if not results.detections:
            return []

        boxes: List[Tuple[int, int, int, int]] = []
        for detection in results.detections:
            rel_box = detection.location_data.relative_bounding_box
            x1 = max(0, int(rel_box.xmin * w))
            y1 = max(0, int(rel_box.ymin * h))
            x2 = min(w, int((rel_box.xmin + rel_box.width) * w))
            y2 = min(h, int((rel_box.ymin + rel_box.height) * h))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes

    def close(self) -> None:
        self._detector.close()
