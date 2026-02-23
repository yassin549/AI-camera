from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp

logger = logging.getLogger("detector")


@dataclass(frozen=True)
class FaceDetection:
    box: Tuple[int, int, int, int]
    confidence: float
    relative_box: Tuple[float, float, float, float]


class FaceDetector:
    """MediaPipe face detector wrapper for CPU-friendly detection."""

    def __init__(
        self,
        min_detection_confidence: float = 0.3,
        model_selection: int = 0,
        debug: bool = False,
    ) -> None:
        if model_selection not in (0, 1):
            raise ValueError("model_selection must be 0 (short-range) or 1 (full-range)")
        self._mp_face_detection = mp.solutions.face_detection
        self._min_detection_confidence = float(min_detection_confidence)
        self._model_selection = int(model_selection)
        self._debug = bool(debug)
        self._detector = self._mp_face_detection.FaceDetection(
            model_selection=self._model_selection,
            min_detection_confidence=self._min_detection_confidence,
        )

    def detect(self, frame_bgr) -> List[FaceDetection]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        raw_detections = results.detections or []
        if not raw_detections:
            return []

        detections: List[FaceDetection] = []
        for idx, detection in enumerate(raw_detections):
            rel_box = detection.location_data.relative_bounding_box
            rel_xmin = float(rel_box.xmin)
            rel_ymin = float(rel_box.ymin)
            rel_width = float(rel_box.width)
            rel_height = float(rel_box.height)
            confidence = float(detection.score[0]) if detection.score else 0.0

            x1 = max(0, int(rel_xmin * w))
            y1 = max(0, int(rel_ymin * h))
            x2 = min(w, int((rel_xmin + rel_width) * w))
            y2 = min(h, int((rel_ymin + rel_height) * h))
            if x2 > x1 and y2 > y1:
                detections.append(
                    FaceDetection(
                        box=(x1, y1, x2, y2),
                        confidence=confidence,
                        relative_box=(rel_xmin, rel_ymin, rel_width, rel_height),
                    )
                )
                continue

            if self._debug:
                logger.info(
                    "Detector dropped invalid bbox idx=%d rel=(%.4f, %.4f, %.4f, %.4f) pix=(%d, %d, %d, %d)",
                    idx,
                    rel_xmin,
                    rel_ymin,
                    rel_width,
                    rel_height,
                    x1,
                    y1,
                    x2,
                    y2,
                )

        if self._debug:
            confidences = [f"{det.confidence:.3f}" for det in detections]
            logger.info(
                "Detector output frame=%dx%d detections=%d confidences=%s min_conf=%.2f model=%d",
                w,
                h,
                len(detections),
                confidences,
                self._min_detection_confidence,
                self._model_selection,
            )
        return detections

    def detect_faces(self, frame_bgr) -> List[Tuple[int, int, int, int]]:
        return [det.box for det in self.detect(frame_bgr)]

    def close(self) -> None:
        self._detector.close()
