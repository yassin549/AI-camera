"""ByteTrack adapter with predict/update split for frame-skipping pipelines."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import iou

LOGGER = logging.getLogger(__name__)

try:
    import supervision as sv
except Exception as exc:  # pragma: no cover - optional import
    sv = None
    LOGGER.warning("supervision unavailable, using simple fallback tracker: %s", exc)


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    score: float
    feature: Optional[np.ndarray] = None


class _SimpleTracker:
    """Fallback IoU tracker with explicit predict/update cycle."""

    def __init__(self, max_age: int = 30) -> None:
        self.max_age = int(max_age)
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, object]] = {}

    def predict(self) -> List[Track]:
        remove_ids = []
        out: List[Track] = []
        for tid, st in self.tracks.items():
            st["age"] = int(st.get("age", 0)) + 1
            if int(st["age"]) > self.max_age:
                remove_ids.append(tid)
                continue
            out.append(
                Track(
                    track_id=int(tid),
                    bbox=tuple(st["bbox"]),
                    score=float(st.get("score", 0.0)),
                    feature=st.get("feature"),
                )
            )
        for tid in remove_ids:
            self.tracks.pop(tid, None)
        return out

    def update(self, detections: List[Track]) -> List[Track]:
        used = set()
        out: List[Track] = []

        for det in detections:
            best_id = None
            best_iou = 0.0
            for tid, st in self.tracks.items():
                if tid in used:
                    continue
                score = iou(det.bbox, tuple(st["bbox"]))
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_id is None or best_iou < 0.25:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {
                    "bbox": det.bbox,
                    "score": det.score,
                    "feature": det.feature,
                    "age": 0,
                }
            else:
                self.tracks[best_id]["bbox"] = det.bbox
                self.tracks[best_id]["score"] = det.score
                self.tracks[best_id]["feature"] = det.feature
                self.tracks[best_id]["age"] = 0
            used.add(best_id)
            out.append(Track(track_id=best_id, bbox=det.bbox, score=det.score, feature=det.feature))

        return out


class TrackerAdapter:
    """Unified tracker adapter (ByteTrack primary, fallback tracker secondary)."""

    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8, max_tracks: int = 20) -> None:
        self.track_thresh = float(track_thresh)
        self.match_thresh = float(match_thresh)
        self.max_tracks = int(max_tracks)

        self.bytetrack = None
        self.fallback = None
        self._latest_tracks: List[Track] = []
        self._last_detections: List[Track] = []
        self._track_features: Dict[int, np.ndarray] = {}
        self._last_update_ts = time.time()

        if sv is not None:
            try:
                self.bytetrack = sv.ByteTrack(
                    track_activation_threshold=self.track_thresh,
                    minimum_matching_threshold=self.match_thresh,
                    frame_rate=30,
                )
                LOGGER.info("Tracker mode: ByteTrack")
            except Exception as exc:
                LOGGER.warning("ByteTrack init failed, using fallback tracker: %s", exc)
                self.fallback = _SimpleTracker(max_age=45)
        else:
            self.fallback = _SimpleTracker(max_age=45)

    def update(self, detections: List[Track]) -> List[Track]:
        """Update tracker on detection frames."""
        self._last_update_ts = time.time()
        if detections:
            self._last_detections = [
                Track(track_id=-1, bbox=tuple(det.bbox), score=float(det.score), feature=det.feature)
                for det in detections
            ]
        else:
            self._last_detections = []
        if self.bytetrack is not None:
            tracked = self._run_bytetrack(detections)
        else:
            assert self.fallback is not None
            tracked = self.fallback.update(detections)

        self._refresh_track_feature_cache(tracked)
        self._latest_tracks = tracked[: self.max_tracks]
        return list(self._latest_tracks)

    def predict(self) -> List[Track]:
        """Advance tracker on non-detection frames without starving tracker state."""
        if self.bytetrack is not None:
            # Let ByteTrack run its own motion model when no fresh detections are available.
            tracked = self._run_bytetrack([])
            if not tracked and self._latest_tracks:
                # Defensive fallback for supervision versions that return no tracks on empty updates.
                tracked = [
                    Track(
                        track_id=int(tr.track_id),
                        bbox=tuple(tr.bbox),
                        score=max(0.05, float(tr.score) * 0.97),
                        feature=tr.feature,
                    )
                    for tr in self._latest_tracks
                ]
        else:
            assert self.fallback is not None
            tracked = self.fallback.predict()
        self._refresh_track_feature_cache(tracked)
        self._latest_tracks = tracked[: self.max_tracks]
        return list(self._latest_tracks)

    def current_tracks(self) -> List[Track]:
        return list(self._latest_tracks)

    def supports_motion_prediction(self) -> bool:
        return self.bytetrack is not None

    def _run_bytetrack(self, detections: List[Track]) -> List[Track]:
        assert self.bytetrack is not None

        if detections:
            xyxy = np.asarray([det.bbox for det in detections], dtype=np.float32)
            conf = np.asarray([det.score for det in detections], dtype=np.float32)
            class_id = np.zeros((len(detections),), dtype=np.int32)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)
        else:
            sv_dets = sv.Detections.empty()

        try:
            tracked = self.bytetrack.update_with_detections(sv_dets)
        except AttributeError:
            tracked = self.bytetrack.update(sv_dets)

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            return []

        out: List[Track] = []
        for i in range(len(tracked.xyxy)):
            tid = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = [int(v) for v in tracked.xyxy[i]]
            score = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            feature: Optional[np.ndarray] = None
            if detections:
                best_iou = 0.0
                best_feature: Optional[np.ndarray] = None
                for det in detections:
                    overlap = iou((x1, y1, x2, y2), tuple(det.bbox))
                    if overlap > best_iou:
                        best_iou = overlap
                        best_feature = det.feature
                if best_iou >= 0.10 and best_feature is not None:
                    feature = best_feature
            if feature is None:
                feature = self._track_features.get(tid)
            out.append(Track(track_id=tid, bbox=(x1, y1, x2, y2), score=score, feature=feature))
        return out

    def _refresh_track_feature_cache(self, tracks: List[Track]) -> None:
        active: Dict[int, np.ndarray] = {}
        for tr in tracks:
            tid = int(tr.track_id)
            if tr.feature is not None:
                active[tid] = tr.feature
                continue
            cached = self._track_features.get(tid)
            if cached is not None:
                tr.feature = cached
                active[tid] = cached
        self._track_features = active
