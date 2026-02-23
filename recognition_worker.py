"""Event-driven recognition worker for face/body identity linking."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import db
from embedder_face import FaceEmbedder
from face_roi_detector import FaceROIDetector
from utils import CounterFPS, FrameTimingStore, LatestFrameStore, SharedTrackStore, StageStats, ensure_dir, timestamp_iso

LOGGER = logging.getLogger(__name__)


class RecognitionWorker:
    """Background worker for event-triggered recognition and identity persistence."""

    def __init__(
        self,
        frame_store: LatestFrameStore,
        track_store: SharedTrackStore,
        cfg: Dict[str, object],
        timing_store: FrameTimingStore,
        stage_stats: StageStats,
        fps_counter: CounterFPS,
        stop_event: threading.Event,
    ) -> None:
        self.frame_store = frame_store
        self.track_store = track_store
        self.cfg = cfg
        self.timing_store = timing_store
        self.stage_stats = stage_stats
        self.fps_counter = fps_counter
        self.stop_event = stop_event

        self.face_detector = FaceROIDetector(min_confidence=float(cfg.get("face_roi_min_confidence", 0.5)))
        self.face_embedder = FaceEmbedder()

        self.face_interval = int(cfg.get("face_interval_frames", 10))
        self.body_interval = int(cfg.get("body_interval_frames", 10))
        self.face_threshold = float(cfg.get("face_threshold", 0.60))
        self.body_threshold = float(cfg.get("body_threshold", 0.40))
        self.cache_seconds = float(cfg.get("cache_seconds", 30))
        self.top_ratio = float(cfg.get("face_top_ratio", 0.50))
        self.disable_body_embedding = bool(cfg.get("disable_body_embedding", True))
        self.persist_body_only = bool(cfg.get("persist_body_only", False))
        self.faces_dir = ensure_dir(str(cfg.get("faces_dir", "./faces")))

        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="recognition-worker")
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.face_detector.close()

    def _run(self) -> None:
        last_frame_id = -1
        while not self.stop_event.is_set():
            packet = self.frame_store.wait_for_new(last_frame_id, timeout=0.05)
            if packet is None:
                continue

            frame = packet.frame
            frame_id = int(packet.frame_id)
            last_frame_id = frame_id

            rec_ms_total = 0.0
            tracks = self.track_store.snapshot()
            now = time.time()

            for st in tracks:
                # Ignore stale tracks.
                if frame_id - int(st.last_seen_frame) > 2:
                    continue

                # Cache gate for already-linked track IDs.
                if st.identity_id is not None and now < float(st.cache_until):
                    continue

                # Event-driven trigger: new track or unresolved track at rate limit.
                should_try_face = bool(st.is_new) or st.identity_id is None
                if should_try_face and (frame_id - int(st.last_face_frame) >= self.face_interval):
                    t0 = time.perf_counter()
                    self.track_store.mark_face_checked(st.track_id, frame_id)
                    face_emb = self._extract_face_embedding(frame, st.bbox)
                    if face_emb is not None:
                        self._match_or_create_face(st.track_id, face_emb, frame, st.bbox, frame_id)
                    rec_ms_total += (time.perf_counter() - t0) * 1000.0
                    continue

                # Optional body-only association path.
                if not self.disable_body_embedding and st.feature is not None:
                    if frame_id - int(st.last_body_frame) >= self.body_interval:
                        t0 = time.perf_counter()
                        self.track_store.mark_body_checked(st.track_id, frame_id)
                        self._match_body(st.track_id, st.feature, frame_id)
                        rec_ms_total += (time.perf_counter() - t0) * 1000.0

            self.timing_store.set(frame_id, "recognition_ms", rec_ms_total)
            self.stage_stats.add("recognition_ms", rec_ms_total)
            self.fps_counter.inc(1)

    def _extract_face_embedding(
        self,
        frame_bgr: np.ndarray,
        person_bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [int(v) for v in person_bbox]
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))

        top_h = max(1, int((y2 - y1) * self.top_ratio))
        roi = frame_bgr[y1:y1 + top_h, x1:x2]
        if roi.size == 0:
            return None

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        faces = self.face_detector.detect(roi_rgb)
        if not faces:
            return None

        fx1, fy1, fx2, fy2, _score = max(faces, key=lambda v: v[4])
        ax1 = max(0, x1 + int(fx1))
        ay1 = max(0, y1 + int(fy1))
        ax2 = min(w, x1 + int(fx2))
        ay2 = min(h, y1 + int(fy2))
        if ax2 <= ax1 or ay2 <= ay1:
            return None

        face_crop = frame_bgr[ay1:ay2, ax1:ax2]
        if face_crop.size == 0:
            return None

        return self.face_embedder.embed_from_bgr(face_crop)

    def _match_or_create_face(
        self,
        track_id: int,
        face_emb: np.ndarray,
        frame_bgr: np.ndarray,
        person_bbox: Tuple[int, int, int, int],
        frame_id: int,
    ) -> None:
        now_ts = timestamp_iso()

        best_id, best_score = db.find_best_face(face_emb)
        if best_id is not None and best_score >= self.face_threshold:
            db.update_last_seen(int(best_id), now_ts)
            self.track_store.assign_identity(
                track_id=track_id,
                identity_id=int(best_id),
                modality="face",
                score=float(best_score),
                frame_id=frame_id,
                cache_seconds=self.cache_seconds,
            )
            return

        x1, y1, x2, y2 = [int(v) for v in person_bbox]
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        sample = frame_bgr[y1:y2, x1:x2]

        # Create identity first so we can use the ID in the filename.
        new_id = db.add_identity(face_emb, None, None, None, now_ts)

        # Save the face sample with identity ID in the filename so that
        # _discover_face_thumb() can find it via the <id>_*.jpg glob pattern.
        sample_path: Optional[str] = None
        if sample.size > 0:
            ts_str = time.strftime("%Y-%m-%d_%H-%M-%S")
            sample_path = str(Path(self.faces_dir) / f"{int(new_id)}_{ts_str}.jpg")
            cv2.imwrite(sample_path, sample)
            # Update the DB row with the sample path.
            db.update_face_sample_path(int(new_id), sample_path)

        self.track_store.assign_identity(
            track_id=track_id,
            identity_id=int(new_id),
            modality="face",
            score=1.0,
            frame_id=frame_id,
            cache_seconds=self.cache_seconds,
        )

    def _match_body(self, track_id: int, body_emb: np.ndarray, frame_id: int) -> None:
        now_ts = timestamp_iso()
        best_id, best_score = db.find_best_body(body_emb)
        if best_id is not None and best_score >= self.body_threshold:
            db.update_last_seen(int(best_id), now_ts)
            db.update_body_ema(int(best_id), body_emb)
            self.track_store.assign_identity(
                track_id=track_id,
                identity_id=int(best_id),
                modality="body",
                score=float(best_score),
                frame_id=frame_id,
                cache_seconds=self.cache_seconds,
            )
            return

        if self.persist_body_only:
            new_id = db.add_identity(None, body_emb, None, None, now_ts)
            self.track_store.assign_identity(
                track_id=track_id,
                identity_id=int(new_id),
                modality="body",
                score=1.0,
                frame_id=frame_id,
                cache_seconds=self.cache_seconds,
            )
