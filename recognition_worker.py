"""Event-driven recognition worker for face/body identity linking."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import cv2
import numpy as np

import db
from embedder_face import FaceEmbedder
from face_roi_detector import FaceROIDetector
from utils import (
    CounterFPS,
    FrameTimingStore,
    LatestFrameStore,
    RuntimeMetrics,
    SharedTrackStore,
    StageStats,
    ensure_dir,
    timestamp_iso,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class _PersistJob:
    kind: str
    payload: Dict[str, Any]


@dataclass
class _MediaJob:
    identity_id: int
    sample_bgr: np.ndarray


def _coerce_size_tuple(raw: object, fallback: Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        try:
            return (max(16, int(raw[0])), max(16, int(raw[1])))
        except Exception:
            return fallback
    return fallback


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
        metrics: Optional[RuntimeMetrics] = None,
    ) -> None:
        self.frame_store = frame_store
        self.track_store = track_store
        self.cfg = cfg
        self.timing_store = timing_store
        self.stage_stats = stage_stats
        self.fps_counter = fps_counter
        self.stop_event = stop_event
        self.metrics = metrics

        self.face_detector = FaceROIDetector(min_confidence=float(cfg.get("face_roi_min_confidence", 0.5)))
        face_input_size = _coerce_size_tuple(cfg.get("face_onnx_input_size", (112, 112)), (112, 112))
        self.face_embedder = FaceEmbedder(
            backend=str(cfg.get("face_embedder_backend", "auto") or "auto"),
            onnx_model_path=str(cfg.get("face_onnx_model_path", "") or ""),
            onnx_input_size=face_input_size,
            onnx_mean=float(cfg.get("face_onnx_mean", 0.5)),
            onnx_std=float(cfg.get("face_onnx_std", 0.5)),
            onnx_output_name=str(cfg.get("face_onnx_output_name", "") or "") or None,
            onnx_output_index=int(cfg.get("face_onnx_output_index", 0)),
            target_dim=int(cfg.get("face_embedding_dim", 128)),
        )
        self.face_detector_available = bool(self.face_detector.available)

        self.face_interval = int(cfg.get("face_interval_frames", 4))
        self.body_interval = int(cfg.get("body_interval_frames", 10))
        target_fps = max(1.0, float(cfg.get("target_fps", 30.0)))
        self.face_interval_seconds = max(0.01, float(self.face_interval) / target_fps)
        self.body_interval_seconds = max(0.01, float(self.body_interval) / target_fps)
        self.face_threshold = float(cfg.get("face_threshold", 0.60))
        self.body_threshold = float(cfg.get("body_threshold", 0.40))
        self.cache_seconds = float(cfg.get("cache_seconds", 30))
        self.top_ratio = float(cfg.get("face_top_ratio", 0.68))
        self.min_face_pixels = int(cfg.get("face_min_pixels", 40))
        self.disable_body_embedding = bool(cfg.get("disable_body_embedding", False))
        self.persist_body_only = bool(cfg.get("persist_body_only", True))
        self.body_fallback_enabled = bool(cfg.get("body_fallback_enabled", True))
        self.body_fallback_after_face_failures = int(cfg.get("body_fallback_after_face_failures", 3))
        self.faces_dir = ensure_dir(str(cfg.get("faces_dir", "./faces")))
        self._face_failures: Dict[int, int] = {}
        self._last_face_attempt_ts: Dict[int, float] = {}
        self._last_body_attempt_ts: Dict[int, float] = {}

        self.persist_queue_size = max(16, int(cfg.get("recognition_persist_queue_size", 512)))
        self.persist_batch_size = max(1, int(cfg.get("recognition_persist_batch_size", 16)))
        self.media_queue_size = max(8, int(cfg.get("recognition_media_queue_size", 256)))
        self.media_batch_size = max(1, int(cfg.get("recognition_media_batch_size", 8)))

        self._persist_queue: "queue.Queue[Optional[_PersistJob]]" = queue.Queue(maxsize=self.persist_queue_size)
        self._media_queue: "queue.Queue[Optional[_MediaJob]]" = queue.Queue(maxsize=self.media_queue_size)

        self._pending_lock = threading.Lock()
        self._pending_identity_tracks: Set[int] = set()

        self._thread: Optional[threading.Thread] = None
        self._persist_thread: Optional[threading.Thread] = None
        self._media_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="recognition-worker")
        self._persist_thread = threading.Thread(target=self._persist_loop, daemon=True, name="recognition-persist")
        self._media_thread = threading.Thread(target=self._media_loop, daemon=True, name="recognition-media")
        self._thread.start()
        self._persist_thread.start()
        self._media_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self._persist_queue.put_nowait(None)
        except Exception:
            pass
        try:
            self._media_queue.put_nowait(None)
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._persist_thread:
            self._persist_thread.join(timeout=2.0)
        if self._media_thread:
            self._media_thread.join(timeout=2.0)
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
            active_track_ids = {int(st.track_id) for st in tracks}
            stale_failure_keys = [tid for tid in self._face_failures if tid not in active_track_ids]
            for tid in stale_failure_keys:
                self._face_failures.pop(tid, None)
                self._last_face_attempt_ts.pop(tid, None)
                self._last_body_attempt_ts.pop(tid, None)
                self._clear_pending_track(tid)

            for st in tracks:
                # Ignore stale tracks.
                if frame_id - int(st.last_seen_frame) > 2:
                    continue

                if self._is_track_pending(st.track_id):
                    self._inc_metric("recognition_pending_identity", 1)
                    continue

                if st.identity_id is not None and not db.identity_exists(int(st.identity_id)):
                    self.track_store.assign_identity(
                        track_id=st.track_id,
                        identity_id=None,
                        modality="none",
                        score=0.0,
                        frame_id=frame_id,
                        cache_seconds=0.0,
                    )
                    st.identity_id = None
                    st.cache_until = 0.0
                    self._inc_metric("stale_identity_cleared", 1)

                # Cache gate for already-linked track IDs.
                if st.identity_id is not None and now < float(st.cache_until):
                    self._inc_metric("suppressed_by_cache", 1)
                    LOGGER.debug(
                        "Recognition suppressed by cache | track_id=%s identity_id=%s cache_for=%.2fs",
                        st.track_id,
                        st.identity_id,
                        float(st.cache_until) - now,
                    )
                    continue

                # Event-driven trigger: new track or unresolved track at rate limit.
                should_try_face = bool(st.is_new) or st.identity_id is None
                last_face_try_ts = float(self._last_face_attempt_ts.get(st.track_id, 0.0))
                face_due_by_frame = frame_id - int(st.last_face_frame) >= self.face_interval
                face_due_by_time = (now - last_face_try_ts) >= self.face_interval_seconds
                if should_try_face and face_due_by_frame and face_due_by_time:
                    t0 = time.perf_counter()
                    self.track_store.mark_face_checked(st.track_id, frame_id)
                    self._last_face_attempt_ts[st.track_id] = now
                    self._inc_metric("recognition_attempts", 1)
                    self._inc_metric("face_attempts", 1)
                    face_emb = self._extract_face_embedding(frame, st.bbox)
                    if face_emb is not None:
                        self._match_or_create_face_async(st.track_id, face_emb, frame, st.bbox, frame_id)
                        self._face_failures[st.track_id] = 0
                    else:
                        self._face_failures[st.track_id] = int(self._face_failures.get(st.track_id, 0)) + 1
                    rec_ms_total += (time.perf_counter() - t0) * 1000.0
                    continue

                # Optional body-only association path.
                should_try_body_fallback = (
                    self.body_fallback_enabled
                    and not self.disable_body_embedding
                    and st.feature is not None
                    and int(self._face_failures.get(st.track_id, 0)) >= self.body_fallback_after_face_failures
                )
                if should_try_body_fallback:
                    last_body_try_ts = float(self._last_body_attempt_ts.get(st.track_id, 0.0))
                    body_due_by_frame = frame_id - int(st.last_body_frame) >= self.body_interval
                    body_due_by_time = (now - last_body_try_ts) >= self.body_interval_seconds
                    if body_due_by_frame and body_due_by_time:
                        t0 = time.perf_counter()
                        self.track_store.mark_body_checked(st.track_id, frame_id)
                        self._last_body_attempt_ts[st.track_id] = now
                        self._inc_metric("recognition_attempts", 1)
                        self._match_body_async(st.track_id, st.feature, frame_id)
                        rec_ms_total += (time.perf_counter() - t0) * 1000.0

            self.timing_store.set(frame_id, "recognition_ms", rec_ms_total)
            self.stage_stats.add("recognition_ms", rec_ms_total)
            self.fps_counter.inc(1)

    def _persist_loop(self) -> None:
        while not self.stop_event.is_set() or not self._persist_queue.empty():
            try:
                first = self._persist_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if first is None:
                if self.stop_event.is_set() and self._persist_queue.empty():
                    break
                continue

            batch: list[_PersistJob] = [first]
            while len(batch) < self.persist_batch_size:
                try:
                    nxt = self._persist_queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    continue
                batch.append(nxt)

            for job in batch:
                self._execute_persist_job(job)

    def _media_loop(self) -> None:
        while not self.stop_event.is_set() or not self._media_queue.empty():
            try:
                first = self._media_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if first is None:
                if self.stop_event.is_set() and self._media_queue.empty():
                    break
                continue

            batch: list[_MediaJob] = [first]
            while len(batch) < self.media_batch_size:
                try:
                    nxt = self._media_queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    continue
                batch.append(nxt)

            for job in batch:
                self._execute_media_job(job)

    def _execute_persist_job(self, job: _PersistJob) -> None:
        kind = str(job.kind)
        payload = job.payload
        try:
            if kind == "update_last_seen":
                db.update_last_seen(int(payload["identity_id"]), str(payload["ts"]))
                self._inc_metric("db_writes", 1)
                return

            if kind == "update_body_ema":
                db.update_body_ema(int(payload["identity_id"]), payload["body_emb"])
                self._inc_metric("db_writes", 1)
                return

            if kind == "create_face_identity":
                track_id = int(payload["track_id"])
                try:
                    new_id = db.add_identity(payload["face_emb"], None, None, None, str(payload["ts"]))
                    self._inc_metric("db_inserts", 1)
                    self._inc_metric("db_writes", 1)
                    self.track_store.assign_identity(
                        track_id=track_id,
                        identity_id=int(new_id),
                        modality="face",
                        score=1.0,
                        frame_id=int(payload["frame_id"]),
                        cache_seconds=self.cache_seconds,
                    )
                    sample = payload.get("sample")
                    if isinstance(sample, np.ndarray) and sample.size > 0:
                        self._enqueue_media(_MediaJob(identity_id=int(new_id), sample_bgr=sample))
                finally:
                    self._clear_pending_track(track_id)
                return

            if kind == "create_body_identity":
                track_id = int(payload["track_id"])
                try:
                    new_id = db.add_identity(None, payload["body_emb"], None, None, str(payload["ts"]))
                    self._inc_metric("db_inserts", 1)
                    self._inc_metric("db_writes", 1)
                    self.track_store.assign_identity(
                        track_id=track_id,
                        identity_id=int(new_id),
                        modality="body",
                        score=1.0,
                        frame_id=int(payload["frame_id"]),
                        cache_seconds=self.cache_seconds,
                    )
                finally:
                    self._clear_pending_track(track_id)
                return

            LOGGER.debug("Unknown persist job kind=%s", kind)
        except Exception:
            if kind in {"create_face_identity", "create_body_identity"}:
                try:
                    self._clear_pending_track(int(payload.get("track_id", -1)))
                except Exception:
                    pass
            LOGGER.exception("Persist job failed | kind=%s", kind)

    def _execute_media_job(self, job: _MediaJob) -> None:
        sample = job.sample_bgr
        if sample is None or sample.size == 0:
            return

        ts_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        ms_part = int((time.time() * 1000.0) % 1000)
        sample_path = str(Path(self.faces_dir) / f"{int(job.identity_id)}_{ts_str}_{ms_part:03d}.jpg")
        ok = False
        try:
            ok = bool(cv2.imwrite(sample_path, sample))
        except Exception:
            ok = False
        if not ok:
            LOGGER.debug("Failed to write face sample | identity_id=%s path=%s", job.identity_id, sample_path)
            return

        try:
            db.update_face_sample_path(int(job.identity_id), sample_path)
            self._inc_metric("db_writes", 1)
            self._inc_metric("media_writes", 1)
        except Exception:
            LOGGER.exception("Failed to persist face sample path | identity_id=%s", job.identity_id)

    def _enqueue_persist(self, job: _PersistJob) -> bool:
        try:
            self._persist_queue.put_nowait(job)
            return True
        except queue.Full:
            self._inc_metric("persist_queue_drop", 1)
            LOGGER.debug("Persist queue full; dropping job kind=%s", job.kind)
            return False

    def _enqueue_media(self, job: _MediaJob) -> bool:
        try:
            self._media_queue.put_nowait(job)
            return True
        except queue.Full:
            self._inc_metric("media_queue_drop", 1)
            LOGGER.debug("Media queue full; dropping sample for identity_id=%s", job.identity_id)
            return False

    def _mark_track_pending(self, track_id: int) -> bool:
        tid = int(track_id)
        with self._pending_lock:
            if tid in self._pending_identity_tracks:
                return False
            self._pending_identity_tracks.add(tid)
        return True

    def _clear_pending_track(self, track_id: int) -> None:
        tid = int(track_id)
        with self._pending_lock:
            self._pending_identity_tracks.discard(tid)

    def _is_track_pending(self, track_id: int) -> bool:
        tid = int(track_id)
        with self._pending_lock:
            return tid in self._pending_identity_tracks

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
        roi_w = max(1, x2 - x1)
        roi = frame_bgr[y1:y1 + top_h, x1:x2]
        if roi_w < self.min_face_pixels or top_h < self.min_face_pixels:
            LOGGER.debug(
                "Face attempt skipped: roi_too_small | roi=%sx%s min=%s bbox=%s",
                roi_w,
                top_h,
                self.min_face_pixels,
                person_bbox,
            )
            return None
        if roi.size == 0:
            LOGGER.debug("Face attempt skipped: empty_roi | bbox=%s", person_bbox)
            return None

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        faces = self.face_detector.detect(roi_rgb)
        if not faces:
            LOGGER.debug("Face attempt skipped: no_face_detected | bbox=%s", person_bbox)
            return None

        self._inc_metric("face_detections", 1)
        fx1, fy1, fx2, fy2, _score = max(faces, key=lambda v: v[4])
        ax1 = max(0, x1 + int(fx1))
        ay1 = max(0, y1 + int(fy1))
        ax2 = min(w, x1 + int(fx2))
        ay2 = min(h, y1 + int(fy2))
        if ax2 <= ax1 or ay2 <= ay1:
            LOGGER.debug("Face attempt skipped: invalid_face_box | face_box=%s", (ax1, ay1, ax2, ay2))
            return None

        face_crop = frame_bgr[ay1:ay2, ax1:ax2]
        if face_crop.size == 0:
            LOGGER.debug("Face attempt skipped: empty_face_crop | face_box=%s", (ax1, ay1, ax2, ay2))
            return None
        if (ax2 - ax1) < self.min_face_pixels or (ay2 - ay1) < self.min_face_pixels:
            LOGGER.debug(
                "Face attempt skipped: face_crop_too_small | face=%sx%s min=%s",
                ax2 - ax1,
                ay2 - ay1,
                self.min_face_pixels,
            )
            return None

        emb = self.face_embedder.embed_from_bgr(face_crop)
        if emb is not None:
            self._inc_metric("embedding_success", 1)
        return emb

    def _match_or_create_face_async(
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
            self.track_store.assign_identity(
                track_id=track_id,
                identity_id=int(best_id),
                modality="face",
                score=float(best_score),
                frame_id=frame_id,
                cache_seconds=self.cache_seconds,
            )
            self._inc_metric("successful_matches", 1)
            self._enqueue_persist(
                _PersistJob(
                    kind="update_last_seen",
                    payload={"identity_id": int(best_id), "ts": now_ts},
                )
            )
            return

        if not self._mark_track_pending(track_id):
            return

        x1, y1, x2, y2 = [int(v) for v in person_bbox]
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        sample = frame_bgr[y1:y2, x1:x2]
        sample_copy = sample.copy() if sample.size > 0 else np.zeros((0, 0, 3), dtype=np.uint8)

        queued = self._enqueue_persist(
            _PersistJob(
                kind="create_face_identity",
                payload={
                    "track_id": int(track_id),
                    "face_emb": face_emb,
                    "frame_id": int(frame_id),
                    "sample": sample_copy,
                    "ts": now_ts,
                },
            )
        )
        if not queued:
            self._clear_pending_track(track_id)

    def _match_body_async(self, track_id: int, body_emb: np.ndarray, frame_id: int) -> None:
        now_ts = timestamp_iso()
        best_id, best_score = db.find_best_body(body_emb)
        if best_id is not None and best_score >= self.body_threshold:
            self.track_store.assign_identity(
                track_id=track_id,
                identity_id=int(best_id),
                modality="body",
                score=float(best_score),
                frame_id=frame_id,
                cache_seconds=self.cache_seconds,
            )
            self._inc_metric("successful_matches", 1)
            self._enqueue_persist(
                _PersistJob(
                    kind="update_last_seen",
                    payload={"identity_id": int(best_id), "ts": now_ts},
                )
            )
            self._enqueue_persist(
                _PersistJob(
                    kind="update_body_ema",
                    payload={"identity_id": int(best_id), "body_emb": body_emb},
                )
            )
            return

        if self.persist_body_only:
            if not self._mark_track_pending(track_id):
                return
            queued = self._enqueue_persist(
                _PersistJob(
                    kind="create_body_identity",
                    payload={
                        "track_id": int(track_id),
                        "body_emb": body_emb,
                        "frame_id": int(frame_id),
                        "ts": now_ts,
                    },
                )
            )
            if not queued:
                self._clear_pending_track(track_id)

    def _inc_metric(self, name: str, amount: int = 1) -> None:
        if self.metrics is not None:
            self.metrics.inc(name, amount)
