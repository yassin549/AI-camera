"""Shared utility helpers for math, serialization, synchronization, and timing."""

from __future__ import annotations

import io
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with epsilon to avoid division by zero."""
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10)
    if denom <= 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector and return float32 output."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-10:
        return v.astype(np.float32)
    return (v / norm).astype(np.float32)


def save_np_to_blob(np_arr: np.ndarray) -> bytes:
    """Serialize a numpy array to bytes via np.save and BytesIO."""
    bio = io.BytesIO()
    np.save(bio, np.asarray(np_arr, dtype=np.float32), allow_pickle=False)
    return bio.getvalue()


def load_np_from_blob(blob: bytes) -> np.ndarray:
    """Deserialize bytes back into a float32 numpy array."""
    bio = io.BytesIO(blob)
    bio.seek(0)
    return np.load(bio, allow_pickle=False).astype(np.float32)


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two boxes in xyxy format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def timestamp_iso(ts: Optional[float] = None) -> str:
    """UTC timestamp in ISO-8601 format."""
    if ts is None:
        return datetime.now(tz=timezone.utc).isoformat()
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def ensure_dir(path: str) -> str:
    """Create directory if needed and return absolute path string."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())


def ensure_rtsp_tcp(url: str) -> str:
    """Append TCP transport hints to RTSP URL when not present."""
    text = str(url or "").strip()
    lower = text.lower()
    if not lower.startswith("rtsp://"):
        return text
    if "tcp" in lower:
        return text
    if "?" in text:
        return f"{text}&tcp"
    return f"{text}?tcp"


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    ts: float
    frame: np.ndarray


class LatestFrameStore:
    """Single-slot latest-frame container for producer/consumer threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest: Optional[FramePacket] = None

    def publish(self, frame_id: int, frame: np.ndarray, ts: Optional[float] = None) -> None:
        if ts is None:
            ts = time.time()
        packet = FramePacket(frame_id=frame_id, ts=float(ts), frame=frame)
        with self._cond:
            self._latest = packet
            self._cond.notify_all()

    def get_latest(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest

    def wait_for_new(self, last_frame_id: int, timeout: float = 0.05) -> Optional[FramePacket]:
        end = time.time() + max(0.0, timeout)
        with self._cond:
            while True:
                if self._latest is not None and self._latest.frame_id > last_frame_id:
                    return self._latest
                remaining = end - time.time()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)


@dataclass
class TrackState:
    track_id: int
    bbox: Tuple[int, int, int, int]
    score: float
    last_seen_frame: int
    created_frame: int
    identity_id: Optional[int] = None
    modality: str = "none"
    identity_score: float = 0.0
    cache_until: float = 0.0
    last_face_frame: int = -1_000_000
    last_body_frame: int = -1_000_000
    last_embedding_frame: int = -1_000_000
    feature: Optional[np.ndarray] = None
    is_new: bool = True


class SharedTrackStore:
    """Thread-safe shared track state table used by tracking/recognition/render."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tracks: Dict[int, TrackState] = {}

    def update_from_tracker(self, tracks: List[Any], frame_id: int, max_tracks: int = 20) -> None:
        with self._lock:
            seen_ids = set()
            for tr in tracks[: max(1, int(max_tracks))]:
                tid = int(tr.track_id)
                seen_ids.add(tid)
                if tid in self._tracks:
                    state = self._tracks[tid]
                    state.bbox = tuple(map(int, tr.bbox))
                    state.score = float(tr.score)
                    state.last_seen_frame = int(frame_id)
                    state.feature = tr.feature
                    state.is_new = False
                else:
                    self._tracks[tid] = TrackState(
                        track_id=tid,
                        bbox=tuple(map(int, tr.bbox)),
                        score=float(tr.score),
                        last_seen_frame=int(frame_id),
                        created_frame=int(frame_id),
                        feature=tr.feature,
                        is_new=True,
                    )

            stale_cutoff = int(frame_id) - 90
            stale = [tid for tid, st in self._tracks.items() if st.last_seen_frame < stale_cutoff]
            for tid in stale:
                self._tracks.pop(tid, None)

    def snapshot(self) -> List[TrackState]:
        with self._lock:
            return [TrackState(**vars(st)) for st in self._tracks.values()]

    def mark_face_checked(self, track_id: int, frame_id: int) -> None:
        with self._lock:
            st = self._tracks.get(int(track_id))
            if st is not None:
                st.last_face_frame = int(frame_id)

    def mark_body_checked(self, track_id: int, frame_id: int) -> None:
        with self._lock:
            st = self._tracks.get(int(track_id))
            if st is not None:
                st.last_body_frame = int(frame_id)

    def assign_identity(
        self,
        track_id: int,
        identity_id: Optional[int],
        modality: str,
        score: float,
        frame_id: int,
        cache_seconds: float,
    ) -> None:
        with self._lock:
            st = self._tracks.get(int(track_id))
            if st is None:
                return
            st.identity_id = None if identity_id is None else int(identity_id)
            st.modality = str(modality)
            st.identity_score = float(score)
            st.cache_until = time.time() + max(0.0, float(cache_seconds))
            st.last_embedding_frame = int(frame_id)

    def set_cache_until(self, track_id: int, cache_until_ts: float) -> None:
        with self._lock:
            st = self._tracks.get(int(track_id))
            if st is not None:
                st.cache_until = float(cache_until_ts)

    def to_api_payload(self) -> Dict[int, Dict[str, Any]]:
        with self._lock:
            payload: Dict[int, Dict[str, Any]] = {}
            for tid, st in self._tracks.items():
                payload[int(tid)] = {
                    "track_id": int(st.track_id),
                    "bbox": [int(v) for v in st.bbox],
                    "score": float(st.score),
                    "identity_id": st.identity_id,
                    "modality": st.modality,
                    "last_score": float(st.identity_score),
                    "last_seen_frame": int(st.last_seen_frame),
                    "timestamp": timestamp_iso(),
                }
            return payload


class FrameTimingStore:
    """Per-frame stage timing store for debug CSV and benchmark summaries."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[int, Dict[str, float]] = defaultdict(dict)

    def set(self, frame_id: int, stage: str, ms: float, accumulate: bool = False) -> None:
        with self._lock:
            row = self._data[int(frame_id)]
            if accumulate and stage in row:
                row[stage] += float(ms)
            else:
                row[stage] = float(ms)

    def get(self, frame_id: int) -> Dict[str, float]:
        with self._lock:
            return dict(self._data.get(int(frame_id), {}))

    def cleanup_older_than(self, frame_id: int) -> None:
        with self._lock:
            keys = [k for k in self._data.keys() if k < int(frame_id)]
            for k in keys:
                self._data.pop(k, None)


class StageStats:
    """Rolling statistics per stage with avg and p95 reporting."""

    def __init__(self, window: int = 2048) -> None:
        self._lock = threading.Lock()
        self._values: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._counts: Dict[str, int] = defaultdict(int)

    def add(self, stage: str, value: float) -> None:
        with self._lock:
            self._values[stage].append(float(value))
            self._counts[stage] += 1

    def summary(self, stage: str) -> Tuple[int, float, float]:
        with self._lock:
            vals = list(self._values.get(stage, []))
            count = int(self._counts.get(stage, 0))
        if not vals:
            return count, 0.0, 0.0
        arr = np.asarray(vals, dtype=np.float32)
        return count, float(arr.mean()), float(np.percentile(arr, 95))


class CounterFPS:
    """Simple event counter used for stage FPS reporting."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self._value += int(n)

    def pop(self) -> int:
        with self._lock:
            val = self._value
            self._value = 0
            return val

    def value(self) -> int:
        with self._lock:
            return self._value
