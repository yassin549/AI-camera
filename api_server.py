"""FastAPI server for AIcam identities + media + realtime transports."""

from __future__ import annotations

import asyncio
import logging
import queue
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import db
from api.identities import router as identities_router
from api.janus_proxy import router as janus_proxy_router
from api.media import mount_static_media, router as media_router
from api.realtime import AIORTC_AVAILABLE, router as realtime_router
from config import (
    APP_NAME,
    APP_VERSION,
    CORS_ORIGINS,
    CORS_ORIGIN_REGEX,
    ENABLE_FRAME_STREAMING,
    HEALTH_FRAME_STALE_SEC,
    MJPEG_JPEG_QUALITY,
    STATIC_PATH,
    is_api_key_valid,
)
from utils import RuntimeMetrics, SharedTrackStore

LOGGER = logging.getLogger("aicam.api")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_from_unix(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    try:
        value = float(ts)
    except Exception:
        return None
    if value <= 0:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _to_media_url(raw_path: Optional[str], media_root: Path) -> Optional[str]:
    if not raw_path:
        return None

    sample = Path(str(raw_path))
    candidate_rel_paths: List[Path] = []
    if sample.is_absolute():
        try:
            candidate_rel_paths.append(sample.relative_to(media_root))
        except Exception:
            candidate_rel_paths = []
    else:
        normalized = Path(str(sample).replace("\\", "/").lstrip("/"))
        candidate_rel_paths.append(normalized)

    if sample.name:
        for subdir in ("faces", "body", "bodies", "samples/faces", "samples/bodies", "data/faces", "data/bodies"):
            candidate_rel_paths.append(Path(subdir) / sample.name)

    seen: Set[str] = set()
    for rel in candidate_rel_paths:
        rel_key = rel.as_posix()
        if rel_key in seen:
            continue
        seen.add(rel_key)
        candidate = media_root / rel
        try:
            if candidate.exists() and candidate.is_file():
                return f"/media/{rel_key}"
        except Exception:
            continue
    return None


def _discover_face_thumb(identity_id: int, media_root: Path) -> Optional[str]:
    search_dirs = [
        media_root / "faces",
        media_root / "data" / "faces",
        media_root / "samples" / "faces",
    ]
    # Primary pattern: <id>_*.jpg (new convention)
    # Legacy patterns: face_*_t*.jpg and yassin_* etc. are caught via DB path
    patterns = [
        f"{identity_id}_*.jpg", f"{identity_id}_*.jpeg", f"{identity_id}_*.png",
        f"face_*_t{identity_id}.jpg", f"face_*_t{identity_id}.jpeg",
    ]
    for directory in search_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if not matches:
                continue
            candidate = matches[0]
            try:
                rel = candidate.relative_to(media_root)
            except Exception:
                continue
            return f"/media/{rel.as_posix()}"
    return None


class MetadataHub:
    """Thread-safe latest-snapshot publisher for low-latency websocket fanout."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._version = 0
        self._latest: Optional[Dict[str, Any]] = None

    def publish(self, payload: Dict[str, Any]) -> None:
        with self._cond:
            self._version += 1
            self._latest = payload
            self._cond.notify_all()

    def snapshot(self) -> Tuple[int, Optional[Dict[str, Any]]]:
        with self._lock:
            return self._version, dict(self._latest) if self._latest is not None else None

    def wait_for_update(self, last_version: int, timeout: float) -> Tuple[int, Optional[Dict[str, Any]], bool]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cond:
            while self._version <= int(last_version):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    payload = dict(self._latest) if self._latest is not None else None
                    return self._version, payload, False
                self._cond.wait(timeout=remaining)
            payload = dict(self._latest) if self._latest is not None else None
            return self._version, payload, True


class ApiRuntimeState:
    """Shared state consumed by API routes and updated from pipeline threads."""

    def __init__(
        self,
        frame_store: Any,
        db_path: str,
        media_root: Optional[str] = None,
        metrics: Optional[RuntimeMetrics] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        track_store: Optional[SharedTrackStore] = None,
    ) -> None:
        self.frame_store = frame_store
        self.db_path = str(db_path)
        self.media_root = Path(media_root or STATIC_PATH)
        self.enable_frame_streaming = bool(ENABLE_FRAME_STREAMING)
        self.metrics = metrics
        self.track_store = track_store

        self._tracks_lock = threading.RLock()
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._capabilities_lock = threading.Lock()
        self._capabilities: Dict[str, Any] = dict(capabilities or {})

        self._thumb_lock = threading.Lock()
        self._thumb_cache: Dict[int, str] = {}
        self._muted_cache: Dict[int, bool] = {}
        self._thumb_refresh_all = True
        self._thumb_refresh_ids: Set[int] = set()

        self._peer_lock = threading.Lock()
        self._peer_connections: Set[Any] = set()

        self._client_lock = threading.Lock()
        self._mjpeg_clients = 0
        self._ws_clients = 0

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = -1
        self._latest_frame_ts = 0.0
        self._jpeg_quality = self._coerce_jpeg_quality(MJPEG_JPEG_QUALITY)
        self._latest_jpeg_frame_id = -1
        self._latest_jpeg_quality = self._jpeg_quality
        self._latest_jpeg = b""
        self._placeholder_jpeg = self._build_placeholder_jpeg()
        self._latest_jpeg = self._placeholder_jpeg
        self._frame_queue: "queue.Queue[Optional[Tuple[np.ndarray, int, float]]]" = queue.Queue(maxsize=2)
        self._frame_encoder_stop = threading.Event()
        self._frame_encoder_thread = threading.Thread(target=self._frame_encoder_loop, daemon=True, name="api-jpeg-encoder")
        self._frame_encoder_thread.start()
        self.metadata_hub = MetadataHub()

        self._publisher_queue: "queue.Queue[Optional[Tuple[Dict[int, Dict[str, Any]], int, Tuple[int, int, int], float]]]" = queue.Queue(
            maxsize=8
        )
        self._publisher_stop = threading.Event()
        self._close_lock = threading.Lock()
        self._closed = False
        self._publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True, name="api-metadata-publisher")
        self._publisher_thread.start()

        self._db_listener_registered = False
        try:
            db.register_change_listener(self._on_identity_store_change)
            self._db_listener_registered = True
        except Exception:
            LOGGER.debug("Could not register DB change listener", exc_info=True)

    def set_capability(self, name: str, value: Any) -> None:
        with self._capabilities_lock:
            self._capabilities[str(name)] = value

    def get_capabilities(self) -> Dict[str, Any]:
        with self._capabilities_lock:
            return dict(self._capabilities)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._publisher_stop.set()
        try:
            self._publisher_queue.put_nowait(None)
        except Exception:
            pass
        self._frame_encoder_stop.set()
        try:
            self._frame_queue.put_nowait(None)
        except Exception:
            pass
        if self._frame_encoder_thread.is_alive():
            self._frame_encoder_thread.join(timeout=2.0)
        if self._publisher_thread.is_alive():
            self._publisher_thread.join(timeout=2.0)
        if self._db_listener_registered:
            try:
                db.unregister_change_listener(self._on_identity_store_change)
            except Exception:
                LOGGER.debug("Could not unregister DB change listener", exc_info=True)
            self._db_listener_registered = False

    def register_peer_connection(self, peer_connection: Any) -> None:
        with self._peer_lock:
            self._peer_connections.add(peer_connection)

    def unregister_peer_connection(self, peer_connection: Any) -> None:
        with self._peer_lock:
            self._peer_connections.discard(peer_connection)

    async def close_peer_connections(self) -> None:
        with self._peer_lock:
            peers = list(self._peer_connections)
            self._peer_connections.clear()
        for peer in peers:
            try:
                await peer.close()
            except Exception:
                LOGGER.debug("Failed to close RTCPeerConnection", exc_info=True)

    def register_mjpeg_client(self) -> None:
        with self._client_lock:
            self._mjpeg_clients += 1

    def unregister_mjpeg_client(self) -> None:
        with self._client_lock:
            self._mjpeg_clients = max(0, self._mjpeg_clients - 1)

    def register_ws_client(self) -> None:
        with self._client_lock:
            self._ws_clients += 1

    def unregister_ws_client(self) -> None:
        with self._client_lock:
            self._ws_clients = max(0, self._ws_clients - 1)

    def get_client_counts(self) -> Tuple[int, int]:
        with self._client_lock:
            return self._mjpeg_clients, self._ws_clients

    def is_capture_running(self) -> bool:
        frame_ts = self._latest_frame_ts
        if frame_ts <= 0:
            packet = self.frame_store.get_latest()
            if packet is None:
                return False
            frame_ts = float(packet.ts)
        if frame_ts <= 0:
            return False
        age = time.time() - frame_ts
        return age <= HEALTH_FRAME_STALE_SEC

    def _build_placeholder_jpeg(self) -> bytes:
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "No video frame yet",
            (160, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(MJPEG_JPEG_QUALITY)])
        if not ok:
            return b""
        return encoded.tobytes()

    @staticmethod
    def _coerce_jpeg_quality(quality: Optional[int]) -> int:
        if quality is None:
            return int(MJPEG_JPEG_QUALITY)
        try:
            parsed = int(quality)
        except Exception:
            return int(MJPEG_JPEG_QUALITY)
        return max(1, min(100, parsed))

    def _encode_jpeg(self, frame: Optional[np.ndarray], quality: int) -> bytes:
        if frame is None or frame.size == 0:
            return self._placeholder_jpeg
        try:
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
            )
            if not ok:
                return self._placeholder_jpeg
            return encoded.tobytes()
        except Exception:
            LOGGER.debug("Failed to encode latest MJPEG frame", exc_info=True)
            return self._placeholder_jpeg

    def _enqueue_frame_for_encode(self, frame: np.ndarray, frame_id: int, frame_ts: Optional[float]) -> None:
        capture_ts = float(frame_ts) if frame_ts is not None else time.time()
        packet = (frame, int(frame_id), capture_ts)
        try:
            self._frame_queue.put_nowait(packet)
            return
        except queue.Full:
            pass
        try:
            _ = self._frame_queue.get_nowait()
        except Exception:
            pass
        try:
            self._frame_queue.put_nowait(packet)
        except Exception:
            LOGGER.debug("Frame encoder queue saturated; dropping frame_id=%s", frame_id)

    def _frame_encoder_loop(self) -> None:
        while not self._frame_encoder_stop.is_set():
            try:
                item = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                break

            latest = item
            while True:
                try:
                    nxt = self._frame_queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._frame_encoder_stop.set()
                    break
                latest = nxt

            frame, frame_id, frame_ts = latest
            with self._client_lock:
                video_clients = int(self._mjpeg_clients + self._ws_clients)
            if video_clients <= 0:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_frame_id = int(frame_id)
                    self._latest_frame_ts = float(frame_ts) if frame_ts > 0 else time.time()
                continue
            with self._frame_lock:
                quality = int(self._jpeg_quality)
            jpeg = self._encode_jpeg(frame, quality)
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_frame_id = int(frame_id)
                self._latest_frame_ts = float(frame_ts) if frame_ts > 0 else time.time()
                self._latest_jpeg = jpeg
                self._latest_jpeg_frame_id = int(frame_id)
                self._latest_jpeg_quality = int(quality)
            if self.metrics is not None:
                self.metrics.set_gauge("video_jpeg_quality", float(quality))
                self.metrics.set_gauge("video_jpeg_bytes", float(len(jpeg)))

    def set_stream_quality(self, quality: Optional[int]) -> int:
        next_quality = self._coerce_jpeg_quality(quality)
        frame: Optional[np.ndarray] = None
        frame_id = -1
        frame_ts = 0.0
        with self._frame_lock:
            if next_quality == int(self._jpeg_quality):
                return int(self._jpeg_quality)
            self._jpeg_quality = int(next_quality)
            frame = self._latest_frame
            frame_id = int(self._latest_frame_id)
            frame_ts = float(self._latest_frame_ts)
        if frame is not None and frame_id >= 0:
            self._enqueue_frame_for_encode(frame, frame_id, frame_ts)
        return int(next_quality)

    def get_stream_quality(self) -> int:
        with self._frame_lock:
            return int(self._jpeg_quality)

    def update_frame(self, frame: np.ndarray, frame_id: int, frame_ts: Optional[float] = None) -> None:
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_frame_id = int(frame_id)
            self._latest_frame_ts = float(frame_ts) if frame_ts is not None else time.time()
        self._enqueue_frame_for_encode(frame, int(frame_id), frame_ts)

    def get_latest_jpeg_packet(self, quality: Optional[int] = None) -> Tuple[int, bytes]:
        if not self.enable_frame_streaming:
            return -1, self._placeholder_jpeg
        if quality is not None:
            self.set_stream_quality(quality)

        with self._frame_lock:
            frame_id = int(self._latest_jpeg_frame_id)
            jpeg = self._latest_jpeg or self._placeholder_jpeg
        if frame_id < 0:
            return -1, self._placeholder_jpeg
        return frame_id, jpeg

    def get_latest_jpeg(self) -> bytes:
        _frame_id, jpeg = self.get_latest_jpeg_packet()
        return jpeg

    def get_tracks(self) -> List[Dict[str, Any]]:
        with self._tracks_lock:
            return [dict(v) for _, v in sorted(self._tracks.items())]

    def remap_identity(self, source_identity_id: int, target_identity_id: int) -> int:
        source_id = int(source_identity_id)
        target_id = int(target_identity_id)
        if source_id == target_id:
            return 0
        updated = 0
        if self.track_store is not None:
            try:
                updated += int(self.track_store.remap_identity(source_id, target_id))
            except Exception:
                LOGGER.debug("Failed to remap identity in shared track store", exc_info=True)
        with self._tracks_lock:
            for state in self._tracks.values():
                if state.get("identity_id") is None:
                    continue
                try:
                    if int(state.get("identity_id")) != source_id:
                        continue
                except Exception:
                    continue
                state["identity_id"] = target_id
                updated += 1
        return updated

    def clear_identity(self, identity_id: int) -> int:
        ident_id = int(identity_id)
        cleared = 0
        if self.track_store is not None:
            try:
                cleared += int(self.track_store.clear_identity(ident_id))
            except Exception:
                LOGGER.debug("Failed to clear identity in shared track store", exc_info=True)
        with self._tracks_lock:
            for state in self._tracks.values():
                if state.get("identity_id") is None:
                    continue
                try:
                    if int(state.get("identity_id")) != ident_id:
                        continue
                except Exception:
                    continue
                state["identity_id"] = None
                state["modality"] = "none"
                state["last_score"] = 0.0
                cleared += 1
        return cleared

    def publish_tracks(
        self,
        tracks_payload: Dict[int, Dict[str, Any]],
        frame_id: int,
        frame_shape: Tuple[int, int, int],
        frame: Optional[np.ndarray] = None,
        frame_ts: Optional[float] = None,
    ) -> None:
        if self.enable_frame_streaming:
            if frame is not None:
                self.update_frame(frame=frame, frame_id=int(frame_id), frame_ts=frame_ts)
            else:
                packet = self.frame_store.get_latest()
                if packet is not None:
                    self.update_frame(packet.frame, int(packet.frame_id), frame_ts=float(packet.ts))
        try:
            self._enqueue_metadata_publish(tracks_payload, int(frame_id), frame_shape, frame_ts)
        except Exception:
            LOGGER.exception("publish_tracks failed | frame_id=%s", frame_id)

    def _enqueue_metadata_publish(
        self,
        tracks_payload: Dict[int, Dict[str, Any]],
        frame_id: int,
        frame_shape: Tuple[int, int, int],
        frame_ts: Optional[float],
    ) -> None:
        payload_copy = {int(k): dict(v) for k, v in tracks_payload.items()}
        capture_ts = 0.0
        if frame_ts is not None:
            try:
                capture_ts = max(0.0, float(frame_ts))
            except Exception:
                capture_ts = 0.0
        packet = (payload_copy, int(frame_id), frame_shape, capture_ts)
        try:
            self._publisher_queue.put_nowait(packet)
            return
        except queue.Full:
            pass
        # Drop stale metadata packet to keep low-latency semantics.
        try:
            _ = self._publisher_queue.get_nowait()
        except Exception:
            pass
        try:
            self._publisher_queue.put_nowait(packet)
        except Exception:
            LOGGER.debug("Metadata publisher queue is saturated; dropping frame_id=%s", frame_id)

    def _on_identity_store_change(self, _event_name: str, identity_id: int) -> None:
        self.request_thumb_refresh(int(identity_id))

    def request_thumb_refresh(self, identity_id: Optional[int]) -> None:
        with self._thumb_lock:
            if identity_id is None:
                self._thumb_refresh_all = True
                self._thumb_refresh_ids.clear()
            elif not self._thumb_refresh_all:
                self._thumb_refresh_ids.add(int(identity_id))

    def _consume_thumb_refresh_request(self) -> Tuple[bool, List[int]]:
        with self._thumb_lock:
            do_full = bool(self._thumb_refresh_all)
            ids = list(self._thumb_refresh_ids)
            self._thumb_refresh_all = False
            self._thumb_refresh_ids.clear()
        return do_full, ids

    def _refresh_thumb_cache_all(self) -> None:
        next_cache: Dict[int, str] = {}
        next_muted: Dict[int, bool] = {}
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(identities)").fetchall()}
            if "is_muted" in columns:
                rows = conn.execute(
                    """
                    SELECT id, face_sample_path, body_sample_path, is_muted
                    FROM identities
                    ORDER BY id ASC
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, face_sample_path, body_sample_path
                    FROM identities
                    ORDER BY id ASC
                    """
                ).fetchall()
            conn.close()
            for row in rows:
                ident_id = int(row["id"])
                thumb = self._resolve_thumb_for_row(ident_id, row["face_sample_path"], row["body_sample_path"])
                if thumb:
                    next_cache[ident_id] = thumb
                next_muted[ident_id] = bool(row["is_muted"]) if "is_muted" in row.keys() else False
        except Exception:
            LOGGER.debug("Thumb cache full refresh failed", exc_info=True)
            return
        with self._thumb_lock:
            self._thumb_cache = next_cache
            self._muted_cache = next_muted

    def _refresh_thumb_cache_for_identity(self, identity_id: int) -> None:
        ident_id = int(identity_id)
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            columns = {str(item[1]) for item in conn.execute("PRAGMA table_info(identities)").fetchall()}
            if "is_muted" in columns:
                row = conn.execute(
                    """
                    SELECT face_sample_path, body_sample_path, is_muted
                    FROM identities
                    WHERE id=?
                    LIMIT 1
                    """,
                    (ident_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT face_sample_path, body_sample_path
                    FROM identities
                    WHERE id=?
                    LIMIT 1
                    """,
                    (ident_id,),
                ).fetchone()
            conn.close()
        except Exception:
            LOGGER.debug("Thumb cache identity refresh failed | id=%s", ident_id, exc_info=True)
            return
        if row is None:
            with self._thumb_lock:
                self._thumb_cache.pop(ident_id, None)
                self._muted_cache.pop(ident_id, None)
            return
        thumb = self._resolve_thumb_for_row(ident_id, row["face_sample_path"], row["body_sample_path"])
        muted = bool(row["is_muted"]) if "is_muted" in row.keys() else False
        with self._thumb_lock:
            if thumb:
                self._thumb_cache[ident_id] = thumb
            else:
                self._thumb_cache.pop(ident_id, None)
            self._muted_cache[ident_id] = muted

    def _resolve_thumb_for_row(
        self,
        identity_id: int,
        face_sample_path: Optional[str],
        body_sample_path: Optional[str],
    ) -> Optional[str]:
        thumb = _to_media_url(face_sample_path, self.media_root) or _to_media_url(body_sample_path, self.media_root)
        if not thumb:
            thumb = _discover_face_thumb(int(identity_id), self.media_root)
        return thumb

    def _publisher_loop(self) -> None:
        self.request_thumb_refresh(None)
        while not self._publisher_stop.is_set():
            do_full, ids = self._consume_thumb_refresh_request()
            if do_full:
                self._refresh_thumb_cache_all()
            elif ids:
                for ident_id in ids:
                    self._refresh_thumb_cache_for_identity(ident_id)
            try:
                item = self._publisher_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                break

            latest = item
            while True:
                try:
                    nxt = self._publisher_queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._publisher_stop.set()
                    break
                latest = nxt
            tracks_payload, frame_id, frame_shape, frame_ts = latest
            with self._tracks_lock:
                self._tracks = {int(k): dict(v) for k, v in tracks_payload.items()}
            try:
                metadata = self._build_metadata_payload(
                    frame_id=frame_id,
                    frame_shape=frame_shape,
                    tracks=tracks_payload,
                    frame_ts=frame_ts,
                )
                self.metadata_hub.publish(metadata)
            except Exception:
                LOGGER.exception("metadata publish failed | frame_id=%s", frame_id)

    def _build_metadata_payload(
        self,
        frame_id: int,
        frame_shape: Tuple[int, int, int],
        tracks: Dict[int, Dict[str, Any]],
        frame_ts: float,
    ) -> Dict[str, Any]:
        source_height = int(frame_shape[0]) if frame_shape else 0
        source_width = int(frame_shape[1]) if len(frame_shape) > 1 else 0
        capture_ts = max(0.0, float(frame_ts or 0.0))
        metadata_lag_ms = max(0.0, (time.time() - capture_ts) * 1000.0) if capture_ts > 0 else None
        if self.metrics is not None and metadata_lag_ms is not None:
            self.metrics.set_gauge("metadata_lag_ms", float(metadata_lag_ms))
        with self._thumb_lock:
            thumb_cache = dict(self._thumb_cache)
            muted_cache = dict(self._muted_cache)
        out_tracks: List[Dict[str, Any]] = []

        for _, track in sorted(tracks.items(), key=lambda item: int(item[0])):
            track_id = int(track.get("track_id", -1))
            identity_id = track.get("identity_id", None)
            modality = str(track.get("modality", "none"))
            score = float(track.get("last_score", 0.0))
            confidence = float(track.get("score", score))
            age_ratio = float(track.get("age_ratio", 0.0))
            age_frames = int(track.get("age_frames", 0))

            bbox_raw = track.get("bbox", [0, 0, 0, 0])
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox_raw]
            except Exception:
                x1, y1, x2, y2 = (0, 0, 0, 0)

            if identity_id is None:
                label = f"Track {track_id}"
                thumb = None
                muted = False
            else:
                ident_int = int(identity_id)
                label = f"ID:{ident_int} ({score:.2f})"
                thumb = thumb_cache.get(ident_int)
                muted = bool(muted_cache.get(ident_int, False))

            out_tracks.append(
                {
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "identity_id": None if identity_id is None else int(identity_id),
                    "label": label,
                    "modality": modality,
                    "confidence": confidence,
                    "age_ratio": age_ratio,
                    "age_frames": age_frames,
                    "thumb": thumb,
                    "muted": muted,
                }
            )

        return {
            "frame_id": int(frame_id),
            "timestamp": _utc_now_iso(),
            "capture_ts": _utc_from_unix(capture_ts),
            "capture_ts_unix": capture_ts if capture_ts > 0 else None,
            "metadata_lag_ms": metadata_lag_ms,
            "source_width": source_width,
            "source_height": source_height,
            "tracks": out_tracks,
        }


def create_app(runtime: ApiRuntimeState) -> FastAPI:
    app = FastAPI(title="AIcam API", version=APP_VERSION)
    app.state.runtime = runtime

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_origin_regex=CORS_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(janus_proxy_router, tags=["janus"])
    app.include_router(identities_router, prefix="/api/identities", tags=["identities"])
    app.include_router(media_router, tags=["media"])
    app.include_router(realtime_router, tags=["realtime"])
    mount_static_media(app, str(runtime.media_root))

    @app.middleware("http")
    async def _media_cache_headers(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/media/"):
            response.headers.setdefault("Cache-Control", "public, max-age=3600")
        return response

    @app.get("/")
    def root() -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": APP_NAME,
            "version": APP_VERSION,
            "status": "ok",
            "frame_streaming_enabled": runtime.enable_frame_streaming,
            "routes": [
                "/api/identities",
                "/api/identities/{id}",
                "/api/realtime/ws",
                "/webrtc/offer",
                "/janus",
            ],
        }
        if not AIORTC_AVAILABLE:
            payload["webrtc"] = "disabled (aiortc unavailable); use Janus WebRTC gateway"
        return payload

    @app.get("/api/health")
    def api_health() -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": "ok",
            "time": _utc_now_iso(),
            "capture_running": runtime.is_capture_running(),
        }
        payload.update(runtime.get_capabilities())
        return payload

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return api_health()

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        running = runtime.is_capture_running()
        if not running:
            raise HTTPException(status_code=503, detail="capture_not_running")
        return {"ok": True, "capture": "running"}

    @app.get("/api/realtime/latest")
    def api_realtime_latest() -> Dict[str, Any]:
        _, payload = runtime.metadata_hub.snapshot()
        if payload is None:
            return {"timestamp": _utc_now_iso(), "tracks": []}
        return payload

    @app.get("/api/tracks")
    def api_tracks() -> List[Dict[str, Any]]:
        return runtime.get_tracks()

    @app.get("/tracks")
    def tracks_legacy() -> List[Dict[str, Any]]:
        return runtime.get_tracks()

    @app.get("/api/debug/status")
    def debug_status() -> Dict[str, Any]:
        media_root_exists = runtime.media_root.exists()
        mjpeg_clients, ws_clients = runtime.get_client_counts()
        try:
            conn = sqlite3.connect(runtime.db_path)
            row = conn.execute("SELECT COUNT(1) FROM identities").fetchone()
            conn.close()
            identities_count = int(row[0]) if row else 0
        except Exception:
            identities_count = 0
        return {
            "api_up": True,
            "media_root_exists": bool(media_root_exists),
            "mjpeg_clients": mjpeg_clients,
            "ws_clients": ws_clients,
            "identities_count": identities_count,
            "frame_streaming_enabled": runtime.enable_frame_streaming,
            **runtime.get_capabilities(),
        }

    @app.get("/metrics")
    def metrics() -> Dict[str, Any]:
        if runtime.metrics is None:
            return {"counters": {}, "gauges": {}, "capabilities": runtime.get_capabilities()}
        snapshot = runtime.metrics.snapshot()
        snapshot["capabilities"] = runtime.get_capabilities()
        return snapshot

    @app.post("/api/perf/client")
    async def perf_client_ingest(request: Request) -> Dict[str, Any]:
        candidate = request.headers.get("x-api-key") or request.query_params.get("api_key")
        if not is_api_key_valid(candidate):
            raise HTTPException(status_code=401, detail="Invalid API key")
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        accepted = 0
        if runtime.metrics is not None:
            for key in ("metadata_lag_ms_p95", "overlay_fps", "overlay_jank_ratio", "janus_ttff_ms"):
                raw = payload.get(key)
                try:
                    value = float(raw)
                except Exception:
                    continue
                if not np.isfinite(value) or value < 0:
                    continue
                runtime.metrics.set_gauge(str(key), float(value))
                accepted += 1
        return {"ok": True, "accepted": accepted}

    @app.on_event("startup")
    async def _on_startup() -> None:
        LOGGER.info("API startup | media_root=%s", runtime.media_root)
        try:
            from api.identities import run_reindex

            report = run_reindex(runtime)
            LOGGER.info("Startup reindex complete | added=%s linked=%s orphans=%s", report["added"], report["linked"], report["orphans"])
            runtime.request_thumb_refresh(None)
        except Exception:
            LOGGER.exception("Startup reindex failed")

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        LOGGER.info("API shutdown | closing peer connections")
        await runtime.close_peer_connections()
        runtime.close()

    return app


def run_api_server(runtime: ApiRuntimeState, host: str, port: int, log_level: str = "warning") -> None:
    app = create_app(runtime)
    config = uvicorn.Config(app=app, host=host, port=int(port), log_level=log_level)
    server = uvicorn.Server(config=config)
    server.run()
