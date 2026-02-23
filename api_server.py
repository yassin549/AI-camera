"""FastAPI server for AIcam identities + media + realtime transports."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.identities import router as identities_router
from api.media import mount_static_media, router as media_router
from api.realtime import AIORTC_AVAILABLE, router as realtime_router
from config import (
    APP_NAME,
    APP_VERSION,
    CORS_ORIGINS,
    HEALTH_FRAME_STALE_SEC,
    MJPEG_JPEG_QUALITY,
    STATIC_PATH,
)

LOGGER = logging.getLogger("aicam.api")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_media_url(raw_path: Optional[str], media_root: Path) -> Optional[str]:
    if not raw_path:
        return None
    media_root = media_root.resolve()
    sample = Path(str(raw_path))
    candidates: List[Path] = []
    if sample.is_absolute():
        candidates.append(sample)
    else:
        candidates.append((media_root / sample).resolve())
        candidates.append((Path.cwd() / sample).resolve())
    if sample.name:
        for subdir in ("faces", "body", "bodies", "samples/faces", "samples/bodies", "data/faces", "data/bodies"):
            candidates.append((media_root / subdir / sample.name).resolve())

    for candidate in candidates:
        try:
            rel = candidate.resolve().relative_to(media_root)
        except Exception:
            continue
        if candidate.exists():
            return f"/media/{rel.as_posix()}"
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
                rel = candidate.resolve().relative_to(media_root.resolve())
            except Exception:
                continue
            return f"/media/{rel.as_posix()}"
    return None


class MetadataHub:
    """Thread-safe latest-snapshot publisher for low-latency websocket fanout."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._version = 0
        self._latest: Optional[Dict[str, Any]] = None

    def publish(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._version += 1
            self._latest = payload

    def snapshot(self) -> Tuple[int, Optional[Dict[str, Any]]]:
        with self._lock:
            return self._version, dict(self._latest) if self._latest is not None else None


class ApiRuntimeState:
    """Shared state consumed by API routes and updated from pipeline threads."""

    def __init__(self, frame_store: Any, db_path: str, media_root: Optional[str] = None) -> None:
        self.frame_store = frame_store
        self.db_path = str(db_path)
        self.media_root = Path(media_root or STATIC_PATH).resolve()

        self._tracks_lock = threading.RLock()
        self._tracks: Dict[int, Dict[str, Any]] = {}

        self._thumb_lock = threading.Lock()
        self._thumb_cache: Dict[int, str] = {}
        self._thumb_cache_expire_at = 0.0

        self._peer_lock = threading.Lock()
        self._peer_connections: Set[Any] = set()

        self._client_lock = threading.Lock()
        self._mjpeg_clients = 0
        self._ws_clients = 0

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = -1
        self._latest_frame_ts = 0.0
        self._latest_jpeg: Optional[bytes] = None
        self._latest_jpeg_frame_id = -1
        self._placeholder_jpeg = self._build_placeholder_jpeg()

        self.metadata_hub = MetadataHub()

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

    def update_frame(self, frame: np.ndarray, frame_id: int) -> None:
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_frame_id = int(frame_id)
            self._latest_frame_ts = time.time()

    def get_latest_jpeg(self) -> bytes:
        packet = self.frame_store.get_latest()
        if packet is not None:
            with self._frame_lock:
                if int(packet.frame_id) != self._latest_frame_id:
                    self._latest_frame = packet.frame
                    self._latest_frame_id = int(packet.frame_id)
                    self._latest_frame_ts = float(packet.ts)

        with self._frame_lock:
            frame = self._latest_frame
            frame_id = self._latest_frame_id
            if frame is None:
                return self._placeholder_jpeg
            if self._latest_jpeg is not None and self._latest_jpeg_frame_id == frame_id:
                return self._latest_jpeg

        try:
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(MJPEG_JPEG_QUALITY)],
            )
            if not ok:
                return self._placeholder_jpeg
            jpeg = encoded.tobytes()
        except Exception:
            LOGGER.debug("Failed to encode latest MJPEG frame", exc_info=True)
            return self._placeholder_jpeg

        with self._frame_lock:
            self._latest_jpeg = jpeg
            self._latest_jpeg_frame_id = frame_id
        return jpeg

    def get_tracks(self) -> List[Dict[str, Any]]:
        with self._tracks_lock:
            return [dict(v) for _, v in sorted(self._tracks.items())]

    def publish_tracks(
        self,
        tracks_payload: Dict[int, Dict[str, Any]],
        frame_id: int,
        frame_shape: Tuple[int, int, int],
    ) -> None:
        packet = self.frame_store.get_latest()
        if packet is not None:
            self.update_frame(packet.frame, int(packet.frame_id))
        with self._tracks_lock:
            self._tracks = {int(k): dict(v) for k, v in tracks_payload.items()}
        metadata = self._build_metadata_payload(frame_id=frame_id, frame_shape=frame_shape, tracks=tracks_payload)
        self.metadata_hub.publish(metadata)

    def _refresh_thumb_cache_if_needed(self) -> None:
        now = time.time()
        if now < self._thumb_cache_expire_at:
            return
        with self._thumb_lock:
            if now < self._thumb_cache_expire_at:
                return
            next_cache: Dict[int, str] = {}
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
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
                    thumb = _to_media_url(row["face_sample_path"], self.media_root) or _to_media_url(
                        row["body_sample_path"], self.media_root
                    )
                    if not thumb:
                        thumb = _discover_face_thumb(ident_id, self.media_root)
                    if thumb:
                        next_cache[ident_id] = thumb
            except Exception:
                LOGGER.debug("Thumb cache refresh failed", exc_info=True)
            self._thumb_cache = next_cache
            self._thumb_cache_expire_at = now + 1.0

    def _build_metadata_payload(
        self,
        frame_id: int,
        frame_shape: Tuple[int, int, int],
        tracks: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        self._refresh_thumb_cache_if_needed()
        source_height = int(frame_shape[0]) if frame_shape else 0
        source_width = int(frame_shape[1]) if len(frame_shape) > 1 else 0
        out_tracks: List[Dict[str, Any]] = []

        for _, track in sorted(tracks.items(), key=lambda item: int(item[0])):
            track_id = int(track.get("track_id", -1))
            identity_id = track.get("identity_id", None)
            modality = str(track.get("modality", "none"))
            score = float(track.get("last_score", 0.0))
            confidence = float(track.get("score", score))

            bbox_raw = track.get("bbox", [0, 0, 0, 0])
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox_raw]
            except Exception:
                x1, y1, x2, y2 = (0, 0, 0, 0)
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            if identity_id is None:
                label = f"Track {track_id}"
                thumb = None
            else:
                ident_int = int(identity_id)
                label = f"ID:{ident_int} ({score:.2f})"
                thumb = self._thumb_cache.get(ident_int)

            out_tracks.append(
                {
                    "track_id": track_id,
                    "bbox": [x1, y1, w, h],
                    "identity_id": None if identity_id is None else int(identity_id),
                    "label": label,
                    "modality": modality,
                    "confidence": confidence,
                    "thumb": thumb,
                }
            )

        return {
            "frame_id": int(frame_id),
            "timestamp": _utc_now_iso(),
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
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
            "routes": [
                "/api/identities",
                "/api/identities/{id}",
                "/api/realtime/ws",
                "/webrtc/offer",
                "/api/media/mjpeg",
            ],
        }
        if not AIORTC_AVAILABLE:
            payload["webrtc"] = "disabled (aiortc unavailable); use /api/media/mjpeg"
        return payload

    @app.get("/api/health")
    def api_health() -> Dict[str, Any]:
        return {"status": "ok", "time": _utc_now_iso()}

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
        }

    @app.on_event("startup")
    async def _on_startup() -> None:
        LOGGER.info("API startup | media_root=%s", runtime.media_root)
        try:
            from api.identities import run_reindex

            report = run_reindex(runtime)
            LOGGER.info("Startup reindex complete | added=%s linked=%s orphans=%s", report["added"], report["linked"], report["orphans"])
        except Exception:
            LOGGER.exception("Startup reindex failed")

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        LOGGER.info("API shutdown | closing peer connections")
        await runtime.close_peer_connections()

    return app


def run_api_server(runtime: ApiRuntimeState, host: str, port: int, log_level: str = "warning") -> None:
    app = create_app(runtime)
    config = uvicorn.Config(app=app, host=host, port=int(port), log_level=log_level)
    server = uvicorn.Server(config=config)
    server.run()
