"""FastAPI server for AIcam identities + media + realtime transports."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    patterns = [f"{identity_id}_*.jpg", f"{identity_id}_*.jpeg", f"{identity_id}_*.png"]
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

    def is_capture_running(self) -> bool:
        packet = self.frame_store.get_latest()
        if packet is None:
            return False
        age = time.time() - float(packet.ts)
        return age <= HEALTH_FRAME_STALE_SEC

    def get_tracks(self) -> List[Dict[str, Any]]:
        with self._tracks_lock:
            return [dict(v) for _, v in sorted(self._tracks.items())]

    def publish_tracks(
        self,
        tracks_payload: Dict[int, Dict[str, Any]],
        frame_id: int,
        frame_shape: Tuple[int, int, int],
    ) -> None:
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

    @app.get("/")
    def root() -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": APP_NAME,
            "version": APP_VERSION,
            "status": "ok",
            "routes": [
                "/api/identities",
                "/api/identities/{id}",
                "/ws/metadata",
                "/webrtc/offer",
                "/stream.mjpeg",
            ],
        }
        if not AIORTC_AVAILABLE:
            payload["webrtc"] = "disabled (aiortc unavailable); use /stream.mjpeg"
        return payload

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        running = runtime.is_capture_running()
        if not running:
            raise HTTPException(status_code=503, detail="capture_not_running")
        return {"ok": True, "capture": "running"}

    @app.get("/api/tracks")
    def api_tracks() -> List[Dict[str, Any]]:
        return runtime.get_tracks()

    @app.get("/tracks")
    def tracks_legacy() -> List[Dict[str, Any]]:
        return runtime.get_tracks()

    @app.on_event("startup")
    async def _on_startup() -> None:
        LOGGER.info("API startup | media_root=%s", runtime.media_root)

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
