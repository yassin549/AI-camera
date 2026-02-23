"""FastAPI server for identities and active tracks endpoints."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

from fastapi import FastAPI
import uvicorn

import db


class RuntimeState:
    """Thread-safe state shared between runtime threads and API handlers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tracks: Dict[int, Dict[str, Any]] = {}

    def set_tracks(self, tracks: Dict[int, Dict[str, Any]]) -> None:
        with self._lock:
            self._tracks = {int(k): dict(v) for k, v in tracks.items()}

    def get_tracks(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(v) for _, v in sorted(self._tracks.items())]


class _IdentityCache:
    """Short TTL cache to avoid heavy DB polling from frequent API clients."""

    def __init__(self, ttl_sec: float = 0.5) -> None:
        self.ttl_sec = float(ttl_sec)
        self.lock = threading.Lock()
        self.last_ts = 0.0
        self.cached: List[Dict[str, Any]] = []

    def get(self) -> List[Dict[str, Any]]:
        now = time.time()
        with self.lock:
            if now - self.last_ts >= self.ttl_sec:
                self.cached = db.list_identities()
                self.last_ts = now
            return list(self.cached)


def build_app(state: RuntimeState) -> FastAPI:
    app = FastAPI(title="AIcam API", version="2.0.0")
    identity_cache = _IdentityCache(ttl_sec=0.5)

    @app.get("/api/identities")
    def api_identities() -> List[Dict[str, Any]]:
        return identity_cache.get()

    @app.get("/api/tracks")
    def api_tracks() -> List[Dict[str, Any]]:
        return state.get_tracks()

    # Backward-compatible routes.
    @app.get("/identities")
    def identities() -> List[Dict[str, Any]]:
        return identity_cache.get()

    @app.get("/tracks")
    def tracks() -> List[Dict[str, Any]]:
        return state.get_tracks()

    return app


def run_api_server(state: RuntimeState, host: str, port: int, log_level: str = "warning") -> None:
    app = build_app(state)
    config = uvicorn.Config(app=app, host=host, port=int(port), log_level=log_level)
    server = uvicorn.Server(config=config)
    server.run()
