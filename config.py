"""Runtime configuration for the AIcam API layer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

APP_NAME = "aicam"
APP_VERSION = "0.1"

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_PATH = Path(os.getenv("AICAM_STATIC_PATH", str(PROJECT_ROOT))).resolve()

# Match camera FPS for smooth streaming; override with AICAM_MJPEG_FPS env var.
MJPEG_FPS = max(1.0, float(os.getenv("AICAM_MJPEG_FPS", "30")))
MJPEG_JPEG_QUALITY = int(os.getenv("AICAM_MJPEG_QUALITY", "80"))
MJPEG_MAX_CLIENTS = max(1, int(os.getenv("AICAM_MJPEG_MAX_CLIENTS", "8")))
ENABLE_FRAME_STREAMING = os.getenv("AICAM_ENABLE_FRAME_STREAMING", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}

_cors_from_env = os.getenv("CORS_ORIGINS", "").strip()
if _cors_from_env:
    CORS_ORIGINS: List[str] = [item.strip() for item in _cors_from_env.split(",") if item.strip()]
else:
    CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", "").strip() or None

API_KEY = os.getenv("API_KEY", "").strip() or None
_janus_upstream = os.getenv("AICAM_JANUS_UPSTREAM", "http://127.0.0.1:8088/janus").strip()
JANUS_UPSTREAM_URL = (_janus_upstream or "http://127.0.0.1:8088/janus").rstrip("/")
DELETE_SAMPLE_FILES = os.getenv("AICAM_DELETE_SAMPLE_FILES", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

HEALTH_FRAME_STALE_SEC = max(1.0, float(os.getenv("AICAM_HEALTH_STALE_SEC", "5")))
WS_HEARTBEAT_SEC = max(3.0, float(os.getenv("AICAM_WS_HEARTBEAT_SEC", "10")))
WS_MAX_CLIENTS = max(1, int(os.getenv("AICAM_WS_MAX_CLIENTS", "64")))


def is_api_key_valid(candidate: Optional[str]) -> bool:
    """Return True when API key protection is disabled or a valid key is provided."""
    if API_KEY is None:
        return True
    return candidate == API_KEY
