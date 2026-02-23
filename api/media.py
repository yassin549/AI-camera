"""Media routes: static mount + MJPEG fallback stream."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator

import cv2
from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import MJPEG_FPS, MJPEG_JPEG_QUALITY, is_api_key_valid

LOGGER = logging.getLogger("aicam.api.media")

router = APIRouter()


def mount_static_media(app: FastAPI, media_root: str) -> None:
    app.mount("/media", StaticFiles(directory=media_root, check_dir=False), name="media")


def _ensure_authorized(request: Request) -> None:
    if not is_api_key_valid(request.headers.get("x-api-key")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _runtime(request: Request) -> Any:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=500, detail="API runtime is not initialized")
    return runtime


def _encode_jpeg(frame) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(MJPEG_JPEG_QUALITY)],
    )
    if not ok:
        raise RuntimeError("Failed to encode MJPEG frame")
    return encoded.tobytes()


async def _mjpeg_generator(runtime: Any, client_host: str) -> AsyncGenerator[bytes, None]:
    interval = max(1.0 / MJPEG_FPS, 0.01)
    last_frame_id = -1
    last_jpeg: bytes | None = None
    LOGGER.info("MJPEG stream opened from %s", client_host)
    try:
        while True:
            packet = runtime.frame_store.get_latest()
            if packet is not None and packet.frame_id != last_frame_id:
                try:
                    last_jpeg = _encode_jpeg(packet.frame)
                    last_frame_id = int(packet.frame_id)
                except Exception:
                    LOGGER.debug("MJPEG encode failed", exc_info=True)

            if last_jpeg is not None:
                payload = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(last_jpeg)}\r\n\r\n".encode("ascii")
                    + last_jpeg
                    + b"\r\n"
                )
                yield payload

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        LOGGER.info("MJPEG stream cancelled for %s", client_host)
        raise
    finally:
        LOGGER.info("MJPEG stream closed for %s", client_host)


@router.get("/stream.mjpeg")
async def stream_mjpeg(request: Request) -> StreamingResponse:
    _ensure_authorized(request)
    runtime = _runtime(request)
    host = request.client.host if request.client else "unknown"
    return StreamingResponse(
        _mjpeg_generator(runtime, host),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
        },
    )
