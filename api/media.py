"""Media routes: static mount + MJPEG fallback stream."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import MJPEG_FPS, MJPEG_MAX_CLIENTS, is_api_key_valid

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


async def _mjpeg_generator(runtime: Any, client_host: str) -> AsyncGenerator[bytes, None]:
    interval = max(1.0 / MJPEG_FPS, 0.01)
    runtime.register_mjpeg_client()
    LOGGER.info("MJPEG stream opened from %s", client_host)
    try:
        while True:
            try:
                frame_bytes = runtime.get_latest_jpeg()
                payload = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("ascii")
                    + frame_bytes
                    + b"\r\n"
                )
                yield payload
            except Exception:
                LOGGER.debug("MJPEG stream iteration failed", exc_info=True)

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        LOGGER.info("MJPEG stream cancelled for %s", client_host)
        raise
    except Exception:
        LOGGER.exception("MJPEG stream error for %s", client_host)
    finally:
        runtime.unregister_mjpeg_client()
        LOGGER.info("MJPEG stream closed for %s", client_host)


async def _stream_mjpeg_impl(request: Request) -> StreamingResponse:
    _ensure_authorized(request)
    runtime = _runtime(request)
    mjpeg_clients, _ = runtime.get_client_counts()
    if mjpeg_clients >= MJPEG_MAX_CLIENTS:
        raise HTTPException(status_code=503, detail="Too many MJPEG clients")
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


@router.get("/api/media/mjpeg")
async def stream_mjpeg_api(request: Request) -> StreamingResponse:
    return await _stream_mjpeg_impl(request)


@router.get("/stream.mjpeg")
async def stream_mjpeg_legacy(request: Request) -> StreamingResponse:
    return await _stream_mjpeg_impl(request)
