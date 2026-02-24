"""Media routes: static mount + MJPEG fallback stream."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator

from fastapi import APIRouter, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import ENABLE_FRAME_STREAMING, MJPEG_FPS, MJPEG_MAX_CLIENTS, is_api_key_valid

LOGGER = logging.getLogger("aicam.api.media")

router = APIRouter()


def mount_static_media(app: FastAPI, media_root: str) -> None:
    app.mount("/media", StaticFiles(directory=media_root, check_dir=False), name="media")


def _ensure_authorized(request: Request) -> None:
    if not is_api_key_valid(request.headers.get("x-api-key")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _ensure_ws_authorized(websocket: WebSocket) -> bool:
    candidate = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
    return is_api_key_valid(candidate)


def _runtime(request: Request) -> Any:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=500, detail="API runtime is not initialized")
    return runtime


def _runtime_from_ws(websocket: WebSocket) -> Any:
    runtime = getattr(websocket.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("API runtime is not initialized")
    return runtime


async def _mjpeg_generator(runtime: Any, client_host: str) -> AsyncGenerator[bytes, None]:
    target_interval = max(1.0 / MJPEG_FPS, 0.005)
    poll_interval = min(target_interval * 0.5, 0.01)
    last_frame_id = -1
    last_sent_at = 0.0
    runtime.register_mjpeg_client()
    LOGGER.info("MJPEG stream opened from %s", client_host)
    try:
        while True:
            try:
                frame_id, frame_bytes = runtime.get_latest_jpeg_packet()
                if frame_id < 0:
                    await asyncio.sleep(poll_interval)
                    continue
                if frame_id == last_frame_id:
                    await asyncio.sleep(poll_interval)
                    continue

                now = asyncio.get_running_loop().time()
                elapsed = now - last_sent_at
                if elapsed < target_interval:
                    await asyncio.sleep(target_interval - elapsed)
                last_sent_at = asyncio.get_running_loop().time()
                last_frame_id = frame_id
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
    if not ENABLE_FRAME_STREAMING:
        raise HTTPException(status_code=503, detail="frame_streaming_disabled")
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


async def _stream_wsjpeg_impl(websocket: WebSocket) -> None:
    if not _ensure_ws_authorized(websocket):
        await websocket.close(code=4401, reason="Invalid API key")
        return
    if not ENABLE_FRAME_STREAMING:
        await websocket.close(code=4403, reason="Frame streaming disabled")
        return

    runtime = _runtime_from_ws(websocket)
    mjpeg_clients, _ = runtime.get_client_counts()
    if mjpeg_clients >= MJPEG_MAX_CLIENTS:
        await websocket.close(code=4403, reason="Too many video clients")
        return

    host = websocket.client.host if websocket.client else "unknown"
    await websocket.accept()
    runtime.register_mjpeg_client()
    LOGGER.info("WS JPEG stream opened from %s", host)

    target_interval = max(1.0 / MJPEG_FPS, 0.005)
    poll_interval = min(target_interval * 0.5, 0.01)
    last_frame_id = -1
    last_sent_at = 0.0
    try:
        while True:
            frame_id, frame_bytes = runtime.get_latest_jpeg_packet()
            if frame_id < 0 or frame_id == last_frame_id:
                await asyncio.sleep(poll_interval)
                continue

            now = asyncio.get_running_loop().time()
            elapsed = now - last_sent_at
            if elapsed < target_interval:
                await asyncio.sleep(target_interval - elapsed)
            await websocket.send_bytes(frame_bytes)
            last_sent_at = asyncio.get_running_loop().time()
            last_frame_id = frame_id
    except WebSocketDisconnect:
        LOGGER.info("WS JPEG stream disconnected from %s", host)
    except Exception:
        LOGGER.exception("WS JPEG stream error for %s", host)
    finally:
        runtime.unregister_mjpeg_client()
        try:
            await websocket.close()
        except Exception:
            pass
        LOGGER.info("WS JPEG stream closed for %s", host)


@router.get("/api/media/mjpeg")
async def stream_mjpeg_api(request: Request) -> StreamingResponse:
    return await _stream_mjpeg_impl(request)


@router.get("/stream.mjpeg")
async def stream_mjpeg_legacy(request: Request) -> StreamingResponse:
    return await _stream_mjpeg_impl(request)


@router.websocket("/api/media/ws")
async def stream_wsjpeg_api(websocket: WebSocket) -> None:
    await _stream_wsjpeg_impl(websocket)


@router.websocket("/ws/video")
async def stream_wsjpeg_legacy(websocket: WebSocket) -> None:
    await _stream_wsjpeg_impl(websocket)
