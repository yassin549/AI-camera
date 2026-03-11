"""Media routes: static mount + MJPEG fallback stream."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator

from fastapi import APIRouter, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import (
    ENABLE_FRAME_STREAMING,
    MJPEG_FPS,
    MJPEG_JPEG_QUALITY,
    MJPEG_JPEG_QUALITY_MIN,
    MJPEG_MAX_CLIENTS,
    WSJPEG_ADAPTIVE,
    WSJPEG_FAST_SEND_MS,
    WSJPEG_MIN_FPS,
    WSJPEG_QUALITY_STEP,
    WSJPEG_SLOW_SEND_MS,
    is_api_key_valid,
)

LOGGER = logging.getLogger("aicam.api.media")

router = APIRouter()


def mount_static_media(app: FastAPI, media_root: str) -> None:
    app.mount("/media", StaticFiles(directory=media_root, check_dir=False), name="media")


def _ensure_authorized(request: Request) -> None:
    candidate = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if not is_api_key_valid(candidate):
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
    runtime.set_stream_quality(MJPEG_JPEG_QUALITY)
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
    mjpeg_clients, video_ws_clients = runtime.get_video_client_counts()
    if (mjpeg_clients + video_ws_clients) >= MJPEG_MAX_CLIENTS:
        raise HTTPException(status_code=503, detail="Too many video clients")
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
    mjpeg_clients, video_ws_clients = runtime.get_video_client_counts()
    if (mjpeg_clients + video_ws_clients) >= MJPEG_MAX_CLIENTS:
        await websocket.close(code=4403, reason="Too many video clients")
        return

    host = websocket.client.host if websocket.client else "unknown"
    await websocket.accept()
    runtime.register_video_ws_client()
    LOGGER.info("WS JPEG stream opened from %s", host)

    requested_fps_raw = websocket.query_params.get("fps")
    requested_quality_raw = websocket.query_params.get("quality") or websocket.query_params.get("q")
    requested_adaptive_raw = websocket.query_params.get("adaptive")
    requested_fps: float | None = None
    requested_quality: int | None = None
    if requested_fps_raw:
        try:
            requested_fps = float(requested_fps_raw)
        except Exception:
            requested_fps = None
    if requested_quality_raw:
        try:
            requested_quality = int(float(requested_quality_raw))
        except Exception:
            requested_quality = None
    effective_fps = MJPEG_FPS
    if requested_fps is not None and requested_fps > 0:
        effective_fps = min(MJPEG_FPS, max(1.0, requested_fps))
    target_fps = float(effective_fps)
    effective_fps = max(WSJPEG_MIN_FPS, float(effective_fps))

    max_quality = max(1, min(100, int(MJPEG_JPEG_QUALITY)))
    min_quality = max(1, min(max_quality, int(MJPEG_JPEG_QUALITY_MIN)))
    effective_quality = max_quality
    if requested_quality is not None:
        effective_quality = max(min_quality, min(max_quality, int(requested_quality)))
    runtime.set_stream_quality(effective_quality)

    adaptive = WSJPEG_ADAPTIVE
    if requested_adaptive_raw:
        adaptive = requested_adaptive_raw.strip().lower() not in {"0", "false", "off", "no"}

    target_interval = max(1.0 / effective_fps, 0.005)
    poll_interval = min(target_interval * 0.5, 0.01)
    last_frame_id = -1
    last_sent_at = 0.0
    stable_fast_windows = 0
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
            send_started_at = asyncio.get_running_loop().time()
            await websocket.send_bytes(frame_bytes)
            last_sent_at = asyncio.get_running_loop().time()
            send_ms = (last_sent_at - send_started_at) * 1000.0
            last_frame_id = frame_id

            if runtime.metrics is not None:
                runtime.metrics.set_gauge("wsjpeg_effective_fps", float(effective_fps))
                runtime.metrics.set_gauge("wsjpeg_effective_quality", float(effective_quality))
                runtime.metrics.set_gauge("wsjpeg_send_ms", float(send_ms))

            if not adaptive:
                continue

            if send_ms > WSJPEG_SLOW_SEND_MS:
                stable_fast_windows = 0
                if effective_quality > min_quality:
                    effective_quality = max(min_quality, effective_quality - WSJPEG_QUALITY_STEP)
                    runtime.set_stream_quality(effective_quality)
                elif effective_fps > WSJPEG_MIN_FPS:
                    effective_fps = max(WSJPEG_MIN_FPS, effective_fps - 1.0)
                target_interval = max(1.0 / effective_fps, 0.005)
                poll_interval = min(target_interval * 0.5, 0.01)
                continue

            if send_ms < WSJPEG_FAST_SEND_MS:
                stable_fast_windows += 1
                if stable_fast_windows >= 5:
                    stable_fast_windows = 0
                    if effective_quality < max_quality:
                        effective_quality = min(max_quality, effective_quality + WSJPEG_QUALITY_STEP)
                        runtime.set_stream_quality(effective_quality)
                    elif effective_fps < target_fps:
                        effective_fps = min(target_fps, effective_fps + 1.0)
                    target_interval = max(1.0 / effective_fps, 0.005)
                    poll_interval = min(target_interval * 0.5, 0.01)
            else:
                stable_fast_windows = max(0, stable_fast_windows - 1)
    except WebSocketDisconnect:
        LOGGER.info("WS JPEG stream disconnected from %s", host)
    except Exception:
        LOGGER.exception("WS JPEG stream error for %s", host)
    finally:
        runtime.unregister_video_ws_client()
        try:
            await websocket.close()
        except Exception:
            pass
        LOGGER.info("WS JPEG stream closed for %s", host)


@router.get("/api/media/mjpeg")
async def stream_mjpeg_api(request: Request) -> StreamingResponse:
    return await _stream_mjpeg_impl(request)


@router.websocket("/api/media/ws")
async def stream_wsjpeg_api(websocket: WebSocket) -> None:
    await _stream_wsjpeg_impl(websocket)


@router.websocket("/ws/video")
async def stream_wsjpeg_legacy(websocket: WebSocket) -> None:
    await _stream_wsjpeg_impl(websocket)
