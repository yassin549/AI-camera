"""Realtime routes: WebSocket metadata + WebRTC offer/answer."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import WS_HEARTBEAT_SEC, WS_MAX_CLIENTS, is_api_key_valid

LOGGER = logging.getLogger("aicam.api.realtime")

router = APIRouter()

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from av import VideoFrame

    AIORTC_AVAILABLE = True
except Exception:  # pragma: no cover - availability depends on host platform
    RTCPeerConnection = None
    RTCSessionDescription = None
    VideoStreamTrack = object  # type: ignore[assignment]
    VideoFrame = None
    AIORTC_AVAILABLE = False


class WebRtcOfferBody(BaseModel):
    sdp: str
    type: str = "offer"


def _runtime_from_request(request: Request) -> Any:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=500, detail="API runtime is not initialized")
    return runtime


def _runtime_from_websocket(websocket: WebSocket) -> Any:
    runtime = getattr(websocket.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("API runtime is not initialized")
    return runtime


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class FrameStoreVideoTrack(VideoStreamTrack):  # type: ignore[misc]
    """aiortc track backed by the in-process latest frame store."""

    kind = "video"

    def __init__(self, runtime: Any) -> None:
        super().__init__()
        self.runtime = runtime
        self._fallback = np.zeros((360, 640, 3), dtype=np.uint8)
        self._last_frame_id = -1
        self._last_frame = self._fallback

    async def recv(self):  # type: ignore[override]
        pts, time_base = await self.next_timestamp()
        packet = self.runtime.frame_store.get_latest()
        if packet is not None and int(packet.frame_id) != self._last_frame_id:
            self._last_frame_id = int(packet.frame_id)
            self._last_frame = packet.frame
        frame = VideoFrame.from_ndarray(self._last_frame, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


async def _wait_for_ice_gathering_complete(pc: Any, timeout_sec: float = 2.0) -> None:
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    async def _on_ice_state_change() -> None:
        if pc.iceGatheringState == "complete":
            done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        pass


def _normalize_offer(raw_body: bytes, content_type: str) -> WebRtcOfferBody:
    if "application/json" in content_type.lower():
        payload = json.loads(raw_body.decode("utf-8") or "{}")
    else:
        payload = {"sdp": raw_body.decode("utf-8"), "type": "offer"}
    return WebRtcOfferBody(**payload)


async def _ws_metadata_impl(websocket: WebSocket) -> None:
    candidate = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
    if not is_api_key_valid(candidate):
        await websocket.close(code=4401, reason="Invalid API key")
        return

    runtime = _runtime_from_websocket(websocket)
    _, ws_clients = runtime.get_client_counts()
    if ws_clients >= WS_MAX_CLIENTS:
        await websocket.close(code=4403, reason="Too many WS clients")
        return

    await websocket.accept()
    runtime.register_ws_client()
    client = websocket.client.host if websocket.client else "unknown"
    LOGGER.info("WS metadata connected from %s", client)
    last_version = 0
    next_heartbeat_at = time.monotonic() + WS_HEARTBEAT_SEC

    await websocket.send_json({"type": "connected", "server_time": _utc_now_iso()})

    try:
        while True:
            now = time.monotonic()
            wait_timeout = max(0.001, min(0.2, next_heartbeat_at - now))
            version, payload, _ = await asyncio.to_thread(
                runtime.metadata_hub.wait_for_update,
                last_version,
                wait_timeout,
            )
            if payload is not None and version > last_version:
                await websocket.send_json(payload)
                last_version = version
                next_heartbeat_at = time.monotonic() + WS_HEARTBEAT_SEC
            elif time.monotonic() >= next_heartbeat_at:
                await websocket.send_json(
                    {
                        "type": "heartbeat",
                        "server_time": _utc_now_iso(),
                        "hint": "reconnect if stream remains idle",
                    }
                )
                next_heartbeat_at = time.monotonic() + WS_HEARTBEAT_SEC

            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if message.strip().lower() == "ping":
                    await websocket.send_json({"type": "pong", "server_time": _utc_now_iso()})
                elif message.strip().lower() == "close":
                    break
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        LOGGER.info("WS metadata disconnected from %s", client)
    except Exception:
        LOGGER.exception("WS metadata error for %s", client)
    finally:
        runtime.unregister_ws_client()
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/api/realtime/ws")
async def ws_metadata_api(websocket: WebSocket) -> None:
    await _ws_metadata_impl(websocket)


@router.websocket("/ws/metadata")
async def ws_metadata_legacy(websocket: WebSocket) -> None:
    await _ws_metadata_impl(websocket)


@router.post("/webrtc/offer")
async def webrtc_offer(request: Request) -> JSONResponse:
    if not is_api_key_valid(request.headers.get("x-api-key")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    runtime = _runtime_from_request(request)

    if not AIORTC_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WebRTC disabled (aiortc unavailable). Use /stream.mjpeg fallback.",
        )

    raw_body = await request.body()
    content_type = request.headers.get("content-type", "")
    try:
        offer = _normalize_offer(raw_body, content_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid offer payload: {exc}") from exc

    pc = RTCPeerConnection()
    runtime.register_peer_connection(pc)
    client = request.client.host if request.client else "unknown"
    LOGGER.info("WebRTC offer received from %s", client)

    @pc.on("connectionstatechange")
    async def _on_connectionstatechange() -> None:
        state = pc.connectionState
        LOGGER.info("WebRTC state=%s for %s", state, client)
        if state in {"failed", "closed", "disconnected"}:
            runtime.unregister_peer_connection(pc)
            await pc.close()

    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
        pc.addTrack(FrameStoreVideoTrack(runtime))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_for_ice_gathering_complete(pc)
        local = pc.localDescription
        if local is None:
            raise RuntimeError("WebRTC local description is empty")
        LOGGER.info("WebRTC answer created for %s", client)
        return JSONResponse({"sdp": local.sdp, "type": local.type})
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("WebRTC negotiation failed for %s", client)
        runtime.unregister_peer_connection(pc)
        try:
            await pc.close()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"WebRTC negotiation failed: {exc}") from exc


@router.post("/api/media/webrtc/offer")
async def webrtc_offer_api(request: Request) -> JSONResponse:
    return await webrtc_offer(request)
