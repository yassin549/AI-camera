"""Janus HTTP signaling reverse proxy routes."""

from __future__ import annotations

import asyncio
import logging
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request, Response, status

from config import JANUS_UPSTREAM_URL, is_api_key_valid

LOGGER = logging.getLogger("aicam.api.janus_proxy")

router = APIRouter()


def _ensure_authorized(request: Request) -> None:
    candidate = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if not is_api_key_valid(candidate):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _upstream_url(path: str, query: str) -> str:
    base = JANUS_UPSTREAM_URL.rstrip("/")
    suffix = path.lstrip("/")
    target = f"{base}/{suffix}" if suffix else base
    if query:
        return f"{target}?{query}"
    return target


def _forward_sync(
    method: str,
    url: str,
    body: Optional[bytes],
    content_type: Optional[str],
) -> Tuple[int, bytes, Dict[str, str]]:
    headers: Dict[str, str] = {}
    if content_type:
        headers["Content-Type"] = content_type

    request = urllib.request.Request(
        url=url,
        data=body if method == "POST" else None,
        method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=70) as response:
            payload = response.read()
            out_headers: Dict[str, str] = {}
            upstream_content_type = response.headers.get("Content-Type")
            if upstream_content_type:
                out_headers["Content-Type"] = upstream_content_type
            return int(getattr(response, "status", 200)), payload, out_headers
    except urllib.error.HTTPError as exc:
        payload = exc.read() if exc.fp is not None else b""
        out_headers: Dict[str, str] = {}
        upstream_content_type = exc.headers.get("Content-Type") if exc.headers else None
        if upstream_content_type:
            out_headers["Content-Type"] = upstream_content_type
        return int(exc.code), payload, out_headers


@router.api_route("/janus", methods=["GET", "POST"])
@router.api_route("/janus/{upstream_path:path}", methods=["GET", "POST"])
async def janus_http_proxy(request: Request, upstream_path: str = "") -> Response:
    _ensure_authorized(request)
    method = request.method.upper()
    query = request.url.query
    target = _upstream_url(upstream_path, query)
    body = await request.body() if method == "POST" else None
    content_type = request.headers.get("content-type")

    try:
        status_code, payload, headers = await asyncio.to_thread(
            _forward_sync,
            method,
            target,
            body,
            content_type,
        )
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise HTTPException(status_code=502, detail=f"Janus upstream unavailable: {reason}") from exc
    except Exception as exc:
        LOGGER.exception("Janus proxy error for %s", target)
        raise HTTPException(status_code=502, detail=f"Janus proxy failure: {exc}") from exc

    return Response(content=payload, status_code=status_code, headers=headers)

