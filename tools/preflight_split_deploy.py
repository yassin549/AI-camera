"""Preflight checks for split deployment (Vercel frontend + local backend)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict

import websockets


def _read_env(key: str, default: str = "") -> str:
    return str(os.getenv(key, default)).strip()


def _ensure_base_url() -> str:
    base = _read_env("VITE_API_BASE") or _read_env("AICAM_PUBLIC_BACKEND")
    if not base:
        raise RuntimeError("Missing VITE_API_BASE (or AICAM_PUBLIC_BACKEND)")
    return base.rstrip("/")


def _api_key() -> str:
    return _read_env("VITE_API_KEY") or _read_env("API_KEY")


def _headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    key = _api_key()
    if key:
        headers["x-api-key"] = key
    return headers


def _get_json(base: str, path: str) -> Any:
    request = urllib.request.Request(f"{base}{path}", headers=_headers(), method="GET")
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.loads(response.read().decode("utf-8"))


async def _check_ws(ws_url: str) -> None:
    headers = _headers()
    async with websockets.connect(
        ws_url,
        additional_headers=headers if headers else None,
        open_timeout=8,
        close_timeout=2,
    ) as ws:
        first = await asyncio.wait_for(ws.recv(), timeout=8)
        if not str(first):
            raise RuntimeError("WS connected but received empty message")


def _derive_ws_url(base: str) -> str:
    explicit = _read_env("VITE_WS_METADATA_URL")
    if explicit:
        return explicit
    if base.startswith("https://"):
        return f"wss://{base[len('https://'):]}/api/realtime/ws"
    if base.startswith("http://"):
        return f"ws://{base[len('http://'):]}/api/realtime/ws"
    return f"{base}/api/realtime/ws"


def main() -> int:
    try:
        base = _ensure_base_url()
        ws_url = _derive_ws_url(base)
        print(f"preflight base={base}")

        health = _get_json(base, "/api/health")
        print("health:", json.dumps(health, separators=(",", ":"))[:240])
        if not bool(health.get("capture_running", False)):
            print("warning: capture_running is false")

        latest = _get_json(base, "/api/realtime/latest")
        tracks = latest.get("tracks", []) if isinstance(latest, dict) else []
        print(f"realtime/latest tracks={len(tracks)}")

        identities = _get_json(base, "/api/identities")
        if isinstance(identities, list):
            print(f"identities={len(identities)}")
        else:
            print("warning: identities payload is not a list")

        asyncio.run(_check_ws(ws_url))
        print(f"metadata ws ok ({ws_url})")
        print("preflight: success")
        return 0
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        print(f"preflight: failed HTTP {exc.code} {exc.reason} {body}".strip())
        return 1
    except Exception as exc:
        print(f"preflight: failed {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
