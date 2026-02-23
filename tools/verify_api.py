"""Quick API smoke-check utility for local development."""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.request

import websockets


BASE = "http://127.0.0.1:8080"


def get_json(path: str) -> dict | list:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=8) as response:
        return json.loads(response.read().decode("utf-8"))


def check_mjpeg() -> None:
    with urllib.request.urlopen(f"{BASE}/api/media/mjpeg", timeout=8) as response:
        chunk = response.read(128)
        if b"--frame" not in chunk:
            raise RuntimeError("MJPEG boundary not found")
        print("mjpeg: ok")


async def check_ws() -> None:
    async with websockets.connect("ws://127.0.0.1:8080/api/realtime/ws") as ws:
        message = await asyncio.wait_for(ws.recv(), timeout=5)
        print("ws: first message:", str(message)[:120])


def main() -> int:
    try:
        print("health:", get_json("/api/health"))
        identities = get_json("/api/identities")
        count = len(identities) if isinstance(identities, list) else 0
        print("identities:", count)
        print("debug:", get_json("/api/debug/status"))
        check_mjpeg()
        asyncio.run(check_ws())
        print("verify_api: success")
        return 0
    except Exception as exc:
        print("verify_api: failed:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
