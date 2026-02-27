"""Probe Janus streaming mountpoint availability via HTTP API.

Usage (PowerShell):
  $env:JANUS_HTTP_URL="http://localhost:8088/janus"
  $env:JANUS_MOUNTPOINT="1"
  python tools/check_janus_mountpoint.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


def _tx() -> str:
    return f"tx{random.randint(100000, 999999)}"


def _post_json(base: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        raw = resp.read().decode("utf-8")
    out = json.loads(raw)
    if out.get("janus") == "error":
        err = out.get("error", {}) if isinstance(out.get("error"), dict) else {}
        code = err.get("code", "unknown")
        reason = err.get("reason", "unknown")
        raise RuntimeError(f"Janus error {code}: {reason}")
    return out


def _get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def main() -> int:
    base = _env("JANUS_HTTP_URL", "http://localhost:8088/janus").rstrip("/")
    mountpoint = int(_env("JANUS_MOUNTPOINT", "1"))
    attempts = max(1, int(_env("JANUS_POLL_ATTEMPTS", "6")))
    sleep_ms = max(100, int(_env("JANUS_POLL_SLEEP_MS", "250")))

    print(f"probe janus={base} mountpoint={mountpoint}")
    session_id: int | None = None

    try:
        created = _post_json(base, "", {"janus": "create", "transaction": _tx()})
        session_id = int(created.get("data", {}).get("id", -1))
        if session_id <= 0:
            raise RuntimeError("Failed to create Janus session")
        print(f"session={session_id}")

        attached = _post_json(
            base,
            f"/{session_id}",
            {"janus": "attach", "plugin": "janus.plugin.streaming", "transaction": _tx()},
        )
        handle_id = int(attached.get("data", {}).get("id", -1))
        if handle_id <= 0:
            raise RuntimeError("Failed to attach Janus streaming plugin")
        print(f"handle={handle_id}")

        _post_json(
            base,
            f"/{session_id}/{handle_id}",
            {
                "janus": "message",
                "transaction": _tx(),
                "body": {"request": "watch", "id": mountpoint},
            },
        )

        for idx in range(attempts):
            rid = int(time.time() * 1000)
            poll_url = f"{base}/{session_id}?{urllib.parse.urlencode({'rid': rid, 'maxev': 1})}"
            event = _get_json(poll_url)
            janus_type = str(event.get("janus", ""))
            if janus_type == "event":
                data = (
                    event.get("plugindata", {}).get("data", {})
                    if isinstance(event.get("plugindata"), dict)
                    else {}
                )
                if isinstance(data, dict) and data.get("error"):
                    print(f"janus_mountpoint_error: {data.get('error')}")
                    return 1
                if event.get("jsep"):
                    print("janus_mountpoint_ok: received offer from streaming plugin")
                    return 0
                print(f"janus_event: {json.dumps(data)}")
            time.sleep(sleep_ms / 1000.0)

        print("janus_mountpoint_probe_inconclusive: no offer event received in polling window")
        return 2
    except urllib.error.URLError as exc:
        print(f"janus_probe_failed_network: {exc}")
        return 1
    except Exception as exc:
        print(f"janus_probe_failed: {exc}")
        return 1
    finally:
        if session_id:
            try:
                _post_json(base, f"/{session_id}", {"janus": "destroy", "transaction": _tx()})
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

