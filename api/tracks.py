"""Track action routes for manual live-ops assignment."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

import db
from config import is_api_key_valid

router = APIRouter()


class AssignTrackIdentityBody(BaseModel):
    identity_id: int
    cache_seconds: Optional[float] = None


def _ensure_authorized(request: Request) -> None:
    candidate = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if not is_api_key_valid(candidate):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _runtime(request: Request) -> Any:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=500, detail="API runtime is not initialized")
    return runtime


@router.post("/api/tracks/{track_id}/assign")
def assign_track_identity(track_id: int, body: AssignTrackIdentityBody, request: Request) -> Dict[str, Any]:
    _ensure_authorized(request)
    runtime = _runtime(request)

    tid = int(track_id)
    ident_id = int(body.identity_id)
    if tid < 0:
        raise HTTPException(status_code=400, detail="track_id must be >= 0")
    if ident_id <= 0:
        raise HTTPException(status_code=400, detail="identity_id must be > 0")
    if not db.identity_exists(ident_id):
        raise HTTPException(status_code=404, detail="Identity not found")

    cache_seconds = 30.0 if body.cache_seconds is None else float(body.cache_seconds)
    cache_seconds = max(0.0, min(300.0, cache_seconds))
    assigned = runtime.assign_track_identity(
        track_id=tid,
        identity_id=ident_id,
        cache_seconds=cache_seconds,
    )
    if not assigned:
        raise HTTPException(status_code=404, detail="Track not found")
    return {
        "ok": True,
        "track_id": tid,
        "identity_id": ident_id,
        "cache_seconds": cache_seconds,
    }
