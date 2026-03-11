"""Identity REST routes."""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel

import db
from api.media_utils import iter_media_candidates, resolve_media_url
from config import DELETE_SAMPLE_FILES, is_api_key_valid

LOGGER = logging.getLogger("aicam.api.identities")

router = APIRouter()


class RenameIdentityBody(BaseModel):
    name: str


class MergeIdentityBody(BaseModel):
    target_id: int


class MuteIdentityBody(BaseModel):
    muted: Optional[bool] = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _ensure_authorized(request: Request) -> None:
    if not is_api_key_valid(request.headers.get("x-api-key")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _runtime(request: Request) -> Any:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=500, detail="API runtime is not initialized")
    return runtime


def _db_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table_name: str) -> Set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return bool(row)


def _quote_ident(name: str) -> str:
    escaped = str(name).replace('"', '""')
    return f'"{escaped}"'


def _ensure_identity_samples_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS identity_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_id INTEGER NOT NULL,
            sample_path TEXT NOT NULL,
            sample_type TEXT NOT NULL DEFAULT 'face',
            created_ts TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_identity_samples_unique ON identity_samples(identity_id, sample_path)"
    )
    conn.commit()


def _ensure_name_column(conn: sqlite3.Connection, columns: Set[str]) -> Set[str]:
    if "name" in columns:
        return columns
    conn.execute("ALTER TABLE identities ADD COLUMN name TEXT")
    conn.commit()
    columns = set(columns)
    columns.add("name")
    return columns


def _ensure_muted_column(conn: sqlite3.Connection, columns: Set[str]) -> Set[str]:
    if "is_muted" in columns:
        return columns
    conn.execute("ALTER TABLE identities ADD COLUMN is_muted INTEGER NOT NULL DEFAULT 0")
    conn.commit()
    updated = set(columns)
    updated.add("is_muted")
    return updated


def _format_iso(value: Optional[str]) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(value)
    try:
        normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return text


def _discover_samples_from_filesystem(identity_id: int, media_root: Path) -> Tuple[List[str], List[str]]:
    face_dirs = [
        media_root / "faces",
        media_root / "data" / "faces",
        media_root / "samples" / "faces",
    ]
    body_dirs = [
        media_root / "faces" / "body",
        media_root / "body",
        media_root / "bodies",
        media_root / "data" / "bodies",
        media_root / "samples" / "bodies",
    ]
    patterns = [
        f"{identity_id}_*.jpg", f"{identity_id}_*.jpeg", f"{identity_id}_*.png",
        f"face_*_t{identity_id}.jpg", f"face_*_t{identity_id}.jpeg",
    ]

    def gather(paths: List[Path]) -> List[str]:
        out: List[str] = []
        for directory in paths:
            if not directory.exists() or not directory.is_dir():
                continue
            for pattern in patterns:
                for candidate in sorted(directory.glob(pattern)):
                    try:
                        rel = candidate.relative_to(media_root)
                    except Exception:
                        continue
                    out.append(f"/media/{rel.as_posix()}")
                    if len(out) >= 8:
                        return out
        return out

    return gather(face_dirs), gather(body_dirs)


def _load_extra_samples(
    conn: sqlite3.Connection,
    identity_id: int,
    media_root: Path,
) -> Tuple[List[str], List[str]]:
    _ensure_identity_samples_table(conn)
    rows_identity_samples = conn.execute(
        """
        SELECT sample_path, sample_type
        FROM identity_samples
        WHERE identity_id=?
        ORDER BY id DESC
        """,
        (identity_id,),
    ).fetchall()
    face_samples: List[str] = []
    body_samples: List[str] = []
    for row in rows_identity_samples:
        sample_url = resolve_media_url(row["sample_path"], media_root)
        if not sample_url:
            continue
        sample_type = str(row["sample_type"] or "").lower()
        if "body" in sample_type:
            body_samples.append(sample_url)
        else:
            face_samples.append(sample_url)

    if not _table_exists(conn, "samples"):
        return face_samples, body_samples
    cols = _table_columns(conn, "samples")
    identity_col = "identity_id" if "identity_id" in cols else ("id_identity" if "id_identity" in cols else None)
    path_col = "path" if "path" in cols else ("sample_path" if "sample_path" in cols else "file_path" if "file_path" in cols else None)
    type_col = "sample_type" if "sample_type" in cols else ("type" if "type" in cols else "modality" if "modality" in cols else None)
    if identity_col is not None and path_col is not None:
        type_expr = type_col if type_col is not None else "NULL"
        query = f"SELECT {path_col} AS sample_path, {type_expr} AS sample_type FROM samples WHERE {identity_col}=?"
        rows = conn.execute(query, (identity_id,)).fetchall()
        for row in rows:
            sample_url = resolve_media_url(row["sample_path"], media_root)
            if not sample_url:
                continue
            kind = str(row["sample_type"] or "").lower()
            if "body" in kind:
                body_samples.append(sample_url)
            else:
                face_samples.append(sample_url)
    return face_samples, body_samples


def _build_identity_select(columns: Set[str], include_embedding_meta: bool = False) -> str:
    name_expr = "name" if "name" in columns else "NULL"
    first_seen_expr = "created_ts" if "created_ts" in columns else ("first_seen" if "first_seen" in columns else "NULL")
    last_seen_expr = "last_seen_ts" if "last_seen_ts" in columns else ("last_seen" if "last_seen" in columns else first_seen_expr)
    face_path_expr = "face_sample_path" if "face_sample_path" in columns else ("sample_path" if "sample_path" in columns else "NULL")
    body_path_expr = "body_sample_path" if "body_sample_path" in columns else "NULL"
    frequency_expr = '"frequency"' if "frequency" in columns else ('"count"' if "count" in columns else "1")
    is_body_expr = "is_body_only" if "is_body_only" in columns else "0"
    is_muted_expr = "is_muted" if "is_muted" in columns else "0"

    fields = [
        "id",
        f"{name_expr} AS name",
        f"{first_seen_expr} AS first_seen",
        f"{last_seen_expr} AS last_seen",
        f"{face_path_expr} AS face_sample_path",
        f"{body_path_expr} AS body_sample_path",
        f"{frequency_expr} AS frequency",
        f"{is_body_expr} AS is_body_only",
        f"{is_muted_expr} AS is_muted",
    ]

    if include_embedding_meta:
        face_emb_col = "face_emb" if "face_emb" in columns else ("face_embedding" if "face_embedding" in columns else None)
        body_emb_col = "body_emb" if "body_emb" in columns else ("body_embedding" if "body_embedding" in columns else None)
        if face_emb_col:
            fields.append(f"CASE WHEN {face_emb_col} IS NOT NULL THEN 1 ELSE 0 END AS has_face_embedding")
        else:
            fields.append("0 AS has_face_embedding")
        if body_emb_col:
            fields.append(f"CASE WHEN {body_emb_col} IS NOT NULL THEN 1 ELSE 0 END AS has_body_embedding")
        else:
            fields.append("0 AS has_body_embedding")
    return ", ".join(fields)


def _row_to_payload(
    row: sqlite3.Row,
    conn: sqlite3.Connection,
    media_root: Path,
    include_embedding_meta: bool = False,
) -> Dict[str, Any]:
    identity_id = int(row["id"])
    face_samples: List[str] = []
    body_samples: List[str] = []

    face_single = resolve_media_url(row["face_sample_path"], media_root)
    body_single = resolve_media_url(row["body_sample_path"], media_root)
    if face_single:
        face_samples.append(face_single)
    if body_single:
        body_samples.append(body_single)

    extra_face, extra_body = _load_extra_samples(conn, identity_id, media_root)
    face_samples.extend(extra_face)
    body_samples.extend(extra_body)

    if not face_samples and not body_samples:
        fs_face, fs_body = _discover_samples_from_filesystem(identity_id, media_root)
        face_samples.extend(fs_face)
        body_samples.extend(fs_body)

    face_samples = sorted(set(face_samples))
    body_samples = sorted(set(body_samples))

    payload: Dict[str, Any] = {
        "id": identity_id,
        "display_name": str(row["name"] or f"Identity {identity_id}"),
        "name": str(row["name"] or f"Identity {identity_id}"),
        "first_seen": _format_iso(row["first_seen"]),
        "last_seen": _format_iso(row["last_seen"]),
        "frequency": int(row["frequency"] or 0),
        "is_muted": bool(row["is_muted"]),
        "sample_images": sorted(set(face_samples + body_samples)),
        "face_samples": face_samples,
        "body_samples": body_samples,
    }

    if include_embedding_meta:
        payload["embedding_meta"] = {
            "has_face_embedding": bool(row["has_face_embedding"]),
            "has_body_embedding": bool(row["has_body_embedding"]),
            "is_body_only": bool(row["is_body_only"]),
        }
    return payload


def _remove_samples(paths: Sequence[Optional[str]], media_root: Path) -> None:
    for raw_path in paths:
        for candidate in iter_media_candidates(raw_path, media_root):
            if candidate.exists():
                try:
                    candidate.unlink()
                except Exception:
                    LOGGER.warning("Failed to delete sample file: %s", candidate, exc_info=True)


def _guess_identity_id_from_path(path: Path) -> Optional[int]:
    name = path.stem.lower()
    patterns = [
        r"^(\d+)_",
        r"^identity[_-]?(\d+)",
        r"^id[_-]?(\d+)",
        r".*[_-]identity[_-]?(\d+)$",
        r".*[_-]id[_-]?(\d+)$",
        r".*[_-]t(\d+)$",
        r".*[_-](\d+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    parent = path.parent.name
    if parent.isdigit():
        return int(parent)
    return None


def _upsert_identity_sample(
    conn: sqlite3.Connection,
    identity_id: int,
    rel_path: str,
    sample_type: str,
) -> bool:
    _ensure_identity_samples_table(conn)
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO identity_samples(identity_id, sample_path, sample_type, created_ts)
        VALUES (?, ?, ?, ?)
        """,
        (identity_id, rel_path, sample_type, datetime.now(timezone.utc).isoformat()),
    )
    return cur.rowcount > 0


def run_reindex(runtime: Any) -> Dict[str, int]:
    media_root = Path(runtime.media_root)
    face_dirs = [
        media_root / "faces",
        media_root / "data" / "faces",
        media_root / "samples" / "faces",
    ]
    body_dirs = [
        media_root / "faces" / "body",
        media_root / "body",
        media_root / "bodies",
        media_root / "data" / "bodies",
        media_root / "samples" / "bodies",
    ]
    identity_dirs = [
        media_root / "identities",
        media_root / "data" / "identities",
        media_root / "samples" / "identities",
    ]

    report = {"added": 0, "linked": 0, "orphans": 0}
    try:
        with _db_connect(runtime.db_path) as conn:
            columns = _table_columns(conn, "identities")
            identity_ids = {int(row["id"]) for row in conn.execute("SELECT id FROM identities").fetchall()}
            _ensure_identity_samples_table(conn)

            scans: List[Tuple[Path, str]] = []
            scans.extend((d, "face") for d in face_dirs)
            scans.extend((d, "body") for d in body_dirs)
            scans.extend((d, "face") for d in identity_dirs)

            for directory, sample_type in scans:
                if not directory.exists() or not directory.is_dir():
                    continue
                for file_path in directory.rglob("*"):
                    if not file_path.is_file():
                        continue
                    if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    identity_id = _guess_identity_id_from_path(file_path)
                    if identity_id is None or identity_id not in identity_ids:
                        report["orphans"] += 1
                        continue
                    try:
                        rel_path = file_path.relative_to(media_root).as_posix()
                    except Exception:
                        report["orphans"] += 1
                        continue
                    inserted = _upsert_identity_sample(conn, identity_id, rel_path, sample_type)
                    if inserted:
                        report["added"] += 1

                    if sample_type == "face" and "face_sample_path" in columns:
                        updated = conn.execute(
                            "UPDATE identities SET face_sample_path=COALESCE(face_sample_path, ?) WHERE id=?",
                            (rel_path, identity_id),
                        ).rowcount
                        if updated:
                            report["linked"] += 1
                    elif sample_type == "body" and "body_sample_path" in columns:
                        updated = conn.execute(
                            "UPDATE identities SET body_sample_path=COALESCE(body_sample_path, ?) WHERE id=?",
                            (rel_path, identity_id),
                        ).rowcount
                        if updated:
                            report["linked"] += 1
            conn.commit()
    except Exception:
        LOGGER.exception("Identity reindex failed")
        raise
    return report


@router.get("")
def list_identities(request: Request) -> List[Dict[str, Any]]:
    _ensure_authorized(request)
    runtime = _runtime(request)
    try:
        with _db_connect(runtime.db_path) as conn:
            columns = _table_columns(conn, "identities")
            select_fields = _build_identity_select(columns, include_embedding_meta=False)
            order_by = "last_seen_ts DESC" if "last_seen_ts" in columns else ("last_seen DESC" if "last_seen" in columns else "id DESC")
            rows = conn.execute(f"SELECT {select_fields} FROM identities ORDER BY {order_by}").fetchall()
            return [_row_to_payload(row, conn, runtime.media_root, include_embedding_meta=False) for row in rows]
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Failed to list identities")
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc


@router.get("/{identity_id}")
def get_identity(identity_id: int, request: Request) -> Dict[str, Any]:
    _ensure_authorized(request)
    runtime = _runtime(request)
    try:
        with _db_connect(runtime.db_path) as conn:
            columns = _table_columns(conn, "identities")
            select_fields = _build_identity_select(columns, include_embedding_meta=True)
            row = conn.execute(
                f"SELECT {select_fields} FROM identities WHERE id=? LIMIT 1",
                (identity_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Identity not found")
            return _row_to_payload(row, conn, runtime.media_root, include_embedding_meta=True)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Failed to load identity %s", identity_id)
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc


@router.post("/{identity_id}/rename")
def rename_identity(identity_id: int, body: RenameIdentityBody, request: Request) -> Dict[str, Any]:
    _ensure_authorized(request)
    runtime = _runtime(request)
    next_name = body.name.strip()
    if not next_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    try:
        with _db_connect(runtime.db_path) as conn:
            columns = _table_columns(conn, "identities")
            columns = _ensure_name_column(conn, columns)
            updated = conn.execute(
                "UPDATE identities SET name=? WHERE id=?",
                (next_name, identity_id),
            ).rowcount
            conn.commit()
            if updated == 0:
                raise HTTPException(status_code=404, detail="Identity not found")
            select_fields = _build_identity_select(columns, include_embedding_meta=True)
            row = conn.execute(
                f"SELECT {select_fields} FROM identities WHERE id=? LIMIT 1",
                (identity_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Identity not found")
            return _row_to_payload(row, conn, runtime.media_root, include_embedding_meta=True)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Failed to rename identity %s", identity_id)
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc


@router.delete("/{identity_id}", status_code=204)
def delete_identity(identity_id: int, request: Request) -> Response:
    _ensure_authorized(request)
    runtime = _runtime(request)
    try:
        with _db_connect(runtime.db_path) as conn:
            columns = _table_columns(conn, "identities")
            face_col = "face_sample_path" if "face_sample_path" in columns else ("sample_path" if "sample_path" in columns else "NULL")
            body_col = "body_sample_path" if "body_sample_path" in columns else "NULL"
            row = conn.execute(
                f"SELECT {face_col} AS face_sample_path, {body_col} AS body_sample_path FROM identities WHERE id=? LIMIT 1",
                (identity_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Identity not found")
            conn.execute("DELETE FROM identities WHERE id=?", (identity_id,))
            conn.commit()

            if DELETE_SAMPLE_FILES:
                _remove_samples([row["face_sample_path"], row["body_sample_path"]], runtime.media_root)
            try:
                db.load_all_embeddings()
            except Exception:
                LOGGER.debug("Failed to reload embedding index after delete", exc_info=True)
            try:
                runtime.clear_identity(identity_id)
            except Exception:
                LOGGER.debug("Failed to clear live track identity after delete", exc_info=True)
            try:
                runtime.request_thumb_refresh(identity_id)
            except Exception:
                LOGGER.debug("Failed to refresh thumb cache after delete", exc_info=True)

        return Response(status_code=204)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Failed to delete identity %s", identity_id)
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc


@router.post("/{identity_id}/snapshot")
def snapshot_identity(identity_id: int, request: Request) -> Dict[str, Any]:
    _ensure_authorized(request)
    runtime = _runtime(request)

    with _db_connect(runtime.db_path) as conn:
        identity_exists = conn.execute("SELECT 1 FROM identities WHERE id=? LIMIT 1", (identity_id,)).fetchone()
        if identity_exists is None:
            raise HTTPException(status_code=404, detail="Identity not found")

    packet = runtime.frame_store.get_latest()
    if packet is None or packet.frame is None:
        raise HTTPException(status_code=409, detail="No frame available for snapshot")
    frame = packet.frame
    if frame.size == 0:
        raise HTTPException(status_code=409, detail="Latest frame is empty")

    selected_bbox: Optional[Tuple[int, int, int, int]] = None
    try:
        tracks = runtime.get_tracks()
        candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []
        for track in tracks:
            try:
                tracked_identity = track.get("identity_id")
                if tracked_identity is None or int(tracked_identity) != int(identity_id):
                    continue
                bbox_raw = track.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(v) for v in bbox_raw]
                if x2 <= x1 or y2 <= y1:
                    continue
                age_frames = int(track.get("age_frames", 0))
                candidates.append((age_frames, (x1, y1, x2, y2)))
            except Exception:
                continue
        if candidates:
            candidates.sort(key=lambda item: item[0])
            selected_bbox = candidates[0][1]
    except Exception:
        LOGGER.debug("Could not derive bbox for snapshot identity=%s", identity_id, exc_info=True)

    frame_h, frame_w = frame.shape[:2]
    if selected_bbox is not None:
        x1, y1, x2, y2 = selected_bbox
        x1 = max(0, min(frame_w - 1, x1))
        x2 = max(0, min(frame_w, x2))
        y1 = max(0, min(frame_h - 1, y1))
        y2 = max(0, min(frame_h, y2))
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame
    else:
        crop = frame

    if crop.size == 0:
        raise HTTPException(status_code=500, detail="Snapshot crop failed")

    ts_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    rel_path = Path("snapshots") / f"identity_{identity_id}" / f"{ts_token}.jpg"
    abs_path = runtime.media_root / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok = cv2.imwrite(str(abs_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not encoded_ok:
        raise HTTPException(status_code=500, detail="Failed to encode snapshot")

    rel_text = rel_path.as_posix()
    try:
        with _db_connect(runtime.db_path) as conn:
            columns = _table_columns(conn, "identities")
            identity_exists = conn.execute("SELECT 1 FROM identities WHERE id=? LIMIT 1", (identity_id,)).fetchone()
            if identity_exists is None:
                raise HTTPException(status_code=404, detail="Identity not found")
            _ensure_identity_samples_table(conn)
            _upsert_identity_sample(conn, identity_id, rel_text, "snapshot")
            if "face_sample_path" in columns:
                conn.execute(
                    "UPDATE identities SET face_sample_path=COALESCE(face_sample_path, ?) WHERE id=?",
                    (rel_text, identity_id),
                )
            conn.commit()
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Failed to persist snapshot for identity %s", identity_id)
        raise HTTPException(status_code=500, detail=f"Snapshot persistence failed: {exc}") from exc

    try:
        runtime.request_thumb_refresh(identity_id)
    except Exception:
        LOGGER.debug("Failed to refresh thumb cache after snapshot", exc_info=True)

    return {"ok": True, "identity_id": identity_id, "sample": f"/media/{rel_text}"}


@router.post("/{identity_id}/mute")
def mute_identity(identity_id: int, request: Request, body: Optional[MuteIdentityBody] = None) -> Dict[str, Any]:
    _ensure_authorized(request)
    runtime = _runtime(request)
    desired_muted = None if body is None else body.muted
    try:
        with _db_connect(runtime.db_path) as conn:
            columns = _table_columns(conn, "identities")
            columns = _ensure_muted_column(conn, columns)
            row = conn.execute(
                "SELECT is_muted FROM identities WHERE id=? LIMIT 1",
                (identity_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Identity not found")
            current = bool(row["is_muted"])
            next_muted = (not current) if desired_muted is None else bool(desired_muted)
            conn.execute(
                "UPDATE identities SET is_muted=? WHERE id=?",
                (1 if next_muted else 0, identity_id),
            )
            conn.commit()
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Failed to update mute state for identity %s", identity_id)
        raise HTTPException(status_code=500, detail=f"Mute failed: {exc}") from exc

    try:
        if not db.set_identity_muted(identity_id, next_muted):
            db.load_all_embeddings()
    except Exception:
        LOGGER.debug("Failed to sync in-memory mute state, forcing embedding reload", exc_info=True)
        try:
            db.load_all_embeddings()
        except Exception:
            LOGGER.debug("Embedding reload after mute also failed", exc_info=True)

    try:
        runtime.request_thumb_refresh(identity_id)
    except Exception:
        LOGGER.debug("Failed to refresh thumb cache after mute", exc_info=True)
    return {"ok": True, "identity_id": identity_id, "muted": next_muted}


@router.post("/{identity_id}/merge")
def merge_identity(identity_id: int, body: MergeIdentityBody, request: Request) -> Dict[str, Any]:
    _ensure_authorized(request)
    runtime = _runtime(request)
    target_id = int(body.target_id)
    source_id = int(identity_id)
    if source_id == target_id:
        raise HTTPException(status_code=400, detail="Source and target identities must be different")

    def _pick_earliest(a: Optional[str], b: Optional[str]) -> str:
        values = [str(v) for v in (a, b) if v]
        if not values:
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return min(values)

    def _pick_latest(a: Optional[str], b: Optional[str]) -> str:
        values = [str(v) for v in (a, b) if v]
        if not values:
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return max(values)

    try:
        with _db_connect(runtime.db_path) as conn:
            columns = _table_columns(conn, "identities")
            columns = _ensure_name_column(conn, columns)
            columns = _ensure_muted_column(conn, columns)

            source = conn.execute("SELECT * FROM identities WHERE id=? LIMIT 1", (source_id,)).fetchone()
            target = conn.execute("SELECT * FROM identities WHERE id=? LIMIT 1", (target_id,)).fetchone()
            if source is None or target is None:
                raise HTTPException(status_code=404, detail="Source or target identity not found")

            face_emb_col = "face_emb" if "face_emb" in columns else ("face_embedding" if "face_embedding" in columns else None)
            body_emb_col = "body_emb" if "body_emb" in columns else ("body_embedding" if "body_embedding" in columns else None)
            first_seen_col = "created_ts" if "created_ts" in columns else ("first_seen" if "first_seen" in columns else None)
            last_seen_col = "last_seen_ts" if "last_seen_ts" in columns else ("last_seen" if "last_seen" in columns else None)
            face_path_col = "face_sample_path" if "face_sample_path" in columns else ("sample_path" if "sample_path" in columns else None)
            body_path_col = "body_sample_path" if "body_sample_path" in columns else None
            frequency_col = "frequency" if "frequency" in columns else ("count" if "count" in columns else None)
            is_body_only_col = "is_body_only" if "is_body_only" in columns else None

            update_pairs: List[Tuple[str, Any]] = []
            if face_emb_col is not None:
                merged_face = target[face_emb_col] if target[face_emb_col] is not None else source[face_emb_col]
                update_pairs.append((face_emb_col, merged_face))
            if body_emb_col is not None:
                merged_body = target[body_emb_col] if target[body_emb_col] is not None else source[body_emb_col]
                update_pairs.append((body_emb_col, merged_body))

            if first_seen_col is not None:
                update_pairs.append((first_seen_col, _pick_earliest(source[first_seen_col], target[first_seen_col])))
            if last_seen_col is not None:
                update_pairs.append((last_seen_col, _pick_latest(source[last_seen_col], target[last_seen_col])))
            if face_path_col is not None:
                update_pairs.append((face_path_col, target[face_path_col] or source[face_path_col]))
            if body_path_col is not None:
                update_pairs.append((body_path_col, target[body_path_col] or source[body_path_col]))
            if "name" in columns:
                source_name = str(source["name"] or "").strip()
                target_name = str(target["name"] or "").strip()
                update_pairs.append(("name", target_name or source_name or f"Identity {target_id}"))
            if frequency_col is not None:
                src_freq = int(source[frequency_col] or 0)
                dst_freq = int(target[frequency_col] or 0)
                update_pairs.append((frequency_col, src_freq + dst_freq))
            if "is_muted" in columns:
                merged_muted = bool(source["is_muted"]) or bool(target["is_muted"])
                update_pairs.append(("is_muted", 1 if merged_muted else 0))
            if is_body_only_col is not None and face_emb_col is not None:
                effective_face = target[face_emb_col] if target[face_emb_col] is not None else source[face_emb_col]
                update_pairs.append((is_body_only_col, 1 if effective_face is None else 0))

            if update_pairs:
                set_clause = ", ".join(f"{_quote_ident(column)}=?" for column, _ in update_pairs)
                params = [value for _, value in update_pairs]
                params.append(target_id)
                conn.execute(f"UPDATE identities SET {set_clause} WHERE id=?", tuple(params))

            if _table_exists(conn, "identity_samples"):
                _ensure_identity_samples_table(conn)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO identity_samples(identity_id, sample_path, sample_type, created_ts)
                    SELECT ?, sample_path, sample_type, created_ts
                    FROM identity_samples
                    WHERE identity_id=?
                    """,
                    (target_id, source_id),
                )
                conn.execute("DELETE FROM identity_samples WHERE identity_id=?", (source_id,))

            if _table_exists(conn, "samples"):
                sample_columns = _table_columns(conn, "samples")
                sample_identity_col = (
                    "identity_id"
                    if "identity_id" in sample_columns
                    else ("id_identity" if "id_identity" in sample_columns else None)
                )
                if sample_identity_col is not None:
                    conn.execute(
                        f"UPDATE samples SET {_quote_ident(sample_identity_col)}=? WHERE {_quote_ident(sample_identity_col)}=?",
                        (target_id, source_id),
                    )

            conn.execute("DELETE FROM identities WHERE id=?", (source_id,))
            conn.commit()
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Failed to merge identity %s into %s", source_id, target_id)
        raise HTTPException(status_code=500, detail=f"Merge failed: {exc}") from exc

    try:
        db.load_all_embeddings()
    except Exception:
        LOGGER.debug("Failed to reload embedding index after merge", exc_info=True)
    try:
        runtime.remap_identity(source_id, target_id)
    except Exception:
        LOGGER.debug("Failed to remap live track identity after merge", exc_info=True)
    try:
        runtime.request_thumb_refresh(source_id)
        runtime.request_thumb_refresh(target_id)
    except Exception:
        LOGGER.debug("Failed to refresh thumb cache after merge", exc_info=True)

    return {"ok": True, "source_id": source_id, "merged_into": str(target_id)}


@router.post("/reindex")
def reindex_identities(request: Request) -> Dict[str, int]:
    _ensure_authorized(request)
    runtime = _runtime(request)
    report = run_reindex(runtime)
    try:
        runtime.request_thumb_refresh(None)
    except Exception:
        LOGGER.debug("Failed to refresh thumb cache after reindex", exc_info=True)
    LOGGER.info(
        "Manual reindex complete | added=%s linked=%s orphans=%s",
        report["added"],
        report["linked"],
        report["orphans"],
    )
    return report
