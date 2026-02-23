"""Identity REST routes."""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel

from config import DELETE_SAMPLE_FILES, is_api_key_valid

LOGGER = logging.getLogger("aicam.api.identities")

router = APIRouter()


class RenameIdentityBody(BaseModel):
    name: str


class MergeIdentityBody(BaseModel):
    target_id: int


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


def _path_to_media_url(raw_path: Optional[str], media_root: Path) -> Optional[str]:
    if not raw_path:
        return None
    media_root = media_root.resolve()
    path = Path(str(raw_path))
    candidates: List[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((media_root / path).resolve())
        candidates.append((Path.cwd() / path).resolve())

    basename = path.name
    if basename:
        for subdir in ("faces", "body", "bodies", "samples/faces", "samples/bodies", "data/faces", "data/bodies"):
            candidates.append((media_root / subdir / basename).resolve())

    for candidate in candidates:
        try:
            rel = candidate.resolve().relative_to(media_root)
        except Exception:
            continue
        if candidate.exists():
            return f"/media/{rel.as_posix()}"
    return None


def _path_candidates(raw_path: Optional[str], media_root: Path) -> Iterable[Path]:
    if not raw_path:
        return []
    path = Path(str(raw_path))
    out: List[Path] = []
    if path.is_absolute():
        out.append(path)
    else:
        out.append((media_root / path).resolve())
        out.append((Path.cwd() / path).resolve())
    basename = path.name
    if basename:
        for subdir in ("faces", "body", "bodies", "samples/faces", "samples/bodies", "data/faces", "data/bodies"):
            out.append((media_root / subdir / basename).resolve())
    return out


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
    patterns = [f"{identity_id}_*.jpg", f"{identity_id}_*.jpeg", f"{identity_id}_*.png"]

    def gather(paths: List[Path]) -> List[str]:
        out: List[str] = []
        for directory in paths:
            if not directory.exists() or not directory.is_dir():
                continue
            for pattern in patterns:
                for candidate in sorted(directory.glob(pattern)):
                    try:
                        rel = candidate.resolve().relative_to(media_root.resolve())
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
        sample_url = _path_to_media_url(row["sample_path"], media_root)
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
            sample_url = _path_to_media_url(row["sample_path"], media_root)
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

    fields = [
        "id",
        f"{name_expr} AS name",
        f"{first_seen_expr} AS first_seen",
        f"{last_seen_expr} AS last_seen",
        f"{face_path_expr} AS face_sample_path",
        f"{body_path_expr} AS body_sample_path",
        f"{frequency_expr} AS frequency",
        f"{is_body_expr} AS is_body_only",
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

    face_single = _path_to_media_url(row["face_sample_path"], media_root)
    body_single = _path_to_media_url(row["body_sample_path"], media_root)
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
        for candidate in _path_candidates(raw_path, media_root):
            try:
                resolved = candidate.resolve()
            except Exception:
                continue
            try:
                resolved.relative_to(media_root.resolve())
            except Exception:
                continue
            if resolved.exists():
                try:
                    resolved.unlink()
                except Exception:
                    LOGGER.warning("Failed to delete sample file: %s", resolved, exc_info=True)


def _guess_identity_id_from_path(path: Path) -> Optional[int]:
    name = path.stem.lower()
    patterns = [
        r"^(\d+)_",
        r"^identity[_-]?(\d+)",
        r"^id[_-]?(\d+)",
        r".*[_-](\d+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, name)
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
    media_root = runtime.media_root.resolve()
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
                    rel_path = file_path.resolve().relative_to(media_root).as_posix()
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

        return Response(status_code=204)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Failed to delete identity %s", identity_id)
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc


@router.post("/{identity_id}/snapshot")
def snapshot_identity(identity_id: int, request: Request) -> Dict[str, Any]:
    _ensure_authorized(request)
    return {"ok": True, "identity_id": identity_id}


@router.post("/{identity_id}/mute")
def mute_identity(identity_id: int, request: Request) -> Dict[str, Any]:
    _ensure_authorized(request)
    return {"ok": True, "identity_id": identity_id}


@router.post("/{identity_id}/merge")
def merge_identity(identity_id: int, body: MergeIdentityBody, request: Request) -> Dict[str, Any]:
    _ensure_authorized(request)
    return {"ok": True, "source_id": identity_id, "merged_into": str(body.target_id)}


@router.post("/reindex")
def reindex_identities(request: Request) -> Dict[str, int]:
    _ensure_authorized(request)
    runtime = _runtime(request)
    report = run_reindex(runtime)
    LOGGER.info(
        "Manual reindex complete | added=%s linked=%s orphans=%s",
        report["added"],
        report["linked"],
        report["orphans"],
    )
    return report
