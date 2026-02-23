"""SQLite-backed identity store with in-memory vector indices."""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from utils import cosine, l2_normalize, load_np_from_blob, save_np_to_blob, timestamp_iso

LOGGER = logging.getLogger(__name__)


@dataclass
class _WriteJob:
    sql: str
    params: tuple
    wait: bool
    done: threading.Event
    result: Optional[int] = None
    error: Optional[Exception] = None


class IdentityStore:
    """Thread-safe identity index backed by SQLite."""

    def __init__(self, db_path: str, ephemeral: bool = False) -> None:
        self.db_path = db_path
        self.ephemeral = bool(ephemeral)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.lock = threading.RLock()
        self._write_q: "queue.Queue[Optional[_WriteJob]]" = queue.Queue(maxsize=2048)
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer.start()

        self.face_ids: List[int] = []
        self.face_matrix = np.zeros((0, 128), dtype=np.float32)
        self.body_ids: List[int] = []
        self.body_matrix = np.zeros((0, 256), dtype=np.float32)
        self._row_cache: Dict[int, Dict[str, object]] = {}

        self._init_schema()
        self.load_all_embeddings()

    @staticmethod
    def _coerce_dim(vec: np.ndarray, dim: int) -> np.ndarray:
        """Force embedding to fixed length for stable matrix ops."""
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size >= dim:
            arr = arr[:dim]
        else:
            arr = np.pad(arr, (0, dim - arr.size), mode="constant")
        return l2_normalize(arr)

    def close(self) -> None:
        self._write_q.put(None)
        self._writer.join(timeout=1.0)
        self.conn.close()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.execute(self._schema_sql("identities"))

        cols = self._table_columns("identities")
        if not cols:
            return

        required = {
            "id",
            "face_emb",
            "body_emb",
            "face_sample_path",
            "body_sample_path",
            "created_ts",
            "last_seen_ts",
            "is_body_only",
        }
        if required.issubset(cols):
            return

        legacy = {"id", "first_seen", "last_seen", "face_embedding", "body_embedding"}
        if legacy.issubset(cols):
            self._migrate_legacy_schema()
            return

        # Partial schema: add missing compatible columns in-place.
        alter_map = {
            "face_emb": "ALTER TABLE identities ADD COLUMN face_emb BLOB",
            "body_emb": "ALTER TABLE identities ADD COLUMN body_emb BLOB",
            "face_sample_path": "ALTER TABLE identities ADD COLUMN face_sample_path TEXT",
            "body_sample_path": "ALTER TABLE identities ADD COLUMN body_sample_path TEXT",
            "created_ts": "ALTER TABLE identities ADD COLUMN created_ts TEXT",
            "last_seen_ts": "ALTER TABLE identities ADD COLUMN last_seen_ts TEXT",
            "is_body_only": "ALTER TABLE identities ADD COLUMN is_body_only INTEGER NOT NULL DEFAULT 0",
        }
        with self.conn:
            for col, stmt in alter_map.items():
                if col not in cols:
                    self.conn.execute(stmt)

            # Backfill new columns where possible.
            if "first_seen" in cols:
                self.conn.execute(
                    "UPDATE identities SET created_ts=COALESCE(created_ts, first_seen)"
                )
            if "last_seen" in cols:
                self.conn.execute(
                    "UPDATE identities SET last_seen_ts=COALESCE(last_seen_ts, last_seen)"
                )
            now = timestamp_iso()
            self.conn.execute(
                "UPDATE identities SET created_ts=COALESCE(created_ts, ?), "
                "last_seen_ts=COALESCE(last_seen_ts, ?)",
                (now, now),
            )
            if "face_embedding" in cols:
                self.conn.execute(
                    "UPDATE identities SET face_emb=COALESCE(face_emb, face_embedding)"
                )
            if "body_embedding" in cols:
                self.conn.execute(
                    "UPDATE identities SET body_emb=COALESCE(body_emb, body_embedding)"
                )
            self.conn.execute(
                "UPDATE identities SET is_body_only=CASE "
                "WHEN face_emb IS NULL THEN 1 ELSE 0 END "
                "WHERE is_body_only IS NULL"
            )

    @staticmethod
    def _schema_sql(table_name: str) -> str:
        return f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_emb BLOB,
                body_emb BLOB,
                face_sample_path TEXT,
                body_sample_path TEXT,
                created_ts TEXT NOT NULL,
                last_seen_ts TEXT NOT NULL,
                is_body_only INTEGER NOT NULL DEFAULT 0
            )
        """

    def _table_columns(self, table_name: str) -> Set[str]:
        rows = self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(r[1]) for r in rows}

    def _migrate_legacy_schema(self) -> None:
        LOGGER.warning("Migrating legacy identities schema to v2 columns.")
        tmp_table = "identities_v2_tmp"
        backup_table = f"identities_legacy_{int(time.time())}"

        with self.conn:
            self.conn.execute(f"DROP TABLE IF EXISTS {tmp_table}")
            self.conn.execute(self._schema_sql(tmp_table))

            rows = self.conn.execute(
                """
                SELECT id, first_seen, last_seen, face_embedding, body_embedding
                FROM identities
                ORDER BY rowid ASC
                """
            ).fetchall()

            used_ids: Set[int] = set()
            for row in rows:
                old_id, first_seen, last_seen, face_blob, body_blob = row
                created_ts = first_seen or timestamp_iso()
                last_seen_ts = last_seen or created_ts
                is_body_only = 1 if face_blob is None else 0

                numeric_id: Optional[int] = None
                try:
                    numeric_id = int(str(old_id))
                except Exception:
                    numeric_id = None
                if numeric_id is not None and numeric_id > 0 and numeric_id not in used_ids:
                    used_ids.add(numeric_id)
                    self.conn.execute(
                        f"""
                        INSERT INTO {tmp_table} (
                            id, face_emb, body_emb, face_sample_path, body_sample_path,
                            created_ts, last_seen_ts, is_body_only
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            numeric_id,
                            face_blob,
                            body_blob,
                            None,
                            None,
                            created_ts,
                            last_seen_ts,
                            is_body_only,
                        ),
                    )
                else:
                    self.conn.execute(
                        f"""
                        INSERT INTO {tmp_table} (
                            face_emb, body_emb, face_sample_path, body_sample_path,
                            created_ts, last_seen_ts, is_body_only
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            face_blob,
                            body_blob,
                            None,
                            None,
                            created_ts,
                            last_seen_ts,
                            is_body_only,
                        ),
                    )

            self.conn.execute(f"ALTER TABLE identities RENAME TO {backup_table}")
            self.conn.execute(f"ALTER TABLE {tmp_table} RENAME TO identities")
            LOGGER.warning("Legacy schema migrated. Backup kept as table %s.", backup_table)

    def _writer_loop(self) -> None:
        while True:
            job = self._write_q.get()
            if job is None:
                return
            try:
                if self.ephemeral:
                    if job.wait:
                        job.result = -1
                    continue
                with self.conn:
                    cur = self.conn.execute(job.sql, job.params)
                    if job.wait:
                        job.result = int(cur.lastrowid)
            except Exception as exc:  # pragma: no cover - defensive path
                job.error = exc
            finally:
                if job.wait:
                    job.done.set()

    def _enqueue(self, sql: str, params: tuple, wait: bool = False) -> int:
        done = threading.Event()
        job = _WriteJob(sql=sql, params=params, wait=wait, done=done)
        self._write_q.put(job)
        if not wait:
            return -1
        done.wait(timeout=5.0)
        if job.error is not None:
            raise job.error
        return int(job.result or -1)

    def load_all_embeddings(self) -> Dict[str, object]:
        with self.lock:
            cur = self.conn.execute(
                """
                SELECT id, face_emb, body_emb, face_sample_path, body_sample_path,
                       created_ts, last_seen_ts, is_body_only
                FROM identities
                ORDER BY id ASC
                """
            )
            rows = cur.fetchall()

            self.face_ids = []
            self.body_ids = []
            face_vecs: List[np.ndarray] = []
            body_vecs: List[np.ndarray] = []
            self._row_cache = {}

            for row in rows:
                (
                    ident_id,
                    face_blob,
                    body_blob,
                    face_sample_path,
                    body_sample_path,
                    created_ts,
                    last_seen_ts,
                    is_body_only,
                ) = row

                self._row_cache[int(ident_id)] = {
                    "id": int(ident_id),
                    "face_sample_path": face_sample_path,
                    "body_sample_path": body_sample_path,
                    "created_ts": created_ts,
                    "last_seen_ts": last_seen_ts,
                    "is_body_only": bool(is_body_only),
                }

                if face_blob is not None:
                    emb = self._coerce_dim(load_np_from_blob(face_blob), 128)
                    self.face_ids.append(int(ident_id))
                    face_vecs.append(emb)
                if body_blob is not None:
                    emb = self._coerce_dim(load_np_from_blob(body_blob), 256)
                    self.body_ids.append(int(ident_id))
                    body_vecs.append(emb)

            self.face_matrix = (
                np.vstack(face_vecs).astype(np.float32)
                if face_vecs
                else np.zeros((0, 128), dtype=np.float32)
            )
            self.body_matrix = (
                np.vstack(body_vecs).astype(np.float32)
                if body_vecs
                else np.zeros((0, 256), dtype=np.float32)
            )

            return {
                "face_ids": self.face_ids,
                "face_embeddings": self.face_matrix,
                "body_ids": self.body_ids,
                "body_embeddings": self.body_matrix,
            }

    def add_identity(
        self,
        face_emb: Optional[np.ndarray],
        body_emb: Optional[np.ndarray],
        face_sample_path: Optional[str],
        body_sample_path: Optional[str],
        ts: str,
    ) -> int:
        norm_face = None if face_emb is None else self._coerce_dim(face_emb, 128)
        norm_body = None if body_emb is None else self._coerce_dim(body_emb, 256)

        if self.ephemeral:
            with self.lock:
                next_id = max(self._row_cache.keys(), default=0) + 1
                self._row_cache[next_id] = {
                    "id": next_id,
                    "face_sample_path": face_sample_path,
                    "body_sample_path": body_sample_path,
                    "created_ts": ts,
                    "last_seen_ts": ts,
                    "is_body_only": face_emb is None,
                }
                if norm_face is not None:
                    self.face_ids.append(next_id)
                    self.face_matrix = np.vstack([self.face_matrix, norm_face])
                if norm_body is not None:
                    self.body_ids.append(next_id)
                    self.body_matrix = np.vstack([self.body_matrix, norm_body])
            return next_id

        face_blob = None if norm_face is None else save_np_to_blob(norm_face)
        body_blob = None if norm_body is None else save_np_to_blob(norm_body)
        ident_id = self._enqueue(
            """
            INSERT INTO identities (
                face_emb, body_emb, face_sample_path, body_sample_path,
                created_ts, last_seen_ts, is_body_only
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                face_blob,
                body_blob,
                face_sample_path,
                body_sample_path,
                ts,
                ts,
                1 if face_emb is None else 0,
            ),
            wait=True,
        )

        with self.lock:
            self._row_cache[ident_id] = {
                "id": ident_id,
                "face_sample_path": face_sample_path,
                "body_sample_path": body_sample_path,
                "created_ts": ts,
                "last_seen_ts": ts,
                "is_body_only": face_emb is None,
            }
            if norm_face is not None:
                self.face_ids.append(ident_id)
                self.face_matrix = np.vstack([self.face_matrix, norm_face])
            if norm_body is not None:
                self.body_ids.append(ident_id)
                self.body_matrix = np.vstack([self.body_matrix, norm_body])

        return ident_id

    def update_last_seen(self, ident_id: int, ts: str) -> None:
        with self.lock:
            if ident_id in self._row_cache:
                self._row_cache[ident_id]["last_seen_ts"] = ts
        if not self.ephemeral:
            self._enqueue(
                "UPDATE identities SET last_seen_ts=? WHERE id=?",
                (ts, ident_id),
                wait=False,
            )

    def update_body_ema(self, ident_id: int, body_emb: np.ndarray, alpha: float = 0.2) -> None:
        """Update body embedding using an EMA and refresh in-memory index."""
        vec = self._coerce_dim(body_emb, 256)
        with self.lock:
            try:
                idx = self.body_ids.index(ident_id)
                current = self.body_matrix[idx]
                merged = l2_normalize((1.0 - alpha) * current + alpha * vec)
                self.body_matrix[idx] = merged
            except ValueError:
                self.body_ids.append(ident_id)
                self.body_matrix = np.vstack([self.body_matrix, vec])
                merged = vec
        if not self.ephemeral:
            self._enqueue(
                "UPDATE identities SET body_emb=? WHERE id=?",
                (save_np_to_blob(merged), ident_id),
                wait=False,
            )

    def find_best_face(self, face_emb: np.ndarray) -> Tuple[Optional[int], float]:
        query = self._coerce_dim(face_emb, 128)
        with self.lock:
            if self.face_matrix.shape[0] == 0:
                return None, -1.0
            sims = self.face_matrix @ query
            idx = int(np.argmax(sims))
            return self.face_ids[idx], float(sims[idx])

    def find_best_body(self, body_emb: np.ndarray) -> Tuple[Optional[int], float]:
        query = self._coerce_dim(body_emb, 256)
        with self.lock:
            if self.body_matrix.shape[0] == 0:
                return None, -1.0
            sims = self.body_matrix @ query
            idx = int(np.argmax(sims))
            return self.body_ids[idx], float(sims[idx])

    def list_identities(self) -> List[Dict[str, object]]:
        with self.lock:
            result: List[Dict[str, object]] = []
            for ident_id in sorted(self._row_cache):
                entry = dict(self._row_cache[ident_id])
                entry["has_face"] = ident_id in self.face_ids
                entry["has_body"] = ident_id in self.body_ids
                result.append(entry)
            return result


_DEFAULT_STORE: Optional[IdentityStore] = None


def configure(db_path: str, ephemeral: bool = False) -> IdentityStore:
    """Initialize and set module-level default store."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is not None:
        _DEFAULT_STORE.close()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _DEFAULT_STORE = IdentityStore(db_path=db_path, ephemeral=ephemeral)
    LOGGER.info("DB configured at %s (ephemeral=%s)", db_path, ephemeral)
    return _DEFAULT_STORE


def _store() -> IdentityStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = IdentityStore("identities.db")
    return _DEFAULT_STORE


def init_db() -> None:
    _store()._init_schema()


def add_identity(
    face_emb: Optional[np.ndarray],
    body_emb: Optional[np.ndarray],
    face_sample_path: Optional[str],
    body_sample_path: Optional[str],
    ts: str,
) -> int:
    return _store().add_identity(face_emb, body_emb, face_sample_path, body_sample_path, ts)


def find_best_face(face_emb: np.ndarray) -> Tuple[Optional[int], float]:
    return _store().find_best_face(face_emb)


def find_best_body(body_emb: np.ndarray) -> Tuple[Optional[int], float]:
    return _store().find_best_body(body_emb)


def update_last_seen(ident_id: int, ts: str) -> None:
    _store().update_last_seen(ident_id, ts)


def load_all_embeddings() -> Dict[str, object]:
    return _store().load_all_embeddings()


def update_body_ema(ident_id: int, body_emb: np.ndarray, alpha: float = 0.2) -> None:
    _store().update_body_ema(ident_id=ident_id, body_emb=body_emb, alpha=alpha)


def list_identities() -> List[Dict[str, object]]:
    return _store().list_identities()


def now_iso() -> str:
    return timestamp_iso()
