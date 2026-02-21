from __future__ import annotations

import io
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import cosine_similarity_batch


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS identities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB NOT NULL,
            first_seen TEXT,
            last_seen TEXT,
            sample_path TEXT
        );
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_identities_last_seen ON identities(last_seen);"
    )
    conn.commit()


def serialize_embedding(embedding: np.ndarray) -> bytes:
    arr = np.asarray(embedding, dtype=np.float32)
    with io.BytesIO() as buf:
        np.save(buf, arr, allow_pickle=False)
        return buf.getvalue()


def deserialize_embedding(blob_bytes: bytes) -> np.ndarray:
    with io.BytesIO(blob_bytes) as buf:
        arr = np.load(buf, allow_pickle=False)
    return np.asarray(arr, dtype=np.float32)


def add_identity(
    conn: sqlite3.Connection,
    embedding: np.ndarray,
    sample_path: Optional[str],
    ts: str,
) -> int:
    blob = serialize_embedding(embedding)
    cursor = conn.execute(
        """
        INSERT INTO identities (embedding, first_seen, last_seen, sample_path)
        VALUES (?, ?, ?, ?);
        """,
        (blob, ts, ts, sample_path),
    )
    conn.commit()
    return int(cursor.lastrowid)


def update_last_seen(conn: sqlite3.Connection, identity_id: int, ts: str) -> None:
    conn.execute(
        "UPDATE identities SET last_seen = ? WHERE id = ?;",
        (ts, identity_id),
    )
    conn.commit()


def update_sample_path(
    conn: sqlite3.Connection, identity_id: int, sample_path: Optional[str]
) -> None:
    conn.execute(
        "UPDATE identities SET sample_path = ? WHERE id = ?;",
        (sample_path, identity_id),
    )
    conn.commit()


def load_all_embeddings(conn: sqlite3.Connection) -> List[Dict[str, object]]:
    rows = conn.execute(
        "SELECT id, embedding, first_seen, last_seen, sample_path FROM identities;"
    ).fetchall()
    result: List[Dict[str, object]] = []
    for row in rows:
        result.append(
            {
                "id": int(row["id"]),
                "embedding": deserialize_embedding(row["embedding"]),
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "sample_path": row["sample_path"],
            }
        )
    return result


def list_identities(conn: sqlite3.Connection) -> List[Dict[str, object]]:
    rows = conn.execute(
        "SELECT id, first_seen, last_seen, sample_path FROM identities;"
    ).fetchall()
    return [dict(row) for row in rows]


def get_identity(conn: sqlite3.Connection, identity_id: int) -> Optional[Dict[str, object]]:
    row = conn.execute(
        "SELECT id, first_seen, last_seen, sample_path FROM identities WHERE id = ?;",
        (identity_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def find_best_match(
    conn: sqlite3.Connection, embedding: np.ndarray
) -> Tuple[Optional[int], float]:
    rows = load_all_embeddings(conn)
    if not rows:
        return None, -1.0

    matrix = np.vstack([np.asarray(r["embedding"], dtype=np.float32) for r in rows])
    query = np.asarray(embedding, dtype=np.float32)
    similarities = cosine_similarity_batch(query, matrix)
    if similarities.size == 0:
        return None, -1.0

    best_idx = int(np.argmax(similarities))
    best_id = int(rows[best_idx]["id"])
    best_similarity = float(similarities[best_idx])
    return best_id, best_similarity
