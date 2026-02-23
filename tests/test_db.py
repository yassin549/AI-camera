from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db


def _unit_norm(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def test_db_persistence_and_matching() -> None:
    conn = None
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        conn = db.get_connection(db_path)
        db.init_db(conn)

        rng = np.random.default_rng(7)
        emb = _unit_norm(rng.random(128, dtype=np.float32))
        ts = "2026-02-21T00:00:00+00:00"
        identity_id = db.add_identity(conn, emb, sample_path=None, ts=ts)
        assert identity_id == "1"

        rows = db.load_all_embeddings(conn)
        assert len(rows) == 1
        assert rows[0]["id"] == identity_id
        np.testing.assert_allclose(rows[0]["embedding"], emb, rtol=1e-5, atol=1e-6)

        noisy = _unit_norm(emb + rng.normal(0.0, 0.001, size=128).astype(np.float32))
        best_id, sim = db.find_best_match(conn, noisy)
        assert best_id == identity_id
        assert sim > 0.95

        ts2 = "2026-02-21T00:00:10+00:00"
        db.update_last_seen(conn, identity_id, ts2)
        identity = db.get_identity(conn, identity_id)
        assert identity is not None
        assert identity["last_seen"] == ts2

        success, _ = db.reassign_identity_id(conn, identity_id, "yassin")
        assert success is True
        assert db.get_identity(conn, identity_id) is None
        moved = db.get_identity(conn, "yassin")
        assert moved is not None
        assert moved["last_seen"] == ts2

        ts3 = "2026-02-21T00:00:20+00:00"
        next_identity_id = db.add_identity(conn, emb, sample_path=None, ts=ts3)
        assert next_identity_id == "2"
    finally:
        if conn is not None:
            conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_init_db_migrates_integer_ids_to_text_ids() -> None:
    conn = None
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        conn = db.get_connection(db_path)
        conn.execute(
            """
            CREATE TABLE identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL,
                first_seen TEXT,
                last_seen TEXT,
                sample_path TEXT
            );
            """
        )
        emb_blob = db.serialize_embedding(np.ones(128, dtype=np.float32))
        conn.execute(
            """
            INSERT INTO identities (embedding, first_seen, last_seen, sample_path)
            VALUES (?, ?, ?, ?);
            """,
            (emb_blob, "2026-02-21T00:00:00+00:00", "2026-02-21T00:00:00+00:00", None),
        )
        conn.commit()

        db.init_db(conn)

        table_info = conn.execute("PRAGMA table_info(identities);").fetchall()
        id_row = next(r for r in table_info if str(r["name"]).lower() == "id")
        assert str(id_row["type"]).upper() == "TEXT"

        rows = db.list_identities(conn)
        assert len(rows) == 1
        assert rows[0]["id"] == "1"
    finally:
        if conn is not None:
            conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)
