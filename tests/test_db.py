from __future__ import annotations

import os
import tempfile

import numpy as np

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

        rows = db.load_all_embeddings(conn)
        assert len(rows) == 1
        assert int(rows[0]["id"]) == identity_id
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
    finally:
        if conn is not None:
            conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)
