from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)


def test_reid_merge_body_identity_then_face_identity() -> None:
    conn = None
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        conn = db.get_connection(db_path)
        db.init_db(conn)

        rng = np.random.default_rng(123)

        # A starts as body-only identity
        body_a = _unit(rng.normal(0.0, 1.0, size=256).astype(np.float32))
        id_a = db.create_identity(
            conn,
            ts="2026-02-22T10:00:00+00:00",
            body_embedding=body_a,
            body_sample="body_a.jpg",
        )

        # B starts as face identity
        face_b = _unit(rng.normal(0.0, 1.0, size=128).astype(np.float32))
        id_b = db.create_identity(
            conn,
            ts="2026-02-22T10:00:05+00:00",
            face_embedding=face_b,
            face_sample="face_b.jpg",
        )

        # Later B also gets strong body support close to A
        body_b = _unit(0.95 * body_a + 0.05 * rng.normal(0.0, 1.0, size=256).astype(np.float32))
        db.update_body_embedding(conn, id_b, body_b, alpha=0.0, ts="2026-02-22T10:00:06+00:00", score=0.9)

        body_similarity = float(np.dot(body_a, body_b))
        canonical, merged, reason = db.maybe_merge_identities(
            conn,
            body_identity_id=id_a,
            face_identity_id=id_b,
            face_similarity=0.80,
            body_similarity=body_similarity,
            face_threshold_high=0.72,
            face_threshold=0.60,
            body_threshold=0.40,
            overlap_seconds=60.0,
            ts="2026-02-22T10:00:07+00:00",
        )

        assert merged is True
        assert reason in {"auto_face_high", "auto_face_body"}
        assert canonical == id_b

        source = db.get_identity(conn, id_a)
        target = db.get_identity(conn, id_b)
        assert source is not None
        assert target is not None
        assert source["merged_into"] == id_b
        assert db.resolve_canonical_identity(conn, id_a) == id_b
    finally:
        if conn is not None:
            conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)
