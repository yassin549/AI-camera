from __future__ import annotations

from pathlib import Path

import numpy as np

import db


def _unit_basis(size: int, index: int) -> np.ndarray:
    vec = np.zeros((size,), dtype=np.float32)
    vec[int(index)] = 1.0
    return vec


def test_db_add_identity_and_match_face_body(tmp_path: Path) -> None:
    db_path = tmp_path / "db_match_test.sqlite"
    db.configure(str(db_path), ephemeral=True)
    db.load_all_embeddings()

    face = _unit_basis(128, 3)
    body = _unit_basis(256, 9)
    identity_id = db.add_identity(face, body, None, None, db.now_iso())

    best_face_id, face_score = db.find_best_face(face)
    best_body_id, body_score = db.find_best_body(body)

    assert best_face_id == identity_id
    assert best_body_id == identity_id
    assert face_score > 0.99
    assert body_score > 0.99


def test_db_muted_flag_roundtrip_and_identity_existence(tmp_path: Path) -> None:
    db_path = tmp_path / "db_mute_roundtrip.sqlite"
    db.configure(str(db_path), ephemeral=False)
    db.load_all_embeddings()

    ident_id = db.add_identity(_unit_basis(128, 0), None, None, None, db.now_iso())
    assert db.identity_exists(int(ident_id)) is True
    assert db.is_identity_muted(int(ident_id)) is False

    assert db.set_identity_muted(int(ident_id), True) is True
    assert db.is_identity_muted(int(ident_id)) is True

    listed = db.list_identities()
    row = next((entry for entry in listed if int(entry["id"]) == int(ident_id)), None)
    assert row is not None
    assert bool(row["is_muted"]) is True
