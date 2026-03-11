from __future__ import annotations

from pathlib import Path

import numpy as np

import db


def _unit_basis(size: int, index: int) -> np.ndarray:
    vec = np.zeros((size,), dtype=np.float32)
    vec[int(index)] = 1.0
    return vec


def test_reid_face_and_body_can_converge_on_same_identity(tmp_path: Path) -> None:
    db_path = tmp_path / "reid_semantics.sqlite"
    db.configure(str(db_path), ephemeral=True)
    db.load_all_embeddings()

    face_target = _unit_basis(128, 2)
    body_target = _unit_basis(256, 7)
    face_other = _unit_basis(128, 11)
    body_other = _unit_basis(256, 12)

    target_id = db.add_identity(face_target, body_target, None, None, db.now_iso())
    _other_id = db.add_identity(face_other, body_other, None, None, db.now_iso())

    best_face_id, best_face_score = db.find_best_face(face_target)
    best_body_id, best_body_score = db.find_best_body(body_target)

    assert best_face_id == target_id
    assert best_body_id == target_id
    assert best_face_score > 0.99
    assert best_body_score > 0.99
