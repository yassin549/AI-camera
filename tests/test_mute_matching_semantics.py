from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db


def _unit_basis(size: int, index: int) -> np.ndarray:
    vec = np.zeros((size,), dtype=np.float32)
    vec[int(index)] = 1.0
    return vec


def test_muted_face_identity_still_matches(tmp_path: Path) -> None:
    db_path = tmp_path / "mute_face_semantics.db"
    db.configure(str(db_path), ephemeral=True)
    db.load_all_embeddings()

    target_face = _unit_basis(128, 0)
    other_face = _unit_basis(128, 1)

    target_id = db.add_identity(target_face, None, None, None, db.now_iso())
    _other_id = db.add_identity(other_face, None, None, None, db.now_iso())
    assert db.set_identity_muted(target_id, True) is True

    best_id, best_score = db.find_best_face(target_face)
    assert best_id == target_id
    assert best_score > 0.99


def test_muted_body_identity_still_matches(tmp_path: Path) -> None:
    db_path = tmp_path / "mute_body_semantics.db"
    db.configure(str(db_path), ephemeral=True)
    db.load_all_embeddings()

    target_body = _unit_basis(256, 0)
    other_body = _unit_basis(256, 1)

    target_id = db.add_identity(None, target_body, None, None, db.now_iso())
    _other_id = db.add_identity(None, other_body, None, None, db.now_iso())
    assert db.set_identity_muted(target_id, True) is True

    best_id, best_score = db.find_best_body(target_body)
    assert best_id == target_id
    assert best_score > 0.99
