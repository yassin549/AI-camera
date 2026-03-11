from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import db
from api_server import ApiRuntimeState, create_app
from tracker_adapter import Track
from utils import LatestFrameStore, SharedTrackStore


def _setup_runtime(tmp_path: Path) -> tuple[ApiRuntimeState, LatestFrameStore, SharedTrackStore, Path]:
    db_path = tmp_path / "identities_actions.db"
    media_root = tmp_path / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    db.configure(str(db_path), ephemeral=False)
    db.init_db()
    db.load_all_embeddings()

    frame_store = LatestFrameStore()
    track_store = SharedTrackStore(max_age_frames=24)
    runtime = ApiRuntimeState(
        frame_store=frame_store,
        db_path=str(db_path),
        media_root=str(media_root),
        track_store=track_store,
    )
    return runtime, frame_store, track_store, db_path


def test_snapshot_and_mute_endpoints(tmp_path: Path) -> None:
    runtime, frame_store, _track_store, db_path = _setup_runtime(tmp_path)
    try:
        identity_id = db.add_identity(
            face_emb=None,
            body_emb=np.ones((256,), dtype=np.float32),
            face_sample_path=None,
            body_sample_path=None,
            ts=db.now_iso(),
        )
        frame = np.full((120, 160, 3), 127, dtype=np.uint8)
        frame_store.publish(frame_id=1, frame=frame, ts=time.time())

        app = create_app(runtime)
        with TestClient(app) as client:
            snapshot = client.post(f"/api/identities/{identity_id}/snapshot", json={})
            assert snapshot.status_code == 200
            snapshot_payload = snapshot.json()
            assert snapshot_payload["ok"] is True
            sample_url = str(snapshot_payload["sample"])
            assert sample_url.startswith("/media/")
            rel = sample_url.replace("/media/", "", 1)
            assert (Path(runtime.media_root) / rel).exists()

            mute_toggle = client.post(f"/api/identities/{identity_id}/mute", json={})
            assert mute_toggle.status_code == 200
            assert mute_toggle.json()["muted"] is True

            unmute = client.post(f"/api/identities/{identity_id}/mute", json={"muted": False})
            assert unmute.status_code == 200
            assert unmute.json()["muted"] is False

            detail = client.get(f"/api/identities/{identity_id}")
            assert detail.status_code == 200
            assert detail.json()["is_muted"] is False

        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT is_muted FROM identities WHERE id=?", (identity_id,)).fetchone()
            assert row is not None
            assert int(row[0]) == 0
    finally:
        runtime.close()


def test_merge_endpoint_moves_identity_samples(tmp_path: Path) -> None:
    runtime, _frame_store, track_store, db_path = _setup_runtime(tmp_path)
    try:
        source_id = db.add_identity(
            face_emb=np.ones((128,), dtype=np.float32),
            body_emb=None,
            face_sample_path="faces/source_face.jpg",
            body_sample_path=None,
            ts="2026-03-01T00:00:00Z",
        )
        track_store.update_from_tracker(
            [Track(track_id=11, bbox=(10, 10, 90, 140), score=0.9, feature=None)],
            frame_id=1,
            max_tracks=20,
            max_age_frames=24,
        )
        track_store.assign_identity(
            track_id=11,
            identity_id=int(source_id),
            modality="face",
            score=0.95,
            frame_id=1,
            cache_seconds=30,
        )
        target_id = db.add_identity(
            face_emb=None,
            body_emb=np.ones((256,), dtype=np.float32),
            face_sample_path=None,
            body_sample_path="body/target_body.jpg",
            ts="2026-03-01T00:00:05Z",
        )

        with sqlite3.connect(db_path) as conn:
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
            conn.execute(
                "INSERT OR IGNORE INTO identity_samples(identity_id, sample_path, sample_type, created_ts) VALUES (?, ?, ?, ?)",
                (source_id, "snapshots/source_a.jpg", "snapshot", "2026-03-01T00:00:01Z"),
            )
            conn.commit()

        app = create_app(runtime)
        with TestClient(app) as client:
            merge = client.post(f"/api/identities/{source_id}/merge", json={"target_id": target_id})
            assert merge.status_code == 200
            payload = merge.json()
            assert payload["ok"] is True
            assert int(payload["merged_into"]) == int(target_id)

        track_states = track_store.snapshot()
        assert track_states
        assert track_states[0].identity_id == int(target_id)

        with sqlite3.connect(db_path) as conn:
            source_row = conn.execute("SELECT id FROM identities WHERE id=?", (source_id,)).fetchone()
            target_row = conn.execute("SELECT id FROM identities WHERE id=?", (target_id,)).fetchone()
            assert source_row is None
            assert target_row is not None
            moved = conn.execute(
                "SELECT sample_path FROM identity_samples WHERE identity_id=?",
                (target_id,),
            ).fetchall()
            assert any(str(row[0]) == "snapshots/source_a.jpg" for row in moved)
    finally:
        runtime.close()


def test_delete_endpoint_clears_live_track_and_index(tmp_path: Path) -> None:
    runtime, _frame_store, track_store, _db_path = _setup_runtime(tmp_path)
    try:
        identity_id = db.add_identity(
            face_emb=np.ones((128,), dtype=np.float32),
            body_emb=None,
            face_sample_path=None,
            body_sample_path=None,
            ts=db.now_iso(),
        )
        track_store.update_from_tracker(
            [Track(track_id=21, bbox=(20, 20, 80, 120), score=0.9, feature=None)],
            frame_id=1,
            max_tracks=20,
            max_age_frames=24,
        )
        track_store.assign_identity(
            track_id=21,
            identity_id=int(identity_id),
            modality="face",
            score=0.91,
            frame_id=1,
            cache_seconds=30,
        )

        app = create_app(runtime)
        with TestClient(app) as client:
            response = client.delete(f"/api/identities/{identity_id}")
            assert response.status_code == 204

        states = track_store.snapshot()
        assert states
        assert states[0].identity_id is None
        assert db.identity_exists(int(identity_id)) is False
    finally:
        runtime.close()


def test_manual_track_assignment_endpoint_links_unresolved_track(tmp_path: Path) -> None:
    runtime, _frame_store, track_store, _db_path = _setup_runtime(tmp_path)
    try:
        identity_id = db.add_identity(
            face_emb=np.ones((128,), dtype=np.float32),
            body_emb=None,
            face_sample_path=None,
            body_sample_path=None,
            ts=db.now_iso(),
        )
        track_store.update_from_tracker(
            [Track(track_id=31, bbox=(20, 20, 90, 150), score=0.92, feature=None)],
            frame_id=3,
            max_tracks=20,
            max_age_frames=24,
        )

        app = create_app(runtime)
        with TestClient(app) as client:
            response = client.post(
                "/api/tracks/31/assign",
                json={"identity_id": int(identity_id), "cache_seconds": 45},
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["ok"] is True
            assert int(payload["track_id"]) == 31
            assert int(payload["identity_id"]) == int(identity_id)

        states = track_store.snapshot()
        track_state = next((state for state in states if int(state.track_id) == 31), None)
        assert track_state is not None
        assert int(track_state.identity_id or -1) == int(identity_id)
        assert str(track_state.modality) == "manual"
    finally:
        runtime.close()
