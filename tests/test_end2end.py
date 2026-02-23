from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import db
from main import RuntimeContext, TrackCache, _process_identity_for_track
from tracker_adapter import Track
from embedder_face import FaceEmbedder
from utils import cosine, save_np_to_blob, load_np_from_blob


class _FaceDetectorSequence:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.idx = 0

    def detect(self, _rgb):
        if self.idx >= len(self.sequence):
            return []
        value = self.sequence[self.idx]
        self.idx += 1
        return value

    def close(self):
        return None


class _FaceEmbedderConstant:
    def __init__(self, emb: np.ndarray):
        self.emb = emb.astype(np.float32)

    def embed_from_bgr(self, _bgr):
        return self.emb


class _Dummy:
    pass


def _base_cfg(tmp_path: Path):
    return {
        "face_interval_frames": 1,
        "body_interval_frames": 1,
        "cache_seconds": 0,
        "face_threshold": 0.60,
        "face_threshold_high": 0.72,
        "body_threshold": 0.40,
        "face_top_ratio": 0.50,
        "persist_body_only": False,
        "faces_dir": str(tmp_path / "faces"),
    }


def _make_ctx(tmp_path: Path, cfg_overrides=None, face_detector=None, face_embedder=None):
    cfg = _base_cfg(tmp_path)
    if cfg_overrides:
        cfg.update(cfg_overrides)

    db_path = tmp_path / "identities_test.db"
    db.configure(str(db_path), ephemeral=False)
    db.init_db()
    db.load_all_embeddings()

    ctx = RuntimeContext(
        config=cfg,
        detector=_Dummy(),
        tracker=_Dummy(),
        face_detector=face_detector or _FaceDetectorSequence([[]]),
        face_embedder=face_embedder or _FaceEmbedderConstant(np.ones(128, dtype=np.float32) / np.sqrt(128.0)),
        api_state=_Dummy(),
        track_cache={},
    )
    return ctx


def test_multiple_faces_produce_different_embeddings():
    embedder = FaceEmbedder()

    img_a = np.full((96, 96, 3), (20, 40, 220), dtype=np.uint8)
    img_b = np.full((96, 96, 3), (220, 40, 20), dtype=np.uint8)

    emb_a = embedder.embed_from_bgr(img_a)
    emb_b = embedder.embed_from_bgr(img_b)

    assert emb_a is not None
    assert emb_b is not None
    assert emb_a.shape[0] == 128
    assert emb_b.shape[0] == 128
    assert cosine(emb_a, emb_b) < 0.999


def test_track_no_face_then_face_links_existing_identity(tmp_path: Path):
    face_vec = np.zeros((128,), dtype=np.float32)
    face_vec[0] = 1.0
    body_vec = np.zeros((256,), dtype=np.float32)
    body_vec[1] = 1.0

    ctx = _make_ctx(
        tmp_path,
        face_detector=_FaceDetectorSequence([[], [(5, 5, 40, 40, 0.95)]]),
        face_embedder=_FaceEmbedderConstant(face_vec),
    )

    existing_id = db.add_identity(face_vec, body_vec, None, None, "2026-02-22T00:00:00+00:00")
    db.load_all_embeddings()

    frame = np.full((180, 120, 3), 127, dtype=np.uint8)
    track = Track(track_id=1, bbox=(10, 10, 100, 170), score=0.9, feature=body_vec)

    first = _process_identity_for_track(ctx, frame, track, frame_idx=0)
    second = _process_identity_for_track(ctx, frame, track, frame_idx=2)

    assert first["identity_id"] in (None, existing_id)
    assert second["identity_id"] == existing_id
    assert second["modality"] == "face"


def test_short_video_pipeline_insert_then_recognize(tmp_path: Path):
    video_path = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 5.0, (160, 120))
    for _ in range(3):
        frame = np.full((120, 160, 3), 160, dtype=np.uint8)
        cv2.rectangle(frame, (20, 10), (110, 110), (0, 255, 0), 2)
        writer.write(frame)
    writer.release()

    face_vec = np.zeros((128,), dtype=np.float32)
    face_vec[5] = 1.0
    body_vec = np.zeros((256,), dtype=np.float32)
    body_vec[7] = 1.0

    ctx = _make_ctx(
        tmp_path,
        face_detector=_FaceDetectorSequence([[(4, 4, 32, 32, 0.9)], [(4, 4, 32, 32, 0.9)], [(4, 4, 32, 32, 0.9)]]),
        face_embedder=_FaceEmbedderConstant(face_vec),
    )

    cap = cv2.VideoCapture(str(video_path))
    ids = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        track_id = 1 if frame_idx == 0 else 2
        track = Track(track_id=track_id, bbox=(20, 10, 110, 110), score=0.9, feature=body_vec)
        result = _process_identity_for_track(ctx, frame, track, frame_idx=frame_idx)
        ids.append(result["identity_id"])
        frame_idx += 1
    cap.release()

    assert ids[0] is not None
    assert ids[1] == ids[0]

    # BLOB roundtrip validation
    blob = save_np_to_blob(face_vec)
    restored = load_np_from_blob(blob)
    assert restored.dtype == np.float32
    assert restored.shape == face_vec.shape
    assert np.allclose(restored, face_vec)

    db.load_all_embeddings()
    best_id, best_score = db.find_best_face(face_vec)
    assert best_id == ids[0]
    assert best_score >= 0.99
