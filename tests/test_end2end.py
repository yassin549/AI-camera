from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

import recognition_worker
from tracker_adapter import Track
from utils import CounterFPS, FrameTimingStore, LatestFrameStore, SharedTrackStore, StageStats


class _DummyFaceDetector:
    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = float(min_confidence)
        self.available = True

    def detect(self, _rgb):
        return []

    def close(self) -> None:
        return None


class _DummyFaceEmbedder:
    def __init__(self, **_: object) -> None:
        pass

    def embed_from_bgr(self, _bgr):
        return None


def _build_worker(monkeypatch, tmp_path: Path, cfg_overrides: dict[str, object] | None = None):
    monkeypatch.setattr(recognition_worker, "FaceROIDetector", _DummyFaceDetector)
    monkeypatch.setattr(recognition_worker, "FaceEmbedder", _DummyFaceEmbedder)

    frame_store = LatestFrameStore()
    track_store = SharedTrackStore(max_age_frames=24)
    stop_event = threading.Event()

    cfg: dict[str, object] = {
        "target_fps": 30.0,
        "face_interval_frames": 1,
        "body_interval_frames": 10,
        "face_threshold": 0.6,
        "body_threshold": 0.4,
        "cache_seconds": 0.0,
        "persist_body_only": False,
        "disable_body_embedding": True,
        "faces_dir": str(tmp_path / "faces"),
    }
    if cfg_overrides:
        cfg.update(cfg_overrides)

    worker = recognition_worker.RecognitionWorker(
        frame_store=frame_store,
        track_store=track_store,
        cfg=cfg,
        timing_store=FrameTimingStore(),
        stage_stats=StageStats(),
        fps_counter=CounterFPS(),
        stop_event=stop_event,
        metrics=None,
    )
    return worker, frame_store, track_store, stop_event


def _wait_until(predicate, timeout_sec: float = 1.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_recognition_does_not_drop_tracks_only_by_frame_gap(monkeypatch, tmp_path: Path) -> None:
    worker, frame_store, track_store, stop_event = _build_worker(
        monkeypatch,
        tmp_path,
        cfg_overrides={
            "recognition_stale_age_ratio": 1.0,
            "recognition_stale_seconds": 0.5,
        },
    )
    calls: list[int] = []

    def _fake_extract(*_args, **_kwargs):
        calls.append(1)
        stop_event.set()
        return None

    worker._extract_face_embedding = _fake_extract  # type: ignore[assignment]
    track_store.update_from_tracker(
        [Track(track_id=1, bbox=(10, 10, 100, 160), score=0.9, feature=None)],
        frame_id=1,
        max_tracks=20,
        max_age_frames=24,
    )

    thread = threading.Thread(target=worker._run, daemon=True)
    thread.start()
    frame_store.publish(frame_id=5, frame=np.zeros((180, 120, 3), dtype=np.uint8), ts=time.time())

    assert _wait_until(lambda: len(calls) == 1, timeout_sec=1.2)
    stop_event.set()
    thread.join(timeout=1.0)
    assert len(calls) == 1


def test_recognition_suppresses_stale_tracks_by_time_and_age(monkeypatch, tmp_path: Path) -> None:
    worker, frame_store, track_store, stop_event = _build_worker(
        monkeypatch,
        tmp_path,
        cfg_overrides={
            "recognition_stale_age_ratio": 0.2,
            "recognition_stale_seconds": 0.05,
        },
    )
    calls: list[int] = []

    def _fake_extract(*_args, **_kwargs):
        calls.append(1)
        return None

    worker._extract_face_embedding = _fake_extract  # type: ignore[assignment]
    track_store.update_from_tracker(
        [Track(track_id=4, bbox=(20, 15, 110, 170), score=0.92, feature=None)],
        frame_id=1,
        max_tracks=20,
        max_age_frames=24,
    )

    thread = threading.Thread(target=worker._run, daemon=True)
    thread.start()

    frame_store.publish(frame_id=8, frame=np.zeros((180, 120, 3), dtype=np.uint8), ts=time.time())
    assert _wait_until(lambda: len(calls) >= 1, timeout_sec=1.2)
    first_calls = len(calls)
    assert first_calls == 1

    time.sleep(0.08)
    frame_store.publish(frame_id=9, frame=np.zeros((180, 120, 3), dtype=np.uint8), ts=time.time())
    time.sleep(0.15)

    stop_event.set()
    thread.join(timeout=1.0)
    assert len(calls) == first_calls
