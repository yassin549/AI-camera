"""AIcam multithreaded orchestrator for smooth CPU-only live tracking and ID."""

from __future__ import annotations

import argparse
import csv
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import yaml

import db
from api_server import ApiRuntimeState, run_api_server
from capture import CaptureWorker
from detector_yolo import Detection, YOLOv8ONNXDetector
from recognition_worker import RecognitionWorker
from tracker_adapter import Track, TrackerAdapter
from utils import (
    CounterFPS,
    FrameTimingStore,
    LatestFrameStore,
    SharedTrackStore,
    StageStats,
)

LOGGER = logging.getLogger("aicam")


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "RTSP_URL" in cfg and "rtsp_url" not in cfg:
        cfg["rtsp_url"] = cfg["RTSP_URL"]

    # Runtime defaults.
    cfg.setdefault("imgsz", [640, 360])
    cfg.setdefault("detection_interval", 3)
    cfg.setdefault("face_interval_frames", 10)
    cfg.setdefault("body_interval_frames", 10)
    cfg.setdefault("face_threshold", 0.60)
    cfg.setdefault("body_threshold", 0.40)
    cfg.setdefault("cache_seconds", 30)
    cfg.setdefault("max_tracks", 20)
    cfg.setdefault("http_port", 8080)
    cfg.setdefault("http_host", "0.0.0.0")
    cfg.setdefault("display_window", True)
    cfg.setdefault("debug", False)
    cfg.setdefault("debug_csv", "./debug/perf.csv")
    cfg.setdefault("disable_body_embedding", True)

    # Model defaults.
    cfg.setdefault("yolo_onnx_path", "./models/yolov8n_person.onnx")
    cfg.setdefault("person_conf_threshold", 0.50)
    cfg.setdefault("person_iou_threshold", 0.50)
    cfg.setdefault("tracker_match_threshold", 0.80)

    return cfg


def _render_overlay(frame, tracks: List[object]) -> None:
    for st in tracks:
        x1, y1, x2, y2 = [int(v) for v in st.bbox]
        ident = st.identity_id
        modality = st.modality
        score = st.identity_score

        color = (30, 220, 30) if ident is not None else (0, 190, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"T{st.track_id} I{ident if ident is not None else '-'} {modality}:{score:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def _log_stage_window(
    stage_stats: StageStats,
    fps_detection: CounterFPS,
    fps_tracking: CounterFPS,
    fps_recognition: CounterFPS,
    fps_render: CounterFPS,
    window_sec: float,
) -> None:
    d_cnt, d_avg, d_p95 = stage_stats.summary("detection_ms")
    t_cnt, t_avg, t_p95 = stage_stats.summary("tracking_ms")
    r_cnt, r_avg, r_p95 = stage_stats.summary("recognition_ms")
    v_cnt, v_avg, v_p95 = stage_stats.summary("render_ms")

    det_fps = fps_detection.pop() / max(window_sec, 1e-6)
    trk_fps = fps_tracking.pop() / max(window_sec, 1e-6)
    rec_fps = fps_recognition.pop() / max(window_sec, 1e-6)
    rnd_fps = fps_render.pop() / max(window_sec, 1e-6)

    LOGGER.info(
        "FPS(det/track/rec/render)=%.1f/%.1f/%.1f/%.1f | "
        "ms avg,p95 det=%.1f,%.1f track=%.1f,%.1f rec=%.1f,%.1f render=%.1f,%.1f | counts d=%d t=%d r=%d v=%d",
        det_fps,
        trk_fps,
        rec_fps,
        rnd_fps,
        d_avg,
        d_p95,
        t_avg,
        t_p95,
        r_avg,
        r_p95,
        v_avg,
        v_p95,
        d_cnt,
        t_cnt,
        r_cnt,
        v_cnt,
    )


def run_pipeline(
    cfg: Dict[str, object],
    *,
    start_api: bool,
    benchmark: bool,
    benchmark_seconds: int,
    display_window: bool,
) -> None:
    db.configure(str(cfg.get("db_path", "./identities.db")), ephemeral=bool(cfg.get("ephemeral_mode", False)))
    db.init_db()
    db.load_all_embeddings()

    frame_store = LatestFrameStore()
    track_store = SharedTrackStore()
    timing_store = FrameTimingStore()
    stage_stats = StageStats(window=4096)

    fps_detection = CounterFPS()
    fps_tracking = CounterFPS()
    fps_recognition = CounterFPS()
    fps_render = CounterFPS()

    stop_event = threading.Event()

    imgsz = tuple(cfg.get("imgsz", [640, 360]))

    detector = YOLOv8ONNXDetector(
        model_path=str(cfg["yolo_onnx_path"]),
        imgsz=imgsz,
        conf_threshold=float(cfg.get("person_conf_threshold", 0.50)),
        iou_threshold=float(cfg.get("person_iou_threshold", 0.50)),
        feature_output_name=cfg.get("yolo_feature_output_name"),
        disable_body_embedding=bool(cfg.get("disable_body_embedding", True)),
    )

    tracker = TrackerAdapter(
        track_thresh=float(cfg.get("person_conf_threshold", 0.50)),
        match_thresh=float(cfg.get("tracker_match_threshold", 0.80)),
        max_tracks=int(cfg.get("max_tracks", 20)),
    )

    capture = CaptureWorker(
        source=str(cfg.get("rtsp_url", "")),
        imgsz=imgsz,
        frame_store=frame_store,
        stop_event=stop_event,
    )

    detection_interval = max(1, int(cfg.get("detection_interval", 3)))
    max_tracks = int(cfg.get("max_tracks", 20))

    def detection_loop() -> None:
        last_frame_id = -1
        while not stop_event.is_set():
            packet = frame_store.wait_for_new(last_frame_id, timeout=0.1)
            if packet is None:
                continue

            frame_id = int(packet.frame_id)
            frame = packet.frame
            last_frame_id = frame_id

            tracks: List[Track]
            if frame_id % detection_interval == 0:
                t0 = time.perf_counter()
                dets = detector.detect(frame)
                det_ms = (time.perf_counter() - t0) * 1000.0
                timing_store.set(frame_id, "detection_ms", det_ms)
                stage_stats.add("detection_ms", det_ms)
                fps_detection.inc(1)

                tr_in = [Track(track_id=-1, bbox=d.bbox, score=d.score, feature=d.feature) for d in dets]
                t1 = time.perf_counter()
                tracks = tracker.update(tr_in)
                tr_ms = (time.perf_counter() - t1) * 1000.0
            else:
                timing_store.set(frame_id, "detection_ms", 0.0)
                t1 = time.perf_counter()
                tracks = tracker.predict()
                tr_ms = (time.perf_counter() - t1) * 1000.0

            timing_store.set(frame_id, "tracking_ms", tr_ms)
            stage_stats.add("tracking_ms", tr_ms)
            fps_tracking.inc(1)

            track_store.update_from_tracker(tracks, frame_id=frame_id, max_tracks=max_tracks)

    api_state = ApiRuntimeState(
        frame_store=frame_store,
        db_path=str(cfg.get("db_path", "./identities.db")),
        media_root=str(cfg.get("media_root", ".")),
    )
    recognition = RecognitionWorker(
        frame_store=frame_store,
        track_store=track_store,
        cfg=cfg,
        timing_store=timing_store,
        stage_stats=stage_stats,
        fps_counter=fps_recognition,
        stop_event=stop_event,
    )

    capture.start()
    detection_thread = threading.Thread(target=detection_loop, daemon=True, name="det-track-worker")
    detection_thread.start()
    recognition.start()

    if start_api:
        api_thread = threading.Thread(
            target=run_api_server,
            kwargs={
                "runtime": api_state,
                "host": str(cfg.get("http_host", "127.0.0.1")),
                "port": int(cfg.get("http_port", 8080)),
                "log_level": "info" if bool(cfg.get("debug", False)) else "warning",
            },
            daemon=True,
            name="api-worker",
        )
        api_thread.start()
        LOGGER.info("API started on http://%s:%s", cfg.get("http_host", "127.0.0.1"), cfg.get("http_port", 8080))

    debug_writer = None
    debug_file = None
    if bool(cfg.get("debug", False)):
        debug_path = Path(str(cfg.get("debug_csv", "./debug/perf.csv")))
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_file = debug_path.open("w", newline="", encoding="utf-8")
        debug_writer = csv.writer(debug_file)
        debug_writer.writerow(["frame_id", "detection_ms", "tracking_ms", "recognition_ms", "render_ms"])

    started_at = time.time()
    next_log_at = started_at + 10.0
    last_rendered = -1

    try:
        while not stop_event.is_set():
            packet = frame_store.wait_for_new(last_rendered, timeout=0.05)
            now = time.time()

            if benchmark and now - started_at >= float(benchmark_seconds):
                break

            if packet is None:
                if now >= next_log_at:
                    _log_stage_window(
                        stage_stats,
                        fps_detection,
                        fps_tracking,
                        fps_recognition,
                        fps_render,
                        window_sec=10.0,
                    )
                    next_log_at = now + 10.0
                continue

            frame_id = int(packet.frame_id)
            frame = packet.frame.copy()
            last_rendered = frame_id

            t0 = time.perf_counter()
            track_states = track_store.snapshot()
            _render_overlay(frame, track_states)

            api_state.publish_tracks(
                track_store.to_api_payload(),
                frame_id=frame_id,
                frame_shape=frame.shape,
            )

            if display_window and not benchmark:
                cv2.imshow("AIcam", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            render_ms = (time.perf_counter() - t0) * 1000.0
            timing_store.set(frame_id, "render_ms", render_ms)
            stage_stats.add("render_ms", render_ms)
            fps_render.inc(1)

            if debug_writer is not None:
                row = timing_store.get(frame_id)
                debug_writer.writerow(
                    [
                        frame_id,
                        f"{row.get('detection_ms', 0.0):.3f}",
                        f"{row.get('tracking_ms', 0.0):.3f}",
                        f"{row.get('recognition_ms', 0.0):.3f}",
                        f"{row.get('render_ms', 0.0):.3f}",
                    ]
                )

            if now >= next_log_at:
                _log_stage_window(
                    stage_stats,
                    fps_detection,
                    fps_tracking,
                    fps_recognition,
                    fps_render,
                    window_sec=10.0,
                )
                next_log_at = now + 10.0
                timing_store.cleanup_older_than(frame_id - 600)
    finally:
        stop_event.set()
        recognition.stop()
        capture.stop()
        detection_thread.join(timeout=2.0)
        if debug_file is not None:
            debug_file.close()
        cv2.destroyAllWindows()

    elapsed = max(1e-6, time.time() - started_at)
    det_count, det_avg, det_p95 = stage_stats.summary("detection_ms")
    trk_count, trk_avg, trk_p95 = stage_stats.summary("tracking_ms")
    rec_count, rec_avg, rec_p95 = stage_stats.summary("recognition_ms")
    rnd_count, rnd_avg, rnd_p95 = stage_stats.summary("render_ms")

    if benchmark:
        LOGGER.info(
            "Benchmark %.1fs | det_fps=%.2f track_fps=%.2f rec_fps=%.2f render_fps=%.2f",
            elapsed,
            det_count / elapsed,
            trk_count / elapsed,
            rec_count / elapsed,
            rnd_count / elapsed,
        )
        LOGGER.info(
            "Stage ms avg,p95 | det=%.2f,%.2f track=%.2f,%.2f rec=%.2f,%.2f render=%.2f,%.2f",
            det_avg,
            det_p95,
            trk_avg,
            trk_p95,
            rec_avg,
            rec_p95,
            rnd_avg,
            rnd_p95,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIcam threaded CPU pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--start-api", action="store_true", help="Start API server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging + perf CSV")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    parser.add_argument("--benchmark-seconds", type=int, default=60)
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.debug:
        cfg["debug"] = True

    setup_logging(bool(cfg.get("debug", False)))

    run_pipeline(
        cfg,
        start_api=bool(args.start_api),
        benchmark=bool(args.benchmark),
        benchmark_seconds=int(args.benchmark_seconds),
        display_window=not bool(args.no_display) and bool(cfg.get("display_window", True)),
    )


if __name__ == "__main__":
    main()
