"""AIcam multithreaded orchestrator for smooth CPU-only live tracking and ID."""

from __future__ import annotations

import argparse
import csv
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yaml

import db
from api_server import ApiRuntimeState, run_api_server
from capture import CaptureWorker
from detector_factory import create_person_detector
from detector_yolo import Detection
from recognition_worker import RecognitionWorker
from tracker_adapter import Track, TrackerAdapter
from utils import (
    CounterFPS,
    FrameTimingStore,
    LatestFrameStore,
    RuntimeMetrics,
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
    cfg.setdefault("imgsz", [640, 384])
    cfg.setdefault("target_fps", 30.0)
    cfg.setdefault("detection_interval", 1)
    cfg.setdefault("detection_interval_max", 6)
    cfg.setdefault("adaptive_detection_interval", True)
    cfg.setdefault("detection_duty_cycle_target", 0.60)
    cfg.setdefault("detection_duty_cycle_hysteresis", 0.12)
    cfg.setdefault("detection_duty_cycle_ema_alpha", 0.25)
    cfg.setdefault("detection_interval_cooldown_frames", 6)
    cfg.setdefault("detection_min_gap_scale", 0.35)
    cfg.setdefault("adaptive_detection_resolution", True)
    cfg.setdefault("detection_resolution_profiles", [])
    cfg.setdefault("detection_resolution_cooldown_detections", 12)
    cfg.setdefault("detection_resolution_warmup_runs", 1)
    cfg.setdefault("face_interval_frames", 4)
    cfg.setdefault("body_interval_frames", 10)
    cfg.setdefault("face_threshold", 0.60)
    cfg.setdefault("body_threshold", 0.40)
    cfg.setdefault("cache_seconds", 30)
    cfg.setdefault("max_tracks", 20)
    cfg.setdefault("max_track_age_frames", 24)
    cfg.setdefault("http_port", 8080)
    cfg.setdefault("http_host", "0.0.0.0")
    cfg.setdefault("display_window", False)
    cfg.setdefault("debug", False)
    cfg.setdefault("debug_csv", "./debug/perf.csv")
    cfg.setdefault("disable_body_embedding", False)
    cfg.setdefault("persist_body_only", True)
    cfg.setdefault("body_fallback_enabled", True)
    cfg.setdefault("body_fallback_after_face_failures", 3)
    cfg.setdefault("body_fallback_model_path", "")
    cfg.setdefault("body_fallback_input_size", [128, 256])
    cfg.setdefault("body_fallback_target_dim", 256)
    cfg.setdefault("body_fallback_mean", 0.5)
    cfg.setdefault("body_fallback_std", 0.5)
    cfg.setdefault("body_fallback_output_name", "")
    cfg.setdefault("body_fallback_output_index", 0)
    cfg.setdefault("face_top_ratio", 0.68)
    cfg.setdefault("face_min_pixels", 40)
    cfg.setdefault("face_roi_min_confidence", 0.50)
    cfg.setdefault("face_embedder_backend", "auto")
    cfg.setdefault("face_onnx_model_path", "")
    cfg.setdefault("face_onnx_input_size", [112, 112])
    cfg.setdefault("face_onnx_mean", 0.5)
    cfg.setdefault("face_onnx_std", 0.5)
    cfg.setdefault("face_onnx_output_name", "")
    cfg.setdefault("face_onnx_output_index", 0)
    cfg.setdefault("face_embedding_dim", 128)
    cfg.setdefault("recognition_persist_queue_size", 512)
    cfg.setdefault("recognition_persist_batch_size", 16)
    cfg.setdefault("recognition_media_queue_size", 256)
    cfg.setdefault("recognition_media_batch_size", 8)

    # Model defaults.
    cfg.setdefault("yolo_onnx_path", "./models/yolov8n_person.onnx")
    cfg.setdefault("yolo_onnx_fast_path", "./models/yolov8n_person_640x384_int8.onnx")
    cfg.setdefault("prefer_fast_person_model", True)
    cfg.setdefault("detector_backend", "yolo_onnx")
    cfg.setdefault("detector_model_path", "")
    cfg.setdefault("rtdetr_onnx_path", "./models/rtdetr-l_person.onnx")
    cfg.setdefault("detector_output_format", "auto")
    cfg.setdefault("detector_person_class_id", 0)
    cfg.setdefault("detector_max_detections", 300)
    cfg.setdefault("person_conf_threshold", 0.50)
    cfg.setdefault("person_iou_threshold", 0.50)
    cfg.setdefault("tracker_match_threshold", 0.80)
    cfg.setdefault("onnx_intra_threads", 0)
    cfg.setdefault("onnx_inter_threads", 1)
    cfg.setdefault("detector_warmup_runs", 2)

    if bool(cfg.get("prefer_fast_person_model", True)):
        fast_candidate = str(cfg.get("yolo_onnx_fast_path", "")).strip()
        current_model = str(cfg.get("yolo_onnx_path", "")).strip()
        if (
            fast_candidate
            and Path(fast_candidate).exists()
            and (not current_model or current_model == "./models/yolov8n_person.onnx")
        ):
            cfg["yolo_onnx_path"] = fast_candidate

    return cfg


def _coerce_resolution(raw: Any, fallback: Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            width = max(1, int(raw[0]))
            height = max(1, int(raw[1]))
            return width, height
        except Exception:
            return fallback
    return fallback


def _build_resolution_profiles(cfg: Dict[str, object], base: Tuple[int, int]) -> List[Tuple[int, int]]:
    raw_profiles = cfg.get("detection_resolution_profiles", [])
    profiles: List[Tuple[int, int]] = [base]

    if isinstance(raw_profiles, (list, tuple)):
        for raw in raw_profiles:
            parsed = _coerce_resolution(raw, base)
            profiles.append(parsed)

    if len(profiles) <= 1:
        scaled_w = int(round(base[0] * 0.8))
        scaled_h = int(round(base[1] * 0.8))
        downsampled = (
            max(320, max(1, scaled_w // 32) * 32),
            max(192, max(1, scaled_h // 32) * 32),
        )
        if downsampled != base:
            profiles.append(downsampled)

    deduped: List[Tuple[int, int]] = []
    seen = set()
    for size in profiles:
        if size in seen:
            continue
        seen.add(size)
        deduped.append(size)
    return deduped


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
) -> Dict[str, float]:
    d_cnt, d_avg, d_p50, d_p95 = stage_stats.summary("detection_ms")
    t_cnt, t_avg, t_p50, t_p95 = stage_stats.summary("tracking_ms")
    r_cnt, r_avg, r_p50, r_p95 = stage_stats.summary("recognition_ms")
    v_cnt, v_avg, v_p50, v_p95 = stage_stats.summary("render_ms")

    det_fps = fps_detection.pop() / max(window_sec, 1e-6)
    trk_fps = fps_tracking.pop() / max(window_sec, 1e-6)
    rec_fps = fps_recognition.pop() / max(window_sec, 1e-6)
    rnd_fps = fps_render.pop() / max(window_sec, 1e-6)

    LOGGER.info(
        "FPS(det/track/rec/render)=%.1f/%.1f/%.1f/%.1f | "
        "ms avg,p50,p95 det=%.1f,%.1f,%.1f track=%.1f,%.1f,%.1f rec=%.1f,%.1f,%.1f render=%.1f,%.1f,%.1f | counts d=%d t=%d r=%d v=%d",
        det_fps,
        trk_fps,
        rec_fps,
        rnd_fps,
        d_avg,
        d_p50,
        d_p95,
        t_avg,
        t_p50,
        t_p95,
        r_avg,
        r_p50,
        r_p95,
        v_avg,
        v_p50,
        v_p95,
        d_cnt,
        t_cnt,
        r_cnt,
        v_cnt,
    )
    return {
        "det_fps": float(det_fps),
        "trk_fps": float(trk_fps),
        "rec_fps": float(rec_fps),
        "rnd_fps": float(rnd_fps),
        "det_avg_ms": float(d_avg),
        "det_p50_ms": float(d_p50),
        "det_p95_ms": float(d_p95),
        "trk_avg_ms": float(t_avg),
        "trk_p50_ms": float(t_p50),
        "trk_p95_ms": float(t_p95),
        "rec_avg_ms": float(r_avg),
        "rec_p50_ms": float(r_p50),
        "rec_p95_ms": float(r_p95),
        "rnd_avg_ms": float(v_avg),
        "rnd_p50_ms": float(v_p50),
        "rnd_p95_ms": float(v_p95),
    }


def run_pipeline(
    cfg: Dict[str, object],
    *,
    start_api: bool,
    benchmark: bool,
    benchmark_seconds: int,
    display_window: bool,
) -> None:
    base_imgsz = _coerce_resolution(cfg.get("imgsz", [640, 384]), (640, 384))
    resolution_profiles = _build_resolution_profiles(cfg, base_imgsz)
    imgsz = resolution_profiles[0]
    detection_interval_cfg = max(1, int(cfg.get("detection_interval", 1)))
    detection_interval_max = max(
        detection_interval_cfg,
        int(cfg.get("detection_interval_max", max(2, detection_interval_cfg))),
    )
    adaptive_detection_interval = bool(cfg.get("adaptive_detection_interval", True))
    adaptive_detection_resolution = bool(cfg.get("adaptive_detection_resolution", True))
    detection_duty_cycle_target = min(0.95, max(0.20, float(cfg.get("detection_duty_cycle_target", 0.60))))
    detection_duty_cycle_hysteresis = min(
        0.35,
        max(0.02, float(cfg.get("detection_duty_cycle_hysteresis", 0.12))),
    )
    detection_duty_cycle_ema_alpha = min(
        1.0,
        max(0.01, float(cfg.get("detection_duty_cycle_ema_alpha", 0.25))),
    )
    detection_interval_cooldown_frames = max(1, int(cfg.get("detection_interval_cooldown_frames", 6)))
    detection_min_gap_scale = min(2.0, max(0.0, float(cfg.get("detection_min_gap_scale", 0.35))))
    detection_resolution_cooldown_detections = max(
        1,
        int(cfg.get("detection_resolution_cooldown_detections", 12)),
    )
    detection_resolution_warmup_runs = max(0, int(cfg.get("detection_resolution_warmup_runs", 1)))
    max_track_age_frames = max(1, int(cfg.get("max_track_age_frames", 24)))
    target_fps = max(1.0, float(cfg.get("target_fps", 30.0)))
    detector_warmup_runs = max(0, int(cfg.get("detector_warmup_runs", 2)))

    LOGGER.info(
        "Runtime config | detection_interval=%s adaptive_detection=%s adaptive_resolution=%s detection_interval_max=%s "
        "duty_target=%.2f duty_hysteresis=%.2f duty_ema_alpha=%.2f cooldown_frames=%s min_gap_scale=%.2f "
        "resolution_profiles=%s resolution_cooldown_detections=%s max_track_age_frames=%s "
        "face_attempt_interval=%s processing_resolution=%sx%s target_fps=%.1f",
        detection_interval_cfg,
        adaptive_detection_interval,
        adaptive_detection_resolution,
        detection_interval_max,
        detection_duty_cycle_target,
        detection_duty_cycle_hysteresis,
        detection_duty_cycle_ema_alpha,
        detection_interval_cooldown_frames,
        detection_min_gap_scale,
        ",".join([f"{w}x{h}" for w, h in resolution_profiles]),
        detection_resolution_cooldown_detections,
        max_track_age_frames,
        int(cfg.get("face_interval_frames", 4)),
        int(imgsz[0]),
        int(imgsz[1]),
        target_fps,
    )

    db.configure(str(cfg.get("db_path", "./identities.db")), ephemeral=bool(cfg.get("ephemeral_mode", False)))
    db.init_db()
    db.load_all_embeddings()

    frame_store = LatestFrameStore()
    track_store = SharedTrackStore(max_age_frames=max_track_age_frames)
    timing_store = FrameTimingStore()
    stage_stats = StageStats(window=4096)
    metrics = RuntimeMetrics()

    fps_detection = CounterFPS()
    fps_tracking = CounterFPS()
    fps_recognition = CounterFPS()
    fps_render = CounterFPS()

    stop_event = threading.Event()

    detector = create_person_detector(cfg, imgsz)
    detector_backend = str(getattr(detector, "detector_backend", str(cfg.get("detector_backend", "yolo_onnx"))))
    detector_model_path = str(getattr(detector, "model_path", str(cfg.get("yolo_onnx_path", ""))))
    LOGGER.info("Detector backend: %s | model path: %s", detector_backend, detector_model_path)
    if detector_warmup_runs > 0:
        warmup_t0 = time.perf_counter()
        detector.warmup(detector_warmup_runs, capture_sizes=resolution_profiles)
        warmup_ms = (time.perf_counter() - warmup_t0) * 1000.0
        LOGGER.info(
            "Detector warmup complete | runs_per_profile=%s profiles=%s elapsed_ms=%.1f",
            detector_warmup_runs,
            len(resolution_profiles),
            warmup_ms,
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

    detection_interval = detection_interval_cfg
    detection_interval_cooldown = 0
    active_resolution_idx = 0
    resolution_cooldown_detections = 0
    max_tracks = int(cfg.get("max_tracks", 20))
    metrics.set_gauge("detection_interval_current", float(detection_interval))
    metrics.set_gauge("detection_duty_target", float(detection_duty_cycle_target))
    metrics.set_gauge("processing_resolution_index", float(active_resolution_idx))
    metrics.set_gauge("processing_width", float(imgsz[0]))
    metrics.set_gauge("processing_height", float(imgsz[1]))

    def detection_loop() -> None:
        nonlocal detection_interval, detection_interval_cooldown, active_resolution_idx, resolution_cooldown_detections
        last_frame_id = -1
        next_detection_frame = 0
        last_frame_mono: Optional[float] = None
        frame_period_ema_sec = 1.0 / max(1.0, target_fps)
        last_detection_start_mono: Optional[float] = None
        duty_cycle_ema = float(detection_duty_cycle_target)
        duty_hi = min(0.99, detection_duty_cycle_target + detection_duty_cycle_hysteresis)
        duty_lo = max(0.01, detection_duty_cycle_target - detection_duty_cycle_hysteresis)
        supports_motion_prediction = bool(tracker.supports_motion_prediction())
        if not supports_motion_prediction:
            LOGGER.warning(
                "Tracker motion prediction unavailable; forcing per-frame detections for responsive overlay updates."
            )

        while not stop_event.is_set():
            packet = frame_store.wait_for_new(last_frame_id, timeout=0.1)
            if packet is None:
                continue

            frame_id = int(packet.frame_id)
            frame = packet.frame
            last_frame_id = frame_id

            tracks: List[Track]
            now_mono = time.monotonic()
            if last_frame_mono is not None:
                frame_delta = now_mono - last_frame_mono
                if 0.0 < frame_delta <= 1.0:
                    frame_period_ema_sec = frame_period_ema_sec * 0.90 + frame_delta * 0.10
            last_frame_mono = now_mono

            effective_detection_interval = max(1, int(detection_interval))
            if not supports_motion_prediction:
                effective_detection_interval = 1

            if frame_id >= next_detection_frame:
                detection_start_mono = time.monotonic()
                t0 = time.perf_counter()
                dets = detector.detect(frame)
                det_ms = (time.perf_counter() - t0) * 1000.0
                det_elapsed_sec = det_ms / 1000.0
                if last_detection_start_mono is None:
                    detection_cycle_sec = max(det_elapsed_sec, frame_period_ema_sec * max(1, effective_detection_interval))
                else:
                    detection_cycle_sec = max(det_elapsed_sec, detection_start_mono - last_detection_start_mono)
                last_detection_start_mono = detection_start_mono

                duty_cycle = min(1.0, det_elapsed_sec / max(1e-6, detection_cycle_sec))
                duty_cycle_ema = (
                    (1.0 - detection_duty_cycle_ema_alpha) * duty_cycle_ema
                    + detection_duty_cycle_ema_alpha * duty_cycle
                )

                timing_store.set(frame_id, "detection_ms", det_ms)
                stage_stats.add("detection_ms", det_ms)
                fps_detection.inc(1)
                metrics.inc("detector_frames", 1)
                metrics.set_gauge("detection_duty_cycle", float(duty_cycle))
                metrics.set_gauge("detection_duty_cycle_ema", float(duty_cycle_ema))
                metrics.set_gauge("detection_cycle_ms", float(detection_cycle_sec * 1000.0))
                metrics.set_gauge("frame_period_ms", float(frame_period_ema_sec * 1000.0))

                tr_in = [Track(track_id=-1, bbox=d.bbox, score=d.score, feature=d.feature) for d in dets]
                t1 = time.perf_counter()
                tracks = tracker.update(tr_in)
                tr_ms = (time.perf_counter() - t1) * 1000.0

                if adaptive_detection_interval and supports_motion_prediction:
                    if detection_interval_cooldown > 0:
                        detection_interval_cooldown -= 1
                    prev_interval = detection_interval
                    duty_metric = duty_cycle_ema

                    if detection_interval_cooldown <= 0:
                        if duty_metric > duty_hi and detection_interval < detection_interval_max:
                            detection_interval += 1
                            detection_interval_cooldown = detection_interval_cooldown_frames
                        elif duty_metric < duty_lo and detection_interval > 1:
                            detection_interval -= 1
                            detection_interval_cooldown = detection_interval_cooldown_frames
                    if detection_interval != prev_interval:
                        LOGGER.info(
                            "Adaptive detection interval adjusted %s -> %s (det_ms=%.2f duty_ema=%.3f target=%.2f cooldown=%s)",
                            prev_interval,
                            detection_interval,
                            det_ms,
                            duty_metric,
                            detection_duty_cycle_target,
                            detection_interval_cooldown,
                        )

                if adaptive_detection_resolution and len(resolution_profiles) > 1:
                    if resolution_cooldown_detections > 0:
                        resolution_cooldown_detections -= 1
                    if resolution_cooldown_detections <= 0:
                        next_resolution_idx = active_resolution_idx
                        if duty_cycle_ema > duty_hi and active_resolution_idx < (len(resolution_profiles) - 1):
                            next_resolution_idx = active_resolution_idx + 1
                        elif (
                            duty_cycle_ema < duty_lo
                            and active_resolution_idx > 0
                            and detection_interval <= detection_interval_cfg
                        ):
                            next_resolution_idx = active_resolution_idx - 1

                        if next_resolution_idx != active_resolution_idx:
                            prev_resolution = resolution_profiles[active_resolution_idx]
                            next_resolution = resolution_profiles[next_resolution_idx]
                            active_resolution_idx = next_resolution_idx
                            capture.set_imgsz(next_resolution)
                            detector.set_capture_size(next_resolution)
                            if detection_resolution_warmup_runs > 0:
                                try:
                                    detector.warmup(
                                        detection_resolution_warmup_runs,
                                        capture_sizes=[next_resolution],
                                    )
                                except Exception:
                                    LOGGER.debug("Resolution-switch warmup failed", exc_info=True)
                            resolution_cooldown_detections = detection_resolution_cooldown_detections
                            metrics.set_gauge("processing_resolution_index", float(active_resolution_idx))
                            metrics.set_gauge("processing_width", float(next_resolution[0]))
                            metrics.set_gauge("processing_height", float(next_resolution[1]))
                            api_state.set_capability(
                                "processing_resolution",
                                {"width": int(next_resolution[0]), "height": int(next_resolution[1])},
                            )
                            LOGGER.info(
                                "Adaptive processing resolution adjusted %sx%s -> %sx%s (duty_ema=%.3f target=%.2f)",
                                int(prev_resolution[0]),
                                int(prev_resolution[1]),
                                int(next_resolution[0]),
                                int(next_resolution[1]),
                                duty_cycle_ema,
                                detection_duty_cycle_target,
                            )

                interval_frames = max(1, int(detection_interval if supports_motion_prediction else 1))
                next_detection_frame = frame_id + interval_frames
                next_gap_sec = max(0.0, float(next_detection_frame - frame_id) * frame_period_ema_sec)
                metrics.set_gauge("detection_interval_current", float(interval_frames))
                metrics.set_gauge("detection_next_gap_ms", float(next_gap_sec * 1000.0))
            else:
                timing_store.set(frame_id, "detection_ms", 0.0)
                t1 = time.perf_counter()
                tracks = tracker.predict()
                tr_ms = (time.perf_counter() - t1) * 1000.0

            timing_store.set(frame_id, "tracking_ms", tr_ms)
            stage_stats.add("tracking_ms", tr_ms)
            fps_tracking.inc(1)
            metrics.inc("tracker_frames", 1)

            track_store.update_from_tracker(
                tracks,
                frame_id=frame_id,
                max_tracks=max_tracks,
                max_age_frames=max_track_age_frames,
            )

            track_payload = track_store.to_api_payload(current_frame_id=frame_id)
            track_ages = [max(0, frame_id - int(t.get("last_seen_frame", frame_id))) for t in track_payload.values()]
            metrics.set_gauge("active_tracks", float(len(track_payload)))
            metrics.set_gauge("avg_track_age", float(sum(track_ages) / len(track_ages)) if track_ages else 0.0)
            if start_api:
                api_state.publish_tracks(
                    track_payload,
                    frame_id=frame_id,
                    frame_shape=frame.shape,
                    frame=frame,
                    frame_ts=float(packet.ts),
                )

    api_state = ApiRuntimeState(
        frame_store=frame_store,
        db_path=str(cfg.get("db_path", "./identities.db")),
        media_root=str(cfg.get("media_root", ".")),
        metrics=metrics,
        track_store=track_store,
    )
    recognition = RecognitionWorker(
        frame_store=frame_store,
        track_store=track_store,
        cfg=cfg,
        timing_store=timing_store,
        stage_stats=stage_stats,
        fps_counter=fps_recognition,
        stop_event=stop_event,
        metrics=metrics,
    )
    if not recognition.face_detector_available:
        LOGGER.critical("Face detector unavailable at startup: mediapipe backend is missing")
    api_state.set_capability("face_detector_available", bool(recognition.face_detector_available))
    api_state.set_capability("detector_backend", detector_backend)
    api_state.set_capability("detector_model_path", detector_model_path)
    api_state.set_capability("face_embedder_backend", str(cfg.get("face_embedder_backend", "auto")))
    api_state.set_capability("face_onnx_model_path", str(cfg.get("face_onnx_model_path", "")))
    api_state.set_capability("body_fallback_model_path", str(cfg.get("body_fallback_model_path", "")))
    api_state.set_capability(
        "detector_model_input",
        {"width": int(detector.model_input_size[0]), "height": int(detector.model_input_size[1])},
    )
    api_state.set_capability("adaptive_detection_interval", adaptive_detection_interval)
    api_state.set_capability("adaptive_detection_resolution", adaptive_detection_resolution)
    api_state.set_capability("processing_resolution", {"width": int(imgsz[0]), "height": int(imgsz[1])})
    api_state.set_capability(
        "processing_resolution_profiles",
        [{"width": int(width), "height": int(height)} for width, height in resolution_profiles],
    )
    api_state.set_capability("detection_duty_cycle_target", detection_duty_cycle_target)
    api_state.set_capability("detection_duty_cycle_ema_alpha", detection_duty_cycle_ema_alpha)

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
    watched_metrics = [
        "detector_frames",
        "tracker_frames",
        "face_attempts",
        "face_detections",
        "embedding_success",
        "db_writes",
    ]
    metric_prev = metrics.counters()
    metric_zero_since = {name: time.time() for name in watched_metrics}

    try:
        while not stop_event.is_set():
            packet = frame_store.wait_for_new(last_rendered, timeout=0.05)
            now = time.time()

            if benchmark and now - started_at >= float(benchmark_seconds):
                break

            if packet is None:
                if now >= next_log_at:
                    stage = _log_stage_window(
                        stage_stats,
                        fps_detection,
                        fps_tracking,
                        fps_recognition,
                        fps_render,
                        window_sec=10.0,
                    )
                    metrics.set_gauge("fps_detector", stage["det_fps"])
                    metrics.set_gauge("fps_tracker", stage["trk_fps"])
                    metrics.set_gauge("fps_recognition", stage["rec_fps"])
                    metrics.set_gauge("fps_render", stage["rnd_fps"])
                    metrics.set_gauge("ms_avg_detection", stage["det_avg_ms"])
                    metrics.set_gauge("ms_p50_detection", stage["det_p50_ms"])
                    metrics.set_gauge("ms_p95_detection", stage["det_p95_ms"])
                    metrics.set_gauge("ms_avg_tracking", stage["trk_avg_ms"])
                    metrics.set_gauge("ms_p50_tracking", stage["trk_p50_ms"])
                    metrics.set_gauge("ms_p95_tracking", stage["trk_p95_ms"])
                    metrics.set_gauge("ms_avg_recognition", stage["rec_avg_ms"])
                    metrics.set_gauge("ms_p50_recognition", stage["rec_p50_ms"])
                    metrics.set_gauge("ms_p95_recognition", stage["rec_p95_ms"])
                    metrics.set_gauge("ms_avg_render", stage["rnd_avg_ms"])
                    metrics.set_gauge("ms_p50_render", stage["rnd_p50_ms"])
                    metrics.set_gauge("ms_p95_render", stage["rnd_p95_ms"])
                    curr = metrics.counters()
                    window_delta = {k: int(curr.get(k, 0) - metric_prev.get(k, 0)) for k in watched_metrics}
                    metric_prev = curr
                    for name in watched_metrics:
                        if window_delta.get(name, 0) <= 0:
                            metric_zero_since[name] = metric_zero_since.get(name, now)
                        else:
                            metric_zero_since[name] = now
                    zero_over = [n for n, ts in metric_zero_since.items() if (now - ts) >= 10.0]
                    if zero_over:
                        LOGGER.warning("metrics_zero_over_10s metrics=%s", ",".join(sorted(zero_over)))
                    LOGGER.info(
                        "runtime_metrics detector_fps=%.2f tracker_fps=%.2f detection_interval=%.0f active_tracks=%s avg_track_age=%.2f "
                        "face_attempts=%s face_detections=%s embedding_success=%s db_writes=%s",
                        stage["det_fps"],
                        stage["trk_fps"],
                        float(metrics.gauges().get("detection_interval_current", 1.0)),
                        int(metrics.gauges().get("active_tracks", 0.0)),
                        float(metrics.gauges().get("avg_track_age", 0.0)),
                        int(window_delta.get("face_attempts", 0)),
                        int(window_delta.get("face_detections", 0)),
                        int(window_delta.get("embedding_success", 0)),
                        int(window_delta.get("db_writes", 0)),
                    )
                    next_log_at = now + 10.0
                continue

            frame_id = int(packet.frame_id)
            last_rendered = frame_id

            t0 = time.perf_counter()
            frame = None
            if display_window and not benchmark:
                track_states = track_store.snapshot()
                frame = packet.frame.copy()
                _render_overlay(frame, track_states)

            if display_window and not benchmark:
                assert frame is not None
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
                stage = _log_stage_window(
                    stage_stats,
                    fps_detection,
                    fps_tracking,
                    fps_recognition,
                    fps_render,
                    window_sec=10.0,
                )
                metrics.set_gauge("fps_detector", stage["det_fps"])
                metrics.set_gauge("fps_tracker", stage["trk_fps"])
                metrics.set_gauge("fps_recognition", stage["rec_fps"])
                metrics.set_gauge("fps_render", stage["rnd_fps"])
                metrics.set_gauge("ms_avg_detection", stage["det_avg_ms"])
                metrics.set_gauge("ms_p50_detection", stage["det_p50_ms"])
                metrics.set_gauge("ms_p95_detection", stage["det_p95_ms"])
                metrics.set_gauge("ms_avg_tracking", stage["trk_avg_ms"])
                metrics.set_gauge("ms_p50_tracking", stage["trk_p50_ms"])
                metrics.set_gauge("ms_p95_tracking", stage["trk_p95_ms"])
                metrics.set_gauge("ms_avg_recognition", stage["rec_avg_ms"])
                metrics.set_gauge("ms_p50_recognition", stage["rec_p50_ms"])
                metrics.set_gauge("ms_p95_recognition", stage["rec_p95_ms"])
                metrics.set_gauge("ms_avg_render", stage["rnd_avg_ms"])
                metrics.set_gauge("ms_p50_render", stage["rnd_p50_ms"])
                metrics.set_gauge("ms_p95_render", stage["rnd_p95_ms"])
                curr = metrics.counters()
                window_delta = {k: int(curr.get(k, 0) - metric_prev.get(k, 0)) for k in watched_metrics}
                metric_prev = curr
                for name in watched_metrics:
                    if window_delta.get(name, 0) <= 0:
                        metric_zero_since[name] = metric_zero_since.get(name, now)
                    else:
                        metric_zero_since[name] = now
                zero_over = [n for n, ts in metric_zero_since.items() if (now - ts) >= 10.0]
                if zero_over:
                    LOGGER.warning("metrics_zero_over_10s metrics=%s", ",".join(sorted(zero_over)))
                LOGGER.info(
                    "runtime_metrics detector_fps=%.2f tracker_fps=%.2f detection_interval=%.0f active_tracks=%s avg_track_age=%.2f "
                    "face_attempts=%s face_detections=%s embedding_success=%s db_writes=%s",
                    stage["det_fps"],
                    stage["trk_fps"],
                    float(metrics.gauges().get("detection_interval_current", 1.0)),
                    int(metrics.gauges().get("active_tracks", 0.0)),
                    float(metrics.gauges().get("avg_track_age", 0.0)),
                    int(window_delta.get("face_attempts", 0)),
                    int(window_delta.get("face_detections", 0)),
                    int(window_delta.get("embedding_success", 0)),
                    int(window_delta.get("db_writes", 0)),
                )
                next_log_at = now + 10.0
                timing_store.cleanup_older_than(frame_id - 600)
    finally:
        stop_event.set()
        recognition.stop()
        capture.stop()
        detection_thread.join(timeout=2.0)
        api_state.close()
        if debug_file is not None:
            debug_file.close()
        cv2.destroyAllWindows()

    elapsed = max(1e-6, time.time() - started_at)
    det_count, det_avg, det_p50, det_p95 = stage_stats.summary("detection_ms")
    trk_count, trk_avg, trk_p50, trk_p95 = stage_stats.summary("tracking_ms")
    rec_count, rec_avg, rec_p50, rec_p95 = stage_stats.summary("recognition_ms")
    rnd_count, rnd_avg, rnd_p50, rnd_p95 = stage_stats.summary("render_ms")

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
            "Stage ms avg,p50,p95 | det=%.2f,%.2f,%.2f track=%.2f,%.2f,%.2f rec=%.2f,%.2f,%.2f render=%.2f,%.2f,%.2f",
            det_avg,
            det_p50,
            det_p95,
            trk_avg,
            trk_p50,
            trk_p95,
            rec_avg,
            rec_p50,
            rec_p95,
            rnd_avg,
            rnd_p50,
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
