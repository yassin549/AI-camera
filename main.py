from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import yaml
from flask import Flask, jsonify

import db
from detector import FaceDetector
from embedder import FaceEmbedder
from utils import ensure_rtsp_tcp, iou, sanitize_ts_for_filename, save_thumbnail, timestamp_now_iso


@dataclass
class AppConfig:
    rtsp_url: str
    cosine_threshold: float
    skip_n: int
    imgsz: Tuple[int, int]
    db_path: str
    faces_dir: str
    http_port: int
    cache_seconds: int


class RecentFaceCache:
    """Tracks recently seen boxes to avoid repeated embedding+DB work."""

    def __init__(self, ttl_seconds: int, iou_threshold: float = 0.45) -> None:
        self.ttl_seconds = ttl_seconds
        self.iou_threshold = iou_threshold
        self.entries: List[Dict[str, object]] = []

    def _purge(self, now_ts: float) -> None:
        self.entries = [
            e for e in self.entries if (now_ts - float(e["last_seen_ts"])) <= self.ttl_seconds
        ]

    def lookup(self, box: Tuple[int, int, int, int], now_ts: float) -> Optional[Dict[str, object]]:
        self._purge(now_ts)
        best_entry = None
        best_iou = 0.0
        for entry in self.entries:
            overlap = iou(box, entry["box"])  # type: ignore[arg-type]
            if overlap >= self.iou_threshold and overlap > best_iou:
                best_entry = entry
                best_iou = overlap

        if best_entry is not None:
            best_entry["box"] = box
            best_entry["last_seen_ts"] = now_ts
        return best_entry

    def upsert(
        self, box: Tuple[int, int, int, int], identity_id: int, score: float, now_ts: float
    ) -> None:
        for entry in self.entries:
            overlap = iou(box, entry["box"])  # type: ignore[arg-type]
            if overlap >= self.iou_threshold:
                entry["box"] = box
                entry["id"] = identity_id
                entry["score"] = score
                entry["last_seen_ts"] = now_ts
                return

        self.entries.append(
            {
                "box": box,
                "id": identity_id,
                "score": score,
                "last_seen_ts": now_ts,
            }
        )


class DashboardState:
    """Thread-safe recognized identity view for HTTP endpoint."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: Dict[int, Dict[str, object]] = {}

    def record(
        self,
        identity_id: int,
        first_seen: Optional[str],
        last_seen: str,
        sample_path: Optional[str],
        score: float,
    ) -> None:
        with self._lock:
            row = self._records.get(identity_id)
            if row is None:
                self._records[identity_id] = {
                    "id": identity_id,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "sample_path": sample_path,
                    "last_score": float(score),
                    "count": 1,
                }
                return

            row["last_seen"] = last_seen
            row["last_score"] = float(score)
            row["count"] = int(row["count"]) + 1
            if not row.get("first_seen") and first_seen:
                row["first_seen"] = first_seen
            if sample_path and not row.get("sample_path"):
                row["sample_path"] = sample_path

    def snapshot(self, active_within_seconds: int) -> List[Dict[str, object]]:
        cutoff = datetime.now(timezone.utc).timestamp() - active_within_seconds
        with self._lock:
            items = []
            for row in self._records.values():
                try:
                    ts = datetime.fromisoformat(str(row["last_seen"])).timestamp()
                except ValueError:
                    continue
                if ts >= cutoff:
                    items.append(dict(row))
            items.sort(key=lambda x: int(x["id"]))
            return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTSP face recognition (CPU-only).")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--demo",
        nargs="?",
        const="demo.mp4",
        default=None,
        help="Demo mode. Optional video file path; falls back to webcam index 0.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose per-frame timing logs.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(config_path: str) -> AppConfig:
    defaults = {
        "RTSP_URL": "",
        "cosine_threshold": 0.60,
        "skip_n": 2,
        "imgsz": [640, 360],
        "db_path": "./identities.db",
        "faces_dir": "./faces",
        "http_port": 8080,
        "cache_seconds": 30,
    }
    cfg_data: Dict[str, object] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                cfg_data = loaded
    merged = {**defaults, **cfg_data}
    imgsz = merged.get("imgsz", [640, 360])
    if not isinstance(imgsz, (list, tuple)) or len(imgsz) != 2:
        imgsz = [640, 360]

    return AppConfig(
        rtsp_url=str(merged.get("RTSP_URL", "")),
        cosine_threshold=float(merged.get("cosine_threshold", 0.60)),
        skip_n=max(0, int(merged.get("skip_n", 2))),
        imgsz=(int(imgsz[0]), int(imgsz[1])),
        db_path=str(merged.get("db_path", "./identities.db")),
        faces_dir=str(merged.get("faces_dir", "./faces")),
        http_port=int(merged.get("http_port", 8080)),
        cache_seconds=max(1, int(merged.get("cache_seconds", 30))),
    )


def choose_video_source(
    cfg: AppConfig, demo_arg: Optional[str]
) -> Tuple[Optional[Union[int, str]], bool]:
    if demo_arg is not None:
        candidate_paths = []
        if demo_arg:
            candidate_paths.append(Path(demo_arg))
        candidate_paths.append(Path("demo.mp4"))
        for candidate in candidate_paths:
            if candidate.exists() and candidate.is_file():
                return str(candidate), False
        return 0, False

    env_rtsp = os.getenv("RTSP_URL", "").strip()
    rtsp_url = env_rtsp or cfg.rtsp_url
    if not rtsp_url:
        return None, False
    return ensure_rtsp_tcp(rtsp_url), True


def open_capture(source: Union[int, str], is_rtsp: bool) -> cv2.VideoCapture:
    if is_rtsp:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    return cap


def draw_annotations(
    frame, annotations: List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]]
) -> None:
    for (x1, y1, x2, y2), label, color in annotations:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def scale_annotations(
    annotations: List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]],
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
) -> List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]]:
    """Scale annotation boxes from source-space to destination-space."""
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0:
        return annotations

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    scaled: List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]] = []
    for (x1, y1, x2, y2), label, color in annotations:
        scaled.append(
            (
                (
                    int(round(x1 * sx)),
                    int(round(y1 * sy)),
                    int(round(x2 * sx)),
                    int(round(y2 * sy)),
                ),
                label,
                color,
            )
        )
    return scaled


def start_http_server(
    state: DashboardState, cache_seconds: int, port: int, logger: logging.Logger
) -> threading.Thread:
    app = Flask(__name__)

    @app.get("/identities")
    def identities():
        rows = state.snapshot(active_within_seconds=cache_seconds)
        return jsonify({"identities": rows, "count": len(rows)})

    def _run():
        logger.info("HTTP endpoint ready at http://127.0.0.1:%d/identities", port)
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=True)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def get_identity_meta(
    conn, meta_cache: Dict[int, Dict[str, object]], identity_id: int
) -> Dict[str, object]:
    if identity_id in meta_cache:
        return meta_cache[identity_id]
    meta = db.get_identity(conn, identity_id)
    if meta is None:
        meta = {"id": identity_id, "first_seen": None, "last_seen": None, "sample_path": None}
    meta_cache[identity_id] = meta
    return meta


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    cfg = load_config(args.config)
    source, is_rtsp = choose_video_source(cfg, args.demo)
    if source is None:
        logger.error("No RTSP source found. Set RTSP_URL or run with --demo.")
        return

    if is_rtsp:
        logger.info("Using RTSP source: %s", source)
    else:
        logger.info("Using demo source: %s", source)

    os.makedirs(cfg.faces_dir, exist_ok=True)
    conn = db.get_connection(cfg.db_path)
    db.init_db(conn)
    identity_meta: Dict[int, Dict[str, object]] = {
        int(row["id"]): row for row in db.list_identities(conn)
    }

    cap = open_capture(source, is_rtsp)
    if not cap.isOpened():
        logger.error("Failed to open capture source: %s", source)
        conn.close()
        return

    detector = FaceDetector()
    embedder = FaceEmbedder()
    recent_faces = RecentFaceCache(ttl_seconds=cfg.cache_seconds)
    dashboard_state = DashboardState()
    start_http_server(dashboard_state, cfg.cache_seconds, cfg.http_port, logger)

    frame_count = 0
    reconnect_backoff = 0.5
    last_db_update_ts: Dict[int, float] = {}
    window_name = "Face Recognition"
    window_initialized = False
    last_window_size: Optional[Tuple[int, int]] = None

    try:
        while True:
            ret, frame = cap.read()
            if (not ret) or frame is None or frame.size == 0:
                logger.warning("Frame read failed. Reconnecting in %.1fs...", reconnect_backoff)
                cap.release()
                time.sleep(reconnect_backoff)
                cap = open_capture(source, is_rtsp)
                reconnect_backoff = min(10.0, reconnect_backoff * 2.0)
                continue

            reconnect_backoff = 0.5
            frame_count += 1
            frame_h, frame_w = frame.shape[:2]
            if not window_initialized:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, frame_w, frame_h)
                window_initialized = True
                last_window_size = (frame_w, frame_h)
            elif last_window_size != (frame_w, frame_h):
                cv2.resizeWindow(window_name, frame_w, frame_h)
                last_window_size = (frame_w, frame_h)

            resized = cv2.resize(frame, cfg.imgsz)

            if frame_count % (cfg.skip_n + 1) != 0:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            detect_t0 = time.perf_counter()
            boxes = detector.detect_faces(resized)
            detect_ms = (time.perf_counter() - detect_t0) * 1000.0
            embed_total_ms = 0.0
            annotations: List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]] = []
            now_mono = time.monotonic()

            for box in boxes:
                ts = timestamp_now_iso()
                cache_entry = recent_faces.lookup(box, now_mono)
                if cache_entry is not None:
                    identity_id = int(cache_entry["id"])
                    score = float(cache_entry["score"])
                    if (now_mono - last_db_update_ts.get(identity_id, 0.0)) >= 3.0:
                        db.update_last_seen(conn, identity_id, ts)
                        last_db_update_ts[identity_id] = now_mono
                        meta_row = get_identity_meta(conn, identity_meta, identity_id)
                        meta_row["last_seen"] = ts
                    meta_row = get_identity_meta(conn, identity_meta, identity_id)
                    dashboard_state.record(
                        identity_id=identity_id,
                        first_seen=meta_row.get("first_seen"),  # type: ignore[arg-type]
                        last_seen=ts,
                        sample_path=meta_row.get("sample_path"),  # type: ignore[arg-type]
                        score=score,
                    )
                    annotations.append((box, f"ID:{identity_id} (score:{score:.2f})", (0, 210, 0)))
                    continue

                embed_t0 = time.perf_counter()
                embedding = embedder.compute_embedding(resized, box)
                embed_total_ms += (time.perf_counter() - embed_t0) * 1000.0
                if embedding is None:
                    continue

                best_id, similarity = db.find_best_match(conn, embedding)
                if best_id is not None and similarity >= cfg.cosine_threshold:
                    db.update_last_seen(conn, best_id, ts)
                    last_db_update_ts[best_id] = now_mono
                    recent_faces.upsert(box, best_id, similarity, now_mono)

                    meta_row = get_identity_meta(conn, identity_meta, best_id)
                    meta_row["last_seen"] = ts
                    dashboard_state.record(
                        identity_id=best_id,
                        first_seen=meta_row.get("first_seen"),  # type: ignore[arg-type]
                        last_seen=ts,
                        sample_path=meta_row.get("sample_path"),  # type: ignore[arg-type]
                        score=similarity,
                    )
                    annotations.append((box, f"ID:{best_id} (score:{similarity:.2f})", (0, 210, 0)))
                    continue

                crop, _ = embedder.extract_face_crop(resized, box)
                new_id = db.add_identity(conn, embedding, sample_path=None, ts=ts)
                thumb_path: Optional[str] = None
                if crop is not None and crop.size > 0:
                    thumb_name = f"{new_id}_{sanitize_ts_for_filename(ts)}.jpg"
                    thumb_path = os.path.join(cfg.faces_dir, thumb_name)
                    if save_thumbnail(thumb_path, crop):
                        db.update_sample_path(conn, new_id, thumb_path)
                    else:
                        thumb_path = None

                identity_meta[new_id] = {
                    "id": new_id,
                    "first_seen": ts,
                    "last_seen": ts,
                    "sample_path": thumb_path,
                }
                recent_faces.upsert(box, new_id, 1.0, now_mono)
                dashboard_state.record(
                    identity_id=new_id,
                    first_seen=ts,
                    last_seen=ts,
                    sample_path=thumb_path,
                    score=1.0,
                )
                annotations.append((box, f"New ID:{new_id}", (0, 165, 255)))

            display_frame = frame.copy()
            scaled_annotations = scale_annotations(
                annotations=annotations,
                src_size=cfg.imgsz,
                dst_size=(frame_w, frame_h),
            )
            draw_annotations(display_frame, scaled_annotations)
            if args.verbose:
                logger.debug(
                    "frame=%d faces=%d detect_ms=%.2f embed_ms=%.2f",
                    frame_count,
                    len(boxes),
                    detect_ms,
                    embed_total_ms,
                )

            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        detector.close()
        cap.release()
        conn.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
