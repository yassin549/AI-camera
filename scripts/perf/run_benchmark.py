"""Phase 6 benchmark harness: repeatable run + transport/latency regression reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class BenchmarkSummary:
    elapsed_s: float
    fps: Dict[str, float]
    stage_ms: Dict[str, Dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeatable AIcam benchmark and write JSON report")
    parser.add_argument("--config", default="config.yaml", help="Config file for main.py")
    parser.add_argument("--seconds", type=int, default=60, help="Benchmark duration seconds")
    parser.add_argument(
        "--output-dir",
        default=str(Path("artifacts") / "perf"),
        help="Directory where benchmark artifacts are written",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="API base URL for telemetry polling")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between telemetry polls")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable to run main.py")
    parser.add_argument("--skip-api", action="store_true", help="Do not start API during benchmark")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=25.0,
        help="Maximum expected seconds until API/capture readiness before marking run invalid",
    )
    parser.add_argument(
        "--require-detector-frames",
        type=int,
        default=1,
        help="Minimum detector frame counter required for a valid run",
    )
    parser.add_argument(
        "--require-tracker-frames",
        type=int,
        default=1,
        help="Minimum tracker frame counter required for a valid run",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Return success even when benchmark quality checks fail",
    )
    parser.add_argument(
        "--allow-existing-service",
        action="store_true",
        help="Allow running when a service is already bound on --base-url",
    )
    parser.add_argument(
        "--compare-to",
        default="",
        help="Optional path to previous report.json to include delta comparison",
    )
    parser.add_argument(
        "--max-metadata-lag-p95-ms",
        type=float,
        default=600.0,
        help="Maximum allowed p95 metadata lag in milliseconds when API telemetry is available",
    )
    parser.add_argument(
        "--max-janus-ttff-p95-ms",
        type=float,
        default=8000.0,
        help="Maximum allowed p95 time-to-first-video-frame for Janus in milliseconds",
    )
    parser.add_argument(
        "--max-overlay-jank-ratio-p95",
        type=float,
        default=0.40,
        help="Maximum allowed p95 overlay jank ratio (0..1)",
    )
    parser.add_argument(
        "--min-overlay-fps-avg",
        type=float,
        default=8.0,
        help="Minimum allowed average overlay FPS when frontend telemetry is available",
    )
    parser.add_argument(
        "--p95-regression-tolerance-ms",
        type=float,
        default=5.0,
        help="Allowed p95 increase (ms) versus --compare-to before failing quality",
    )
    parser.add_argument(
        "--fps-regression-tolerance",
        type=float,
        default=0.5,
        help="Allowed FPS drop versus --compare-to before failing quality",
    )
    return parser.parse_args()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid config at {path}")
    return payload


def write_config(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def fetch_json(url: str, timeout: float = 0.5) -> Optional[Dict[str, Any]]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=max(0.1, float(timeout))) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            if isinstance(payload, dict):
                return payload
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    return None


def parse_log_summary(log_text: str) -> BenchmarkSummary:
    fps_match = re.search(
        r"Benchmark\s+([0-9.]+)s\s+\|\s+det_fps=([0-9.]+)\s+track_fps=([0-9.]+)\s+rec_fps=([0-9.]+)\s+render_fps=([0-9.]+)",
        log_text,
    )
    stage_match = re.search(
        r"Stage ms avg,p50,p95 \| det=([0-9.]+),([0-9.]+),([0-9.]+)\s+track=([0-9.]+),([0-9.]+),([0-9.]+)\s+rec=([0-9.]+),([0-9.]+),([0-9.]+)\s+render=([0-9.]+),([0-9.]+),([0-9.]+)",
        log_text,
    )
    if not fps_match:
        raise RuntimeError("Could not parse benchmark FPS summary from logs")
    if not stage_match:
        raise RuntimeError("Could not parse stage avg,p50,p95 summary from logs")

    elapsed = float(fps_match.group(1))
    fps = {
        "detector": float(fps_match.group(2)),
        "tracker": float(fps_match.group(3)),
        "recognition": float(fps_match.group(4)),
        "render": float(fps_match.group(5)),
    }
    stage_ms = {
        "detection": {
            "avg": float(stage_match.group(1)),
            "p50": float(stage_match.group(2)),
            "p95": float(stage_match.group(3)),
        },
        "tracking": {
            "avg": float(stage_match.group(4)),
            "p50": float(stage_match.group(5)),
            "p95": float(stage_match.group(6)),
        },
        "recognition": {
            "avg": float(stage_match.group(7)),
            "p50": float(stage_match.group(8)),
            "p95": float(stage_match.group(9)),
        },
        "render": {
            "avg": float(stage_match.group(10)),
            "p50": float(stage_match.group(11)),
            "p95": float(stage_match.group(12)),
        },
    }
    return BenchmarkSummary(elapsed_s=elapsed, fps=fps, stage_ms=stage_ms)


def _as_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not (parsed == parsed):  # NaN guard
        return None
    return parsed


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (max(0.0, min(100.0, p)) / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def summarize_transport(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {"samples": 0}

    ws_clients = [int(s.get("debug_status", {}).get("ws_clients", 0) or 0) for s in samples]
    mjpeg_clients = [int(s.get("debug_status", {}).get("mjpeg_clients", 0) or 0) for s in samples]
    active_tracks = [float(s.get("metrics", {}).get("gauges", {}).get("active_tracks", 0.0) or 0.0) for s in samples]
    detector_frames = [int(s.get("metrics", {}).get("counters", {}).get("detector_frames", 0) or 0) for s in samples]
    tracker_frames = [int(s.get("metrics", {}).get("counters", {}).get("tracker_frames", 0) or 0) for s in samples]
    metadata_lag_samples_ms: List[float] = []
    metadata_lag_client_samples_ms: List[float] = []
    metadata_publish_lag_samples_ms: List[float] = []
    janus_ttff_samples_ms: List[float] = []
    overlay_fps_samples: List[float] = []
    overlay_jank_samples: List[float] = []

    first_api_up: Optional[float] = None
    first_capture_running: Optional[float] = None
    for sample in samples:
        t_rel = float(sample.get("t_rel_s", 0.0) or 0.0)
        debug_status = sample.get("debug_status", {})
        health = sample.get("health", {})
        if first_api_up is None and bool(debug_status.get("api_up")):
            first_api_up = t_rel
        if first_capture_running is None and bool(health.get("capture_running", False)):
            first_capture_running = t_rel
        latest_meta = sample.get("realtime_latest", {})
        if isinstance(latest_meta, dict):
            capture_ts_unix = _as_float(latest_meta.get("capture_ts_unix"))
            recv_wall_unix = _as_float(sample.get("t_wall_unix"))
            if capture_ts_unix is not None and recv_wall_unix is not None:
                lag_ms = max(0.0, (recv_wall_unix - capture_ts_unix) * 1000.0)
                metadata_lag_samples_ms.append(float(lag_ms))
            publish_lag_ms = _as_float(latest_meta.get("metadata_lag_ms"))
            if publish_lag_ms is not None and publish_lag_ms >= 0:
                metadata_publish_lag_samples_ms.append(float(publish_lag_ms))
        gauges = sample.get("metrics", {}).get("gauges", {})
        if isinstance(gauges, dict):
            metadata_lag_client = _as_float(gauges.get("metadata_lag_ms_p95"))
            if metadata_lag_client is not None and metadata_lag_client >= 0:
                metadata_lag_client_samples_ms.append(float(metadata_lag_client))
            janus_ttff = _as_float(gauges.get("janus_ttff_ms"))
            if janus_ttff is not None and janus_ttff >= 0:
                janus_ttff_samples_ms.append(float(janus_ttff))
            overlay_fps = _as_float(gauges.get("overlay_fps"))
            if overlay_fps is not None and overlay_fps >= 0:
                overlay_fps_samples.append(float(overlay_fps))
            overlay_jank = _as_float(gauges.get("overlay_jank_ratio"))
            if overlay_jank is not None and overlay_jank >= 0:
                overlay_jank_samples.append(float(overlay_jank))

    metadata_lag_client_p50 = _percentile(metadata_lag_client_samples_ms, 50)
    metadata_lag_client_p95 = _percentile(metadata_lag_client_samples_ms, 95)
    metadata_lag_capture_to_poll_p50 = _percentile(metadata_lag_samples_ms, 50)
    metadata_lag_capture_to_poll_p95 = _percentile(metadata_lag_samples_ms, 95)
    metadata_lag_e2e_p50 = (
        metadata_lag_client_p50 if metadata_lag_client_p50 is not None else metadata_lag_capture_to_poll_p50
    )
    metadata_lag_e2e_p95 = (
        metadata_lag_client_p95 if metadata_lag_client_p95 is not None else metadata_lag_capture_to_poll_p95
    )

    return {
        "samples": len(samples),
        "ws_clients_max": max(ws_clients) if ws_clients else 0,
        "mjpeg_clients_max": max(mjpeg_clients) if mjpeg_clients else 0,
        "active_tracks_max": max(active_tracks) if active_tracks else 0.0,
        "active_tracks_avg": (sum(active_tracks) / len(active_tracks)) if active_tracks else 0.0,
        "detector_frames_max": max(detector_frames) if detector_frames else 0,
        "tracker_frames_max": max(tracker_frames) if tracker_frames else 0,
        "first_api_up_s": first_api_up,
        "first_capture_running_s": first_capture_running,
        "metadata_lag_samples": len(metadata_lag_samples_ms),
        "metadata_lag_client_samples": len(metadata_lag_client_samples_ms),
        "metadata_lag_capture_to_poll_ms_p50": metadata_lag_capture_to_poll_p50,
        "metadata_lag_capture_to_poll_ms_p95": metadata_lag_capture_to_poll_p95,
        "metadata_lag_ms_p50": metadata_lag_capture_to_poll_p50,
        "metadata_lag_ms_p95": metadata_lag_capture_to_poll_p95,
        "metadata_lag_client_ms_p50": metadata_lag_client_p50,
        "metadata_lag_client_ms_p95": metadata_lag_client_p95,
        "metadata_lag_e2e_ms_p50": metadata_lag_e2e_p50,
        "metadata_lag_e2e_ms_p95": metadata_lag_e2e_p95,
        "metadata_lag_e2e_source": "client_metrics" if metadata_lag_client_p95 is not None else "api_poll",
        "metadata_publish_lag_ms_p50": _percentile(metadata_publish_lag_samples_ms, 50),
        "metadata_publish_lag_ms_p95": _percentile(metadata_publish_lag_samples_ms, 95),
        "janus_ttff_ms_p50": _percentile(janus_ttff_samples_ms, 50),
        "janus_ttff_ms_p95": _percentile(janus_ttff_samples_ms, 95),
        "overlay_fps_avg": (sum(overlay_fps_samples) / len(overlay_fps_samples)) if overlay_fps_samples else None,
        "overlay_jank_ratio_p95": _percentile(overlay_jank_samples, 95),
    }


def _summary_from_metrics(samples: List[Dict[str, Any]], elapsed_hint_s: float) -> Optional[BenchmarkSummary]:
    for sample in reversed(samples):
        metrics = sample.get("metrics", {})
        gauges = metrics.get("gauges", {}) if isinstance(metrics, dict) else {}
        if not isinstance(gauges, dict):
            continue
        required = [
            "fps_detector",
            "fps_tracker",
            "fps_recognition",
            "fps_render",
            "ms_avg_detection",
            "ms_p50_detection",
            "ms_p95_detection",
            "ms_avg_tracking",
            "ms_p50_tracking",
            "ms_p95_tracking",
            "ms_avg_recognition",
            "ms_p50_recognition",
            "ms_p95_recognition",
            "ms_avg_render",
            "ms_p50_render",
            "ms_p95_render",
        ]
        if not all(key in gauges for key in required):
            continue
        return BenchmarkSummary(
            elapsed_s=max(0.0, float(elapsed_hint_s)),
            fps={
                "detector": float(gauges["fps_detector"]),
                "tracker": float(gauges["fps_tracker"]),
                "recognition": float(gauges["fps_recognition"]),
                "render": float(gauges["fps_render"]),
            },
            stage_ms={
                "detection": {
                    "avg": float(gauges["ms_avg_detection"]),
                    "p50": float(gauges["ms_p50_detection"]),
                    "p95": float(gauges["ms_p95_detection"]),
                },
                "tracking": {
                    "avg": float(gauges["ms_avg_tracking"]),
                    "p50": float(gauges["ms_p50_tracking"]),
                    "p95": float(gauges["ms_p95_tracking"]),
                },
                "recognition": {
                    "avg": float(gauges["ms_avg_recognition"]),
                    "p50": float(gauges["ms_p50_recognition"]),
                    "p95": float(gauges["ms_p95_recognition"]),
                },
                "render": {
                    "avg": float(gauges["ms_avg_render"]),
                    "p50": float(gauges["ms_p50_render"]),
                    "p95": float(gauges["ms_p95_render"]),
                },
            },
        )
    return None


def _evaluate_quality(
    samples: List[Dict[str, Any]],
    summary: Optional[BenchmarkSummary],
    log_text: str,
    *,
    expect_api: bool,
    startup_timeout_s: float,
    require_detector_frames: int,
    require_tracker_frames: int,
    max_metadata_lag_p95_ms: float,
    max_janus_ttff_p95_ms: float,
    max_overlay_jank_ratio_p95: float,
    min_overlay_fps_avg: float,
) -> Dict[str, Any]:
    transport = summarize_transport(samples)
    issues: List[str] = []
    warnings: List[str] = []

    det_max = int(transport.get("detector_frames_max", 0) or 0)
    trk_max = int(transport.get("tracker_frames_max", 0) or 0)
    if expect_api:
        if det_max < max(0, int(require_detector_frames)):
            issues.append(f"detector_frames_below_required(max={det_max}, required={int(require_detector_frames)})")
        if trk_max < max(0, int(require_tracker_frames)):
            issues.append(f"tracker_frames_below_required(max={trk_max}, required={int(require_tracker_frames)})")
    elif det_max == 0 and trk_max == 0:
        warnings.append("telemetry_counters_unavailable_without_api")

    if expect_api:
        api_up = transport.get("first_api_up_s")
        capture_up = transport.get("first_capture_running_s")
        if api_up is None:
            issues.append("api_never_ready")
        elif float(api_up) > float(startup_timeout_s):
            issues.append(f"api_ready_after_timeout({float(api_up):.2f}s>{float(startup_timeout_s):.2f}s)")
        if capture_up is None:
            issues.append("capture_never_running")
        elif float(capture_up) > float(startup_timeout_s):
            issues.append(f"capture_ready_after_timeout({float(capture_up):.2f}s>{float(startup_timeout_s):.2f}s)")
        metadata_lag_p95 = _as_float(transport.get("metadata_lag_e2e_ms_p95"))
        if metadata_lag_p95 is None:
            metadata_lag_p95 = _as_float(transport.get("metadata_lag_ms_p95"))
        if metadata_lag_p95 is None:
            warnings.append("metadata_lag_e2e_p95_unavailable")
        elif metadata_lag_p95 > float(max_metadata_lag_p95_ms):
            issues.append(
                f"metadata_lag_e2e_p95_over_budget({metadata_lag_p95:.1f}ms>{float(max_metadata_lag_p95_ms):.1f}ms)"
            )
        janus_ttff_p95 = _as_float(transport.get("janus_ttff_ms_p95"))
        if janus_ttff_p95 is None:
            warnings.append("janus_ttff_p95_unavailable")
        elif janus_ttff_p95 > float(max_janus_ttff_p95_ms):
            issues.append(
                f"janus_ttff_p95_over_budget({janus_ttff_p95:.1f}ms>{float(max_janus_ttff_p95_ms):.1f}ms)"
            )
        overlay_jank_p95 = _as_float(transport.get("overlay_jank_ratio_p95"))
        if overlay_jank_p95 is None:
            warnings.append("overlay_jank_ratio_p95_unavailable")
        elif overlay_jank_p95 > float(max_overlay_jank_ratio_p95):
            issues.append(
                f"overlay_jank_ratio_p95_over_budget({overlay_jank_p95:.3f}>{float(max_overlay_jank_ratio_p95):.3f})"
            )
        overlay_fps_avg = _as_float(transport.get("overlay_fps_avg"))
        if overlay_fps_avg is None:
            warnings.append("overlay_fps_avg_unavailable")
        elif overlay_fps_avg < float(min_overlay_fps_avg):
            issues.append(
                f"overlay_fps_avg_under_budget({overlay_fps_avg:.2f}<{float(min_overlay_fps_avg):.2f})"
            )

    lower_log = log_text.lower()
    if "error while attempting to bind on address" in lower_log:
        issues.append("api_port_bind_conflict")
    if "capture connect failed" in lower_log:
        warnings.append("capture_connect_retries_detected")
    if "capture read failed" in lower_log:
        warnings.append("capture_read_retries_detected")

    if summary is None:
        issues.append("summary_missing")
    else:
        if float(summary.fps.get("detector", 0.0)) <= 0.0:
            issues.append("detector_fps_zero")
        if float(summary.fps.get("tracker", 0.0)) <= 0.0:
            issues.append("tracker_fps_zero")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": sorted(set(warnings)),
        "checks": {
            "expect_api": bool(expect_api),
            "startup_timeout_s": float(startup_timeout_s),
            "require_detector_frames": int(require_detector_frames),
            "require_tracker_frames": int(require_tracker_frames),
            "max_metadata_lag_p95_ms": float(max_metadata_lag_p95_ms),
            "max_janus_ttff_p95_ms": float(max_janus_ttff_p95_ms),
            "max_overlay_jank_ratio_p95": float(max_overlay_jank_ratio_p95),
            "min_overlay_fps_avg": float(min_overlay_fps_avg),
            "detector_frames_max": det_max,
            "tracker_frames_max": trk_max,
            "first_api_up_s": transport.get("first_api_up_s"),
            "first_capture_running_s": transport.get("first_capture_running_s"),
            "metadata_lag_ms_p95": transport.get("metadata_lag_ms_p95"),
            "metadata_lag_e2e_ms_p95": transport.get("metadata_lag_e2e_ms_p95"),
            "metadata_lag_e2e_source": transport.get("metadata_lag_e2e_source"),
            "janus_ttff_ms_p95": transport.get("janus_ttff_ms_p95"),
            "overlay_jank_ratio_p95": transport.get("overlay_jank_ratio_p95"),
            "overlay_fps_avg": transport.get("overlay_fps_avg"),
        },
    }


def _extract_for_delta(report: Dict[str, Any], path: Tuple[str, ...]) -> Optional[float]:
    cur: Any = report
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    try:
        return float(cur)
    except Exception:
        return None


def _build_comparison(current_report: Dict[str, Any], previous_path: Path) -> Dict[str, Any]:
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    metrics_to_compare = {
        "detector_fps": ("summary", "fps", "detector"),
        "tracker_fps": ("summary", "fps", "tracker"),
        "recognition_fps": ("summary", "fps", "recognition"),
        "detection_p95_ms": ("summary", "stage_ms", "detection", "p95"),
        "tracking_p95_ms": ("summary", "stage_ms", "tracking", "p95"),
        "recognition_p95_ms": ("summary", "stage_ms", "recognition", "p95"),
        "render_p95_ms": ("summary", "stage_ms", "render", "p95"),
        "metadata_lag_p95_ms": ("transport", "summary", "metadata_lag_ms_p95"),
        "metadata_lag_e2e_p95_ms": ("transport", "summary", "metadata_lag_e2e_ms_p95"),
        "janus_ttff_p95_ms": ("transport", "summary", "janus_ttff_ms_p95"),
        "overlay_fps_avg": ("transport", "summary", "overlay_fps_avg"),
        "overlay_jank_ratio_p95": ("transport", "summary", "overlay_jank_ratio_p95"),
    }
    deltas: Dict[str, Dict[str, Optional[float]]] = {}
    for name, path in metrics_to_compare.items():
        prev_val = _extract_for_delta(previous, path)
        cur_val = _extract_for_delta(current_report, path)
        delta = None if prev_val is None or cur_val is None else float(cur_val - prev_val)
        deltas[name] = {"current": cur_val, "previous": prev_val, "delta": delta}
    return {
        "previous_report": str(previous_path.resolve()),
        "previous_run_id": previous.get("run_id"),
        "delta": deltas,
    }


def _evaluate_regression(
    current_report: Dict[str, Any],
    previous_path: Path,
    *,
    p95_tolerance_ms: float,
    fps_tolerance: float,
) -> Dict[str, Any]:
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    issues: List[str] = []
    checks: Dict[str, Dict[str, Optional[float]]] = {}

    higher_is_worse = {
        "detection_p95_ms": ("summary", "stage_ms", "detection", "p95"),
        "tracking_p95_ms": ("summary", "stage_ms", "tracking", "p95"),
        "recognition_p95_ms": ("summary", "stage_ms", "recognition", "p95"),
        "render_p95_ms": ("summary", "stage_ms", "render", "p95"),
        "metadata_lag_p95_ms": ("transport", "summary", "metadata_lag_ms_p95"),
        "metadata_lag_e2e_p95_ms": ("transport", "summary", "metadata_lag_e2e_ms_p95"),
        "janus_ttff_p95_ms": ("transport", "summary", "janus_ttff_ms_p95"),
        "overlay_jank_p95": ("transport", "summary", "overlay_jank_ratio_p95"),
    }
    for name, path in higher_is_worse.items():
        prev_val = _extract_for_delta(previous, path)
        cur_val = _extract_for_delta(current_report, path)
        checks[name] = {"current": cur_val, "previous": prev_val, "delta": None if prev_val is None or cur_val is None else cur_val - prev_val}
        if prev_val is None or cur_val is None:
            continue
        allowed = float(p95_tolerance_ms)
        if name.endswith("_p95") or "jank" in name:
            allowed = max(0.0, allowed / 1000.0)
        if cur_val > prev_val + allowed:
            issues.append(f"{name}_regressed({cur_val:.3f}>{prev_val:.3f}+{allowed:.3f})")

    higher_is_better = {
        "detector_fps": ("summary", "fps", "detector"),
        "tracker_fps": ("summary", "fps", "tracker"),
        "recognition_fps": ("summary", "fps", "recognition"),
        "overlay_fps_avg": ("transport", "summary", "overlay_fps_avg"),
    }
    for name, path in higher_is_better.items():
        prev_val = _extract_for_delta(previous, path)
        cur_val = _extract_for_delta(current_report, path)
        checks[name] = {"current": cur_val, "previous": prev_val, "delta": None if prev_val is None or cur_val is None else cur_val - prev_val}
        if prev_val is None or cur_val is None:
            continue
        if cur_val < prev_val - float(fps_tolerance):
            issues.append(f"{name}_regressed({cur_val:.3f}<{prev_val:.3f}-{float(fps_tolerance):.3f})")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "checks": checks,
        "previous_report": str(previous_path.resolve()),
        "tolerance": {
            "p95_ms": float(p95_tolerance_ms),
            "fps": float(fps_tolerance),
        },
    }


def main() -> int:
    args = parse_args()
    started_at = now_utc_iso()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir) / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    src_config = Path(args.config).resolve()
    cfg = load_config(src_config)
    cfg["debug"] = True
    cfg["display_window"] = False
    cfg["debug_csv"] = str((run_dir / "timings.csv").as_posix())
    run_config = run_dir / "benchmark_config.yaml"
    write_config(run_config, cfg)
    run_config_hash = hashlib.sha256(run_config.read_bytes()).hexdigest()

    log_path = run_dir / "benchmark.log"
    report_path = run_dir / "report.json"
    process_cmd = [
        args.python_exe,
        "main.py",
        "--config",
        str(run_config),
        "--benchmark",
        "--benchmark-seconds",
        str(int(args.seconds)),
        "--no-display",
    ]
    if not args.skip_api:
        process_cmd.append("--start-api")

    if not args.allow_existing_service and not args.skip_api:
        existing = fetch_json(f"{args.base_url.rstrip('/')}/metrics", timeout=0.4)
        if existing is not None:
            print(
                f"Refusing to start benchmark: existing service detected at {args.base_url.rstrip('/')}. "
                "Use --allow-existing-service or a different --base-url/http_port."
            )
            return 4

    git_commit = run_cmd(["git", "rev-parse", "HEAD"])
    git_branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    telemetry_samples: List[Dict[str, Any]] = []
    telemetry_lock = threading.Lock()
    telemetry_stop = threading.Event()
    telemetry_started_mono = time.monotonic()

    def telemetry_worker() -> None:
        interval = max(0.2, float(args.poll_interval))
        while not telemetry_stop.is_set():
            now_rel = round(time.monotonic() - telemetry_started_mono, 3)
            now_wall = time.time()
            status = fetch_json(f"{args.base_url.rstrip('/')}/api/debug/status")
            metrics = fetch_json(f"{args.base_url.rstrip('/')}/metrics")
            health = fetch_json(f"{args.base_url.rstrip('/')}/api/health")
            realtime_latest = fetch_json(f"{args.base_url.rstrip('/')}/api/realtime/latest")
            with telemetry_lock:
                telemetry_samples.append(
                    {
                        "t_rel_s": now_rel,
                        "t_wall_unix": now_wall,
                        "debug_status": status or {},
                        "metrics": metrics or {},
                        "health": health or {},
                        "realtime_latest": realtime_latest or {},
                    }
                )
            telemetry_stop.wait(interval)

    telemetry_thread = threading.Thread(target=telemetry_worker, daemon=True, name="perf-telemetry")
    telemetry_thread.start()

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            process_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            text = line.rstrip("\n")
            print(text)
            log_file.write(text + "\n")
            log_file.flush()

        return_code = proc.wait(timeout=max(30, int(args.seconds) + 30))
    telemetry_stop.set()
    telemetry_thread.join(timeout=2.0)

    if return_code != 0:
        print(f"Benchmark process failed with exit code {return_code}")
        return return_code

    log_text = log_path.read_text(encoding="utf-8")

    with telemetry_lock:
        samples_snapshot = list(telemetry_samples)

    summary: Optional[BenchmarkSummary] = None
    summary_source = "logs"
    summary_error = ""
    try:
        summary = parse_log_summary(log_text)
    except Exception as exc:
        summary_error = str(exc)
        summary = _summary_from_metrics(samples_snapshot, float(args.seconds))
        summary_source = "metrics_fallback" if summary is not None else "missing"

    quality = _evaluate_quality(
        samples_snapshot,
        summary,
        log_text,
        expect_api=not bool(args.skip_api),
        startup_timeout_s=float(args.startup_timeout),
        require_detector_frames=int(args.require_detector_frames),
        require_tracker_frames=int(args.require_tracker_frames),
        max_metadata_lag_p95_ms=float(args.max_metadata_lag_p95_ms),
        max_janus_ttff_p95_ms=float(args.max_janus_ttff_p95_ms),
        max_overlay_jank_ratio_p95=float(args.max_overlay_jank_ratio_p95),
        min_overlay_fps_avg=float(args.min_overlay_fps_avg),
    )

    summary_payload: Dict[str, Any]
    if summary is None:
        summary_payload = {
            "elapsed_s": 0.0,
            "fps": {"detector": 0.0, "tracker": 0.0, "recognition": 0.0, "render": 0.0},
            "stage_ms": {
                "detection": {"avg": 0.0, "p50": 0.0, "p95": 0.0},
                "tracking": {"avg": 0.0, "p50": 0.0, "p95": 0.0},
                "recognition": {"avg": 0.0, "p50": 0.0, "p95": 0.0},
                "render": {"avg": 0.0, "p50": 0.0, "p95": 0.0},
            },
        }
    else:
        summary_payload = {
            "elapsed_s": summary.elapsed_s,
            "fps": summary.fps,
            "stage_ms": summary.stage_ms,
        }

    report: Dict[str, Any] = {
        "schema_version": "phase6.v1",
        "phase": "phase6",
        "started_at": started_at,
        "completed_at": now_utc_iso(),
        "run_id": run_stamp,
        "benchmark_seconds_requested": int(args.seconds),
        "command": process_cmd,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": git_commit,
            "git_branch": git_branch,
        },
        "paths": {
            "source_config": str(src_config),
            "run_config": str(run_config),
            "run_config_sha256": run_config_hash,
            "log": str(log_path),
            "timings_csv": str(run_dir / "timings.csv"),
        },
        "summary": summary_payload,
        "summary_source": summary_source,
        "summary_error": summary_error,
        "quality": quality,
        "transport": {
            "poll_base_url": args.base_url,
            "poll_interval_s": float(args.poll_interval),
            "summary": summarize_transport(samples_snapshot),
            "samples": samples_snapshot,
        },
    }

    if str(args.compare_to).strip():
        compare_path = Path(str(args.compare_to).strip())
        if compare_path.exists() and compare_path.is_file():
            try:
                report["comparison"] = _build_comparison(report, compare_path)
                regression = _evaluate_regression(
                    report,
                    compare_path,
                    p95_tolerance_ms=float(args.p95_regression_tolerance_ms),
                    fps_tolerance=float(args.fps_regression_tolerance),
                )
                report["regression"] = regression
                if not bool(regression.get("valid", True)):
                    quality["valid"] = False
                    merged = list(quality.get("issues", []))
                    merged.extend([f"regression:{issue}" for issue in regression.get("issues", [])])
                    quality["issues"] = sorted(set(merged))
                    report["quality"] = quality
            except Exception as exc:
                report["comparison_error"] = str(exc)
        else:
            report["comparison_error"] = f"compare_to_not_found: {compare_path}"

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print("")
    print(f"Benchmark report written: {report_path}")
    print(f"Benchmark log written: {log_path}")
    print(f"Benchmark config written: {run_config}")
    if quality["valid"]:
        print("Benchmark quality: VALID")
        return 0
    print(f"Benchmark quality: INVALID | issues={'; '.join(quality.get('issues', []))}")
    if args.allow_invalid:
        print("Returning success due to --allow-invalid.")
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
