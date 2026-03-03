"""Detector factory to keep backend selection config-driven."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

from detector_rtdetr import RTDETRONNXDetector
from detector_yolo import YOLOv8ONNXDetector

LOGGER = logging.getLogger(__name__)


def _pick_model_path(cfg: Dict[str, object], backend: str) -> str:
    explicit = str(cfg.get("detector_model_path", "") or "").strip()
    if explicit:
        return explicit

    if backend in {"rtdetr", "rtdetr_onnx"}:
        return str(cfg.get("rtdetr_onnx_path", "./models/rtdetr-l_person.onnx"))

    # Default backend path remains YOLO for backward compatibility.
    return str(cfg.get("yolo_onnx_path", "./models/yolov8n_person.onnx"))


def _coerce_body_input_size(raw: object) -> Tuple[int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            width = max(32, int(raw[0]))
            height = max(32, int(raw[1]))
            return width, height
        except Exception:
            pass
    return (128, 256)


def create_person_detector(cfg: Dict[str, object], imgsz: Tuple[int, int]):
    backend = str(cfg.get("detector_backend", "yolo_onnx") or "yolo_onnx").strip().lower()
    model_path = _pick_model_path(cfg, backend)

    if not Path(model_path).exists():
        LOGGER.warning("Configured detector model does not exist: %s", model_path)

    common_kwargs = {
        "model_path": model_path,
        "imgsz": imgsz,
        "conf_threshold": float(cfg.get("person_conf_threshold", 0.50)),
        "iou_threshold": float(cfg.get("person_iou_threshold", 0.50)),
        "disable_body_embedding": bool(cfg.get("disable_body_embedding", True)),
        "body_fallback_model_path": str(cfg.get("body_fallback_model_path", "")) or None,
        "body_fallback_input_size": _coerce_body_input_size(cfg.get("body_fallback_input_size", (128, 256))),
        "body_fallback_target_dim": int(cfg.get("body_fallback_target_dim", 256)),
        "body_fallback_mean": float(cfg.get("body_fallback_mean", 0.5)),
        "body_fallback_std": float(cfg.get("body_fallback_std", 0.5)),
        "body_fallback_output_name": str(cfg.get("body_fallback_output_name", "")) or None,
        "body_fallback_output_index": int(cfg.get("body_fallback_output_index", 0)),
        "onnx_intra_threads": int(cfg.get("onnx_intra_threads", 0) or 0),
        "onnx_inter_threads": int(cfg.get("onnx_inter_threads", 1) or 0),
    }

    if backend in {"rtdetr", "rtdetr_onnx"}:
        try:
            detector = RTDETRONNXDetector(
                **common_kwargs,
                person_class_id=int(cfg.get("detector_person_class_id", 0)),
                output_format=str(cfg.get("detector_output_format", "auto") or "auto"),
                max_detections=int(cfg.get("detector_max_detections", 300)),
            )
            selected_backend = "rtdetr_onnx"
        except Exception as exc:
            LOGGER.warning("Failed to initialize RT-DETR detector (%s), falling back to YOLO ONNX.", exc)
            yolo_path = str(cfg.get("yolo_onnx_path", "./models/yolov8n_person.onnx"))
            detector = YOLOv8ONNXDetector(
                **{
                    **common_kwargs,
                    "model_path": yolo_path,
                },
                feature_output_name=cfg.get("yolo_feature_output_name"),
            )
            selected_backend = "yolo_onnx"
    else:
        detector = YOLOv8ONNXDetector(
            **common_kwargs,
            feature_output_name=cfg.get("yolo_feature_output_name"),
        )
        selected_backend = "yolo_onnx"

    setattr(detector, "detector_backend", selected_backend)
    return detector
