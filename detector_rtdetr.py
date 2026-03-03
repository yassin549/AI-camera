"""RT-DETR ONNX person detector with optional body-embedding fallback."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from detector_yolo import Detection
from embedder_body import MobileNetBodyFallback

LOGGER = logging.getLogger(__name__)


class RTDETRONNXDetector:
    """RT-DETR ONNX detector that returns person detections in frame coordinates."""

    def __init__(
        self,
        model_path: str,
        imgsz: Tuple[int, int] = (640, 360),
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.50,
        person_class_id: int = 0,
        output_format: str = "auto",
        max_detections: int = 300,
        disable_body_embedding: bool = True,
        body_fallback_model_path: Optional[str] = None,
        body_fallback_input_size: Tuple[int, int] = (128, 256),
        body_fallback_target_dim: int = 256,
        body_fallback_mean: float = 0.5,
        body_fallback_std: float = 0.5,
        body_fallback_output_name: Optional[str] = None,
        body_fallback_output_index: int = 0,
        onnx_intra_threads: Optional[int] = None,
        onnx_inter_threads: Optional[int] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.capture_size = (int(imgsz[0]), int(imgsz[1]))
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.person_class_id = int(person_class_id)
        self.output_format = str(output_format or "auto").strip().lower()
        if self.output_format not in {"auto", "xyxy_score_cls", "xyxy_logits"}:
            self.output_format = "auto"
        self.max_detections = max(1, int(max_detections))
        self.disable_body_embedding = bool(disable_body_embedding)
        self.body_feature_dim = max(16, int(body_fallback_target_dim))

        intra_threads = self._resolve_thread_count(
            explicit=onnx_intra_threads,
            env_var="AICAM_ORT_INTRA_THREADS",
            fallback=max(1, min(6, (os.cpu_count() or 2) // 2)),
        )
        inter_threads = self._resolve_thread_count(
            explicit=onnx_inter_threads,
            env_var="AICAM_ORT_INTER_THREADS",
            fallback=1,
        )
        execution_mode_label, execution_mode = self._resolve_execution_mode(
            str(os.getenv("AICAM_ORT_EXECUTION_MODE", "sequential"))
        )
        graph_opt_label, graph_opt_level = self._resolve_graph_opt_level(
            str(os.getenv("AICAM_ORT_GRAPH_OPT_LEVEL", "all"))
        )
        enable_mem_pattern = self._resolve_bool_env("AICAM_ORT_ENABLE_MEM_PATTERN", True)
        enable_cpu_mem_arena = self._resolve_bool_env("AICAM_ORT_ENABLE_CPU_MEM_ARENA", True)
        intra_spin = self._resolve_bool_env("AICAM_ORT_INTRA_SPIN", True)
        inter_spin = self._resolve_bool_env("AICAM_ORT_INTER_SPIN", True)

        so = ort.SessionOptions()
        if intra_threads > 0:
            so.intra_op_num_threads = intra_threads
        if inter_threads > 0:
            so.inter_op_num_threads = inter_threads
        so.execution_mode = execution_mode
        so.graph_optimization_level = graph_opt_level
        so.enable_mem_pattern = bool(enable_mem_pattern)
        so.enable_cpu_mem_arena = bool(enable_cpu_mem_arena)
        try:
            so.add_session_config_entry("session.intra_op.allow_spinning", "1" if intra_spin else "0")
            so.add_session_config_entry("session.inter_op.allow_spinning", "1" if inter_spin else "0")
        except Exception:
            LOGGER.debug("Could not apply ORT spinning config entries", exc_info=True)

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.model_input_size = self._resolve_model_input_size(self.input_shape, self.capture_size)

        self.body_fallback: Optional[MobileNetBodyFallback] = None
        if not self.disable_body_embedding:
            self.body_fallback = MobileNetBodyFallback(
                model_path=body_fallback_model_path,
                input_size=body_fallback_input_size,
                target_dim=self.body_feature_dim,
                mean=float(body_fallback_mean),
                std=float(body_fallback_std),
                output_name=body_fallback_output_name,
                output_index=int(body_fallback_output_index),
            )

        LOGGER.info(
            "RT-DETR detector input canvas: %sx%s | capture frame: %sx%s | output_format=%s",
            self.model_input_size[0],
            self.model_input_size[1],
            self.capture_size[0],
            self.capture_size[1],
            self.output_format,
        )
        LOGGER.info(
            "ONNXRuntime threads: intra=%s inter=%s",
            intra_threads if intra_threads > 0 else "default",
            inter_threads if inter_threads > 0 else "default",
        )
        LOGGER.info(
            "ONNXRuntime tuning: mode=%s graph_opt=%s mem_pattern=%s cpu_mem_arena=%s intra_spin=%s inter_spin=%s",
            execution_mode_label,
            graph_opt_label,
            str(enable_mem_pattern).lower(),
            str(enable_cpu_mem_arena).lower(),
            str(intra_spin).lower(),
            str(inter_spin).lower(),
        )
        LOGGER.info(
            "Body embedding mode: %s",
            "crop_fallback" if self.body_fallback else "disabled",
        )

    @staticmethod
    def _resolve_thread_count(explicit: Optional[int], env_var: str, fallback: int) -> int:
        if explicit is not None:
            try:
                return max(0, int(explicit))
            except Exception:
                return max(0, int(fallback))
        raw = str(os.getenv(env_var, "")).strip()
        if not raw:
            return max(0, int(fallback))
        try:
            return max(0, int(raw))
        except Exception:
            return max(0, int(fallback))

    @staticmethod
    def _resolve_bool_env(env_var: str, fallback: bool) -> bool:
        raw = str(os.getenv(env_var, "")).strip().lower()
        if not raw:
            return bool(fallback)
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        return bool(fallback)

    @staticmethod
    def _resolve_execution_mode(raw: str) -> Tuple[str, ort.ExecutionMode]:
        normalized = str(raw).strip().lower()
        if normalized in {"parallel", "ort_parallel"}:
            return "parallel", ort.ExecutionMode.ORT_PARALLEL
        return "sequential", ort.ExecutionMode.ORT_SEQUENTIAL

    @staticmethod
    def _resolve_graph_opt_level(raw: str) -> Tuple[str, ort.GraphOptimizationLevel]:
        normalized = str(raw).strip().lower()
        if normalized in {"disable", "disabled", "0", "off"}:
            return "disable", ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if normalized in {"basic", "1"}:
            return "basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        if normalized in {"extended", "2"}:
            return "extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        return "all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    @staticmethod
    def _resolve_model_input_size(
        input_shape: Sequence[object],
        fallback: Tuple[int, int],
    ) -> Tuple[int, int]:
        if len(input_shape) >= 4:
            h = input_shape[2]
            w = input_shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                return int(w), int(h)
        return fallback

    def set_capture_size(self, imgsz: Tuple[int, int]) -> None:
        self.capture_size = (max(1, int(imgsz[0])), max(1, int(imgsz[1])))

    def warmup(self, runs: int = 2, capture_sizes: Optional[Sequence[Tuple[int, int]]] = None) -> None:
        run_count = max(0, int(runs))
        if run_count <= 0:
            return
        warmup_sizes: List[Tuple[int, int]] = []
        if capture_sizes:
            for size in capture_sizes:
                if not isinstance(size, (tuple, list)) or len(size) < 2:
                    continue
                w = max(1, int(size[0]))
                h = max(1, int(size[1]))
                warmup_sizes.append((w, h))
        if not warmup_sizes:
            warmup_sizes = [self.capture_size]

        seen: set[Tuple[int, int]] = set()
        deduped: List[Tuple[int, int]] = []
        for size in warmup_sizes:
            if size in seen:
                continue
            seen.add(size)
            deduped.append(size)

        for width, height in deduped:
            dummy = np.zeros((height, width, 3), dtype=np.uint8)
            for _ in range(run_count):
                self.detect(dummy)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        tensor, map_info = self._to_model_tensor(frame_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        boxes_model, scores, class_ids = self._decode_outputs(outputs)
        if boxes_model.shape[0] == 0:
            return []

        person_mask = (class_ids == self.person_class_id) & (scores >= self.conf_threshold)
        if not np.any(person_mask):
            return []

        boxes_model = boxes_model[person_mask]
        scores = scores[person_mask]
        if boxes_model.shape[0] > self.max_detections:
            order = np.argsort(scores)[::-1][: self.max_detections]
            boxes_model = boxes_model[order]
            scores = scores[order]

        keep_idx = self._nms(boxes_model, scores, self.iou_threshold)

        out: List[Detection] = []
        for i in keep_idx:
            box = boxes_model[int(i)]
            x1, y1, x2, y2 = self._map_model_box_to_frame(
                (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                map_info,
                frame_bgr.shape[1],
                frame_bgr.shape[0],
            )
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    score=float(scores[int(i)]),
                    class_id=self.person_class_id,
                    feature=None,
                )
            )

        if not self.disable_body_embedding and self.body_fallback is not None:
            for det in out:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                crop = frame_bgr[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
                det.feature = self.body_fallback.embed(crop)

        return out

    def _decode_outputs(self, outputs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Try explicit (boxes, scores) style heads first.
        boxes_from_pair, scores_from_pair, cls_from_pair = self._decode_boxes_scores_pair(outputs)
        if boxes_from_pair.shape[0] > 0:
            return boxes_from_pair, scores_from_pair, cls_from_pair

        # Fallback: parse each output independently, select the richest candidate.
        candidates: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for output in outputs:
            decoded = self._decode_single_output(output)
            if decoded[0].shape[0] > 0:
                candidates.append(decoded)
        if not candidates:
            empty = np.zeros((0, 4), dtype=np.int32)
            return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        # Prefer higher-confidence candidates, then higher cardinality.
        def _rank(item: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[float, int]:
            boxes, scores, _ = item
            best = float(np.max(scores)) if scores.size else 0.0
            return best, int(boxes.shape[0])

        best = max(candidates, key=_rank)
        return best

    def _decode_boxes_scores_pair(self, outputs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrices = [self._to_matrix(out) for out in outputs]
        matrices = [m for m in matrices if m is not None and m.shape[0] > 0]
        if len(matrices) < 2:
            empty = np.zeros((0, 4), dtype=np.int32)
            return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        boxes_mat: Optional[np.ndarray] = None
        score_mat: Optional[np.ndarray] = None
        for mat in matrices:
            if mat.shape[1] == 4 and boxes_mat is None:
                boxes_mat = mat
                continue
            if mat.shape[1] >= 1:
                score_mat = mat if score_mat is None else score_mat

        if boxes_mat is None or score_mat is None:
            empty = np.zeros((0, 4), dtype=np.int32)
            return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        n = min(boxes_mat.shape[0], score_mat.shape[0])
        boxes = boxes_mat[:n].astype(np.float32, copy=False)
        score_block = score_mat[:n].astype(np.float32, copy=False)
        if score_block.shape[1] == 1:
            scores = self._normalize_scores(score_block[:, 0])
            class_ids = np.full((n,), self.person_class_id, dtype=np.int32)
        else:
            class_scores = self._normalize_class_scores(score_block)
            class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
            scores = class_scores[np.arange(n), class_ids]

        boxes_int = self._sanitize_model_boxes(boxes)
        return boxes_int, scores.astype(np.float32), class_ids

    def _decode_single_output(self, output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mat = self._to_matrix(output)
        if mat is None or mat.shape[0] == 0 or mat.shape[1] < 5:
            empty = np.zeros((0, 4), dtype=np.int32)
            return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        matrix = mat.astype(np.float32, copy=False)
        boxes = matrix[:, :4]

        if self.output_format == "xyxy_score_cls":
            scores = self._normalize_scores(matrix[:, 4])
            class_ids = np.rint(matrix[:, 5] if matrix.shape[1] > 5 else 0.0).astype(np.int32)
            return self._sanitize_model_boxes(boxes), scores, class_ids

        if self.output_format == "xyxy_logits":
            class_scores = self._normalize_class_scores(matrix[:, 4:])
            class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
            scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
            return self._sanitize_model_boxes(boxes), scores.astype(np.float32), class_ids

        # auto mode: detect whether col[5] looks like class-id column.
        if matrix.shape[1] >= 6 and self._looks_like_class_column(matrix[:, 5]):
            scores = self._normalize_scores(matrix[:, 4])
            class_ids = np.rint(matrix[:, 5]).astype(np.int32)
            return self._sanitize_model_boxes(boxes), scores, class_ids

        if matrix.shape[1] >= 6:
            class_scores = self._normalize_class_scores(matrix[:, 4:])
            class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
            scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
            return self._sanitize_model_boxes(boxes), scores.astype(np.float32), class_ids

        # Unknown single-score output: treat as person-only.
        scores = self._normalize_scores(matrix[:, 4])
        class_ids = np.full((matrix.shape[0],), self.person_class_id, dtype=np.int32)
        return self._sanitize_model_boxes(boxes), scores, class_ids

    def _to_model_tensor(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        fh, fw = frame_bgr.shape[:2]
        tw, th = self.model_input_size

        if fw == tw and fh == th:
            canvas = frame_bgr
            map_info = {"mode": 0.0, "x_off": 0.0, "y_off": 0.0}
        elif fw <= tw and fh <= th:
            canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
            x_off = (tw - fw) // 2
            y_off = (th - fh) // 2
            canvas[y_off:y_off + fh, x_off:x_off + fw] = frame_bgr
            map_info = {"mode": 1.0, "x_off": float(x_off), "y_off": float(y_off)}
        else:
            x0 = max(0, (fw - tw) // 2)
            y0 = max(0, (fh - th) // 2)
            canvas = frame_bgr[y0:y0 + th, x0:x0 + tw]
            map_info = {"mode": 2.0, "x_off": float(x0), "y_off": float(y0)}

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return tensor, map_info

    def _sanitize_model_boxes(self, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.int32)

        parsed = boxes.astype(np.float32, copy=False)
        tw, th = self.model_input_size

        # Normalized coordinates are common in exported DETR heads.
        if float(np.max(np.abs(parsed))) <= 2.0:
            parsed = parsed.copy()
            parsed[:, [0, 2]] *= float(tw)
            parsed[:, [1, 3]] *= float(th)

        x1 = parsed[:, 0]
        y1 = parsed[:, 1]
        x2 = parsed[:, 2]
        y2 = parsed[:, 3]
        invalid = (x2 <= x1) | (y2 <= y1)
        if np.mean(invalid.astype(np.float32)) > 0.60:
            # Some exports represent boxes as cx,cy,w,h; convert when most rows are invalid.
            cx = parsed[:, 0]
            cy = parsed[:, 1]
            bw = np.maximum(0.0, parsed[:, 2])
            bh = np.maximum(0.0, parsed[:, 3])
            x1 = cx - (bw * 0.5)
            y1 = cy - (bh * 0.5)
            x2 = cx + (bw * 0.5)
            y2 = cy + (bh * 0.5)

        x1 = np.clip(x1, 0.0, float(max(0, tw - 1)))
        y1 = np.clip(y1, 0.0, float(max(0, th - 1)))
        x2 = np.clip(x2, 1.0, float(max(1, tw)))
        y2 = np.clip(y2, 1.0, float(max(1, th)))
        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            return np.zeros((0, 4), dtype=np.int32)

        out = np.stack([x1[valid], y1[valid], x2[valid], y2[valid]], axis=1)
        return out.astype(np.int32)

    @staticmethod
    def _to_matrix(arr: np.ndarray) -> Optional[np.ndarray]:
        pred = np.asarray(arr)
        if pred.size == 0:
            return None
        if pred.ndim >= 3 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.ndim == 3 and pred.shape[0] > 1:
            pred = pred[0]
        pred = np.squeeze(pred)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if pred.ndim != 2:
            return None
        # Some exporters return [K, N] for logits-like tensors; transpose if needed.
        if pred.shape[0] <= 96 and pred.shape[1] >= 256 and pred.shape[1] > pred.shape[0]:
            pred = pred.T
        return pred.astype(np.float32, copy=False)

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        out = np.asarray(scores, dtype=np.float32).reshape(-1)
        if out.size == 0:
            return out
        max_v = float(np.max(out))
        min_v = float(np.min(out))
        if min_v < 0.0 or max_v > 1.0:
            clipped = np.clip(out, -50.0, 50.0)
            out = 1.0 / (1.0 + np.exp(-clipped))
        return out.astype(np.float32, copy=False)

    @staticmethod
    def _normalize_class_scores(class_scores: np.ndarray) -> np.ndarray:
        raw = np.asarray(class_scores, dtype=np.float32)
        if raw.size == 0:
            return raw
        max_v = float(np.max(raw))
        min_v = float(np.min(raw))
        if min_v < 0.0 or max_v > 1.0:
            shifted = raw - np.max(raw, axis=1, keepdims=True)
            shifted = np.clip(shifted, -50.0, 50.0)
            exp = np.exp(shifted)
            denom = np.sum(exp, axis=1, keepdims=True) + 1e-9
            return (exp / denom).astype(np.float32, copy=False)
        return raw.astype(np.float32, copy=False)

    @staticmethod
    def _looks_like_class_column(values: np.ndarray) -> bool:
        col = np.asarray(values, dtype=np.float32).reshape(-1)
        if col.size == 0:
            return False
        rounded = np.rint(col)
        close = np.mean((np.abs(col - rounded) <= 1e-3).astype(np.float32))
        finite = bool(np.isfinite(col).all())
        if not finite:
            return False
        if float(np.max(np.abs(col))) > 10000.0:
            return False
        return close >= 0.90

    def _map_model_box_to_frame(
        self,
        bbox: Tuple[int, int, int, int],
        map_info: Dict[str, float],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        mode = int(map_info["mode"])
        x_off = int(map_info["x_off"])
        y_off = int(map_info["y_off"])

        if mode == 1:
            x1 -= x_off
            x2 -= x_off
            y1 -= y_off
            y2 -= y_off
        elif mode == 2:
            x1 += x_off
            x2 += x_off
            y1 += y_off
            y2 += y_off

        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(x1 + 1, min(frame_w, x2))
        y2 = max(y1 + 1, min(frame_h, y2))
        return x1, y1, x2, y2

    @staticmethod
    def _nms(
        boxes: Sequence[Tuple[int, int, int, int]] | np.ndarray,
        scores: Sequence[float] | np.ndarray,
        iou_thr: float,
    ) -> List[int]:
        boxes_arr = np.asarray(boxes, dtype=np.float32)
        scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        if boxes_arr.size == 0 or scores_arr.size == 0:
            return []
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = np.argsort(scores_arr)[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            union = areas[i] + areas[rest] - inter + 1e-10
            overlap = inter / union
            order = rest[np.where(overlap < float(iou_thr))[0]]
        return keep
