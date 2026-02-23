"""YOLOv8n ONNX person detector optimized for CPU and fixed-shape inference."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from embedder_body import body_descriptor_from_backbone
from utils import iou

LOGGER = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    score: float
    class_id: int = 0
    feature: Optional[np.ndarray] = None


class YOLOv8ONNXDetector:
    """Person-only ONNXRuntime detector with optional intermediate body features."""

    def __init__(
        self,
        model_path: str,
        imgsz: Tuple[int, int] = (640, 360),
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.50,
        feature_output_name: Optional[str] = None,
        disable_body_embedding: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.capture_size = (int(imgsz[0]), int(imgsz[1]))
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.disable_body_embedding = bool(disable_body_embedding)

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2) // 2))
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.det_output_name = self.output_names[0]

        self.model_input_size = self._resolve_model_input_size(self.input_shape, self.capture_size)

        self.feature_output_name: Optional[str] = None
        if not self.disable_body_embedding:
            if feature_output_name and feature_output_name in self.output_names:
                self.feature_output_name = feature_output_name
            elif len(self.output_names) > 1:
                for out in self.session.get_outputs()[1:]:
                    if len(out.shape) == 4:
                        self.feature_output_name = out.name
                        break

        LOGGER.info(
            "Detector input canvas: %sx%s | capture frame: %sx%s",
            self.model_input_size[0],
            self.model_input_size[1],
            self.capture_size[0],
            self.capture_size[1],
        )
        LOGGER.info(
            "Body embedding mode: %s",
            "yolo_backbone" if self.feature_output_name else "disabled",
        )

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

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        tensor, map_info = self._to_model_tensor(frame_bgr)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        out_by_name: Dict[str, np.ndarray] = {k: v for k, v in zip(self.output_names, outputs)}

        raw_det = out_by_name[self.det_output_name]
        backbone = out_by_name.get(self.feature_output_name) if self.feature_output_name else None

        detections = self._postprocess(raw_det, map_info, frame_bgr.shape[1], frame_bgr.shape[0])

        if backbone is not None:
            for det in detections:
                det.feature = body_descriptor_from_backbone(
                    backbone_feature=backbone,
                    det_bbox_input_xyxy=self._map_frame_box_to_model(det.bbox, map_info),
                    model_input_size=self.model_input_size,
                    target_dim=256,
                )

        return detections

    def _to_model_tensor(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Convert capture frame to detector input canvas without resizing when possible.

        Single-resize rule: frame is already resized once in capture thread.
        Here we only pad/crop into model canvas.
        """
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
            # Last resort for mismatched config: center crop to avoid extra resize work.
            x0 = max(0, (fw - tw) // 2)
            y0 = max(0, (fh - th) // 2)
            canvas = frame_bgr[y0:y0 + th, x0:x0 + tw]
            map_info = {"mode": 2.0, "x_off": float(x0), "y_off": float(y0)}

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return tensor, map_info

    def _postprocess(
        self,
        raw: np.ndarray,
        map_info: Dict[str, float],
        frame_w: int,
        frame_h: int,
    ) -> List[Detection]:
        pred = np.asarray(raw)
        pred = np.squeeze(pred)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if pred.ndim != 2:
            return []

        # Exported YOLOv8 ONNX typically [84,8400] -> transpose to [8400,84]
        if pred.shape[0] <= 96 and pred.shape[1] > pred.shape[0]:
            pred = pred.T

        boxes_model: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []

        cols = pred.shape[1]
        if cols >= 84:
            # [cx,cy,w,h, class0, class1, ...]
            xywh = pred[:, :4]
            cls_person = pred[:, 4]  # class 0: person
            keep = cls_person >= self.conf_threshold
            xywh = xywh[keep]
            cls_person = cls_person[keep]
            tw, th = self.model_input_size

            for row, score in zip(xywh, cls_person):
                cx, cy, bw, bh = [float(v) for v in row]
                x1 = int(max(0.0, cx - bw / 2.0))
                y1 = int(max(0.0, cy - bh / 2.0))
                x2 = int(min(float(tw), cx + bw / 2.0))
                y2 = int(min(float(th), cy + bh / 2.0))
                if x2 > x1 and y2 > y1:
                    boxes_model.append((x1, y1, x2, y2))
                    scores.append(float(score))
        elif cols >= 6:
            # Decoded format [x1,y1,x2,y2,score,cls]
            for row in pred:
                cls = int(row[5])
                if cls != 0:
                    continue
                score = float(row[4])
                if score < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = [int(v) for v in row[:4]]
                if x2 > x1 and y2 > y1:
                    boxes_model.append((x1, y1, x2, y2))
                    scores.append(score)

        keep_idx = self._nms(boxes_model, scores, self.iou_threshold)

        out: List[Detection] = []
        for i in keep_idx:
            x1, y1, x2, y2 = self._map_model_box_to_frame(boxes_model[i], map_info, frame_w, frame_h)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(Detection(bbox=(x1, y1, x2, y2), score=float(scores[i]), class_id=0, feature=None))
        return out

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
            # Padded model canvas -> remove offsets.
            x1 -= x_off
            x2 -= x_off
            y1 -= y_off
            y2 -= y_off
        elif mode == 2:
            # Center crop -> add crop offsets back.
            x1 += x_off
            x2 += x_off
            y1 += y_off
            y2 += y_off

        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(x1 + 1, min(frame_w, x2))
        y2 = max(y1 + 1, min(frame_h, y2))
        return x1, y1, x2, y2

    def _map_frame_box_to_model(
        self,
        bbox: Tuple[int, int, int, int],
        map_info: Dict[str, float],
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        mode = int(map_info["mode"])
        x_off = int(map_info["x_off"])
        y_off = int(map_info["y_off"])

        if mode == 1:
            return (x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off)
        if mode == 2:
            return (x1 - x_off, y1 - y_off, x2 - x_off, y2 - y_off)
        return bbox

    @staticmethod
    def _nms(
        boxes: Sequence[Tuple[int, int, int, int]],
        scores: Sequence[float],
        iou_thr: float,
    ) -> List[int]:
        if not boxes:
            return []
        order = np.argsort(np.asarray(scores, dtype=np.float32))[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            survivors = []
            for j in rest:
                if iou(boxes[i], boxes[int(j)]) < iou_thr:
                    survivors.append(int(j))
            order = np.asarray(survivors, dtype=np.int32)
        return keep
