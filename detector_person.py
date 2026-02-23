from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tracker_adapter import Detection


@dataclass
class DetectorConfig:
    backend: str = "onnx"
    model_path: str = "./models/yolov8n_person.onnx"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.50
    input_size: Tuple[int, int] = (640, 640)
    onnx_threads: int = 4


class PersonDetector:
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        raise NotImplementedError


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return keep


class PersonDetectorONNX(PersonDetector):
    """ONNXRuntime YOLOv8 person detector (class 0 = person)."""

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError("onnxruntime is required for ONNX detector backend") from exc

        if not os.path.exists(cfg.model_path):
            raise FileNotFoundError(f"Person detector model not found: {cfg.model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(max(1, cfg.onnx_threads))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            cfg.model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        in_shape = self.session.get_inputs()[0].shape
        if len(in_shape) >= 4 and isinstance(in_shape[2], int) and isinstance(in_shape[3], int):
            self.input_hw = (int(in_shape[2]), int(in_shape[3]))
        else:
            self.input_hw = cfg.input_size

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, float]:
        h, w = frame_bgr.shape[:2]
        ih, iw = self.input_hw
        resized = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]
        sx = float(w / iw)
        sy = float(h / ih)
        return blob, sx, sy

    def _decode(self, output: np.ndarray, sx: float, sy: float) -> List[Detection]:
        pred = np.asarray(output)

        if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
            pred = np.transpose(pred, (0, 2, 1))
        pred = pred[0]

        if pred.shape[1] < 6:
            return []

        if pred.shape[1] > 6:
            obj = pred[:, 4]
            cls = pred[:, 5:]
            cls_id = np.argmax(cls, axis=1)
            cls_score = cls[np.arange(cls.shape[0]), cls_id]
            scores = obj * cls_score
        else:
            cls_id = pred[:, 5].astype(np.int32)
            scores = pred[:, 4]

        person_mask = cls_id == 0
        scores = scores[person_mask]
        boxes = pred[person_mask, :4]

        conf_mask = scores >= float(self.cfg.confidence_threshold)
        scores = scores[conf_mask]
        boxes = boxes[conf_mask]

        if boxes.size == 0:
            return []

        xyxy = np.zeros_like(boxes, dtype=np.float32)
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * sx
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * sy
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * sx
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * sy

        keep = _nms(xyxy, scores, float(self.cfg.iou_threshold))
        out: List[Detection] = []
        for idx in keep:
            x1, y1, x2, y2 = xyxy[idx].tolist()
            out.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    score=float(scores[idx]),
                    class_id=0,
                )
            )
        return out

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        inp, sx, sy = self._preprocess(frame_bgr)
        outputs = self.session.run(None, {self.input_name: inp})
        return self._decode(outputs[0], sx, sy)


class PersonDetectorUltralytics(PersonDetector):
    def __init__(self, cfg: DetectorConfig) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("ultralytics is required for pytorch backend") from exc

        model_path = cfg.model_path
        self.model = YOLO(model_path)
        self.cfg = cfg

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        res = self.model.predict(
            source=frame_bgr,
            conf=float(self.cfg.confidence_threshold),
            iou=float(self.cfg.iou_threshold),
            classes=[0],
            verbose=False,
            device="cpu",
            imgsz=max(self.cfg.input_size),
        )
        out: List[Detection] = []
        if not res:
            return out
        boxes = res[0].boxes
        if boxes is None:
            return out
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(np.float32)
            out.append(
                Detection(
                    bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    score=float(b.conf.item()),
                    class_id=0,
                )
            )
        return out


class PersonDetectorMotion(PersonDetector):
    """Very fast fallback for fixed cameras (background subtraction)."""

    def __init__(self, min_area: int = 1800) -> None:
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=24,
            detectShadows=False,
        )
        self.min_area = int(min_area)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        fg = self.bg.apply(frame_bgr)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h < 50 or w < 24:
                continue
            out.append(Detection(bbox=(int(x), int(y), int(x + w), int(y + h)), score=0.50, class_id=0))

        if len(out) > 30:
            out = sorted(out, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)[:30]
        return out


class PersonDetectorHOG(PersonDetector):
    """Fallback detector when ONNX/Ultralytics models are unavailable."""

    def __init__(self) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        rects, weights = self.hog.detectMultiScale(
            frame_bgr,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        out: List[Detection] = []
        for (x, y, w, h), score in zip(rects, weights):
            out.append(Detection(bbox=(int(x), int(y), int(x + w), int(y + h)), score=float(score), class_id=0))
        return out


def create_person_detector(
    backend: str = "onnx",
    model_path: str = "./models/yolov8n_person.onnx",
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.50,
    input_size: Sequence[int] = (640, 640),
    onnx_threads: int = 4,
) -> PersonDetector:
    cfg = DetectorConfig(
        backend=backend,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        input_size=(int(input_size[0]), int(input_size[1])),
        onnx_threads=int(onnx_threads),
    )

    choice = str(backend).strip().lower()
    if choice in {"torch", "ultralytics", "pytorch"}:
        try:
            return PersonDetectorUltralytics(cfg)
        except Exception:
            return PersonDetectorMotion()

    try:
        return PersonDetectorONNX(cfg)
    except Exception:
        return PersonDetectorMotion()
