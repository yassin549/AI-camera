"""Face embedding helper using face_recognition / dlib."""

from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from utils import l2_normalize

LOGGER = logging.getLogger(__name__)

try:
    import face_recognition
except Exception as exc:  # pragma: no cover - import guard
    face_recognition = None
    LOGGER.warning("face_recognition unavailable: %s", exc)

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - import guard
    ort = None
    LOGGER.warning("onnxruntime unavailable for face embedding: %s", exc)


class FaceEmbedder:
    """Wrap face_recognition.face_encodings and return 128-D float32 vectors."""

    def __init__(
        self,
        num_jitters: int = 1,
        model: str = "small",
        backend: str = "auto",
        onnx_model_path: Optional[str] = None,
        onnx_input_size: Tuple[int, int] = (112, 112),
        onnx_mean: float = 0.5,
        onnx_std: float = 0.5,
        onnx_output_name: Optional[str] = None,
        onnx_output_index: int = 0,
        target_dim: int = 128,
    ) -> None:
        self.num_jitters = int(num_jitters)
        self.model = model
        self.target_dim = max(32, int(target_dim))
        self.backend = str(backend or "auto").strip().lower()
        self.onnx_model_path = str(onnx_model_path or "").strip()
        self.onnx_mean = float(onnx_mean)
        self.onnx_std = max(1e-6, float(onnx_std))
        self.onnx_output_name = str(onnx_output_name).strip() if onnx_output_name else None
        self.onnx_output_index = int(onnx_output_index)
        self.onnx_input_size = (max(32, int(onnx_input_size[0])), max(32, int(onnx_input_size[1])))

        self._onnx_session: Optional[ort.InferenceSession] = None
        self._onnx_input_name: Optional[str] = None
        self._onnx_output_name: Optional[str] = None
        self._onnx_layout: str = "nchw"
        self._active_backend: str = "color_fallback"

        self._init_backends()

    def _init_backends(self) -> None:
        prefer_onnx = self.backend in {"auto", "onnx", "onnxruntime"}
        if prefer_onnx and self.onnx_model_path:
            self._try_init_onnx()
            if self._onnx_session is not None:
                self._active_backend = "onnx"
                return

        if self.backend in {"auto", "face_recognition", "dlib"} and face_recognition is not None:
            self._active_backend = "face_recognition"
            return

        self._active_backend = "color_fallback"
        if self.backend in {"onnx", "onnxruntime"} and self._onnx_session is None:
            LOGGER.warning(
                "Face embedder requested ONNX backend but model is unavailable. Falling back to %s.",
                self._active_backend,
            )

    def _try_init_onnx(self) -> None:
        if ort is None:
            return
        if not self.onnx_model_path:
            return
        if not os.path.exists(self.onnx_model_path):
            LOGGER.warning("Face ONNX model path not found: %s", self.onnx_model_path)
            return
        try:
            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(
                self.onnx_model_path,
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            input_meta = session.get_inputs()[0]
            self._onnx_input_name = input_meta.name
            self._resolve_onnx_input_layout(input_meta.shape)

            outputs = session.get_outputs()
            output_names = [o.name for o in outputs]
            if self.onnx_output_name and self.onnx_output_name in output_names:
                self._onnx_output_name = self.onnx_output_name
            elif output_names:
                idx = max(0, min(len(output_names) - 1, self.onnx_output_index))
                self._onnx_output_name = output_names[idx]
            else:
                self._onnx_output_name = None

            self._onnx_session = session
            LOGGER.info(
                "Face embedding backend: onnx (%s) input=%sx%s layout=%s output=%s dim=%s",
                self.onnx_model_path,
                self.onnx_input_size[0],
                self.onnx_input_size[1],
                self._onnx_layout,
                self._onnx_output_name or "default",
                self.target_dim,
            )
        except Exception as exc:
            LOGGER.warning("Failed to initialize face ONNX embedder: %s", exc)
            self._onnx_session = None
            self._onnx_input_name = None
            self._onnx_output_name = None

    def _resolve_onnx_input_layout(self, input_shape: object) -> None:
        shape = list(input_shape) if isinstance(input_shape, (list, tuple)) else []
        if len(shape) >= 4:
            c_first = shape[1]
            c_last = shape[3]
            h = shape[2]
            w = shape[3]
            if isinstance(c_first, int) and c_first == 3:
                self._onnx_layout = "nchw"
                if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                    self.onnx_input_size = (int(w), int(h))
                return
            if isinstance(c_last, int) and c_last == 3:
                self._onnx_layout = "nhwc"
                h_nhwc = shape[1]
                w_nhwc = shape[2]
                if isinstance(w_nhwc, int) and isinstance(h_nhwc, int) and w_nhwc > 0 and h_nhwc > 0:
                    self.onnx_input_size = (int(w_nhwc), int(h_nhwc))
                return
        self._onnx_layout = "nchw"

    def embed_from_bgr(self, bgr_face: np.ndarray) -> Optional[np.ndarray]:
        if bgr_face is None or bgr_face.size == 0:
            return None

        if self._active_backend == "onnx" and self._onnx_session is not None:
            emb = self._embed_from_onnx(bgr_face)
            if emb is not None:
                return emb
            # If ONNX inference fails at runtime, fall through to secondary backend.

        if self._active_backend == "face_recognition" and face_recognition is not None:
            emb = self._embed_from_face_recognition(bgr_face)
            if emb is not None:
                return emb

        return self._embed_from_color_fallback(bgr_face)

    def _embed_from_face_recognition(self, bgr_face: np.ndarray) -> Optional[np.ndarray]:
        # Keep minimum workable resolution for dlib encoder.
        face = bgr_face
        h, w = face.shape[:2]
        if min(h, w) < 64:
            scale = 64.0 / float(min(h, w))
            face = cv2.resize(face, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(
            rgb,
            known_face_locations=None,
            num_jitters=self.num_jitters,
            model=self.model,
        )
        if encodings:
            return self._reshape_to_target(np.asarray(encodings[0], dtype=np.float32))
        return None

    def _embed_from_onnx(self, bgr_face: np.ndarray) -> Optional[np.ndarray]:
        assert self._onnx_session is not None
        input_name = self._onnx_input_name
        if not input_name:
            return None

        rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.onnx_input_size, interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - self.onnx_mean) / self.onnx_std

        if self._onnx_layout == "nhwc":
            tensor = tensor[None, ...]
        else:
            tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        try:
            if self._onnx_output_name:
                raw = self._onnx_session.run([self._onnx_output_name], {input_name: tensor})[0]
            else:
                raw = self._onnx_session.run(None, {input_name: tensor})[0]
        except Exception:
            LOGGER.debug("Face ONNX inference failed", exc_info=True)
            return None
        return self._reshape_to_target(np.asarray(raw, dtype=np.float32))

    def _reshape_to_target(self, vec: np.ndarray) -> np.ndarray:
        flat = np.asarray(vec, dtype=np.float32).reshape(-1)
        if flat.size >= self.target_dim:
            flat = flat[: self.target_dim]
        else:
            flat = np.pad(flat, (0, self.target_dim - flat.size), mode="constant")
        return l2_normalize(flat)

    def _embed_from_color_fallback(self, bgr_face: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)

        # Deterministic fallback if dlib is missing or no face encoding produced.
        resized = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
        vec = resized.astype(np.float32).reshape(-1)
        return self._reshape_to_target(vec)
