"""RTSP/video capture worker with reconnect/backoff and single resize per frame."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional, Tuple

import cv2

from utils import LatestFrameStore, ensure_rtsp_tcp

LOGGER = logging.getLogger(__name__)


class CaptureWorker:
    """Capture thread that publishes only the latest resized frame."""

    def __init__(
        self,
        source: str,
        imgsz: Tuple[int, int],
        frame_store: LatestFrameStore,
        stop_event: threading.Event,
    ) -> None:
        self.source = str(source)
        self._imgsz_lock = threading.Lock()
        self._imgsz = (max(1, int(imgsz[0])), max(1, int(imgsz[1])))
        self.frame_store = frame_store
        self.stop_event = stop_event
        self.capture_buffer = max(1, int(os.getenv("AICAM_CAPTURE_BUFFER", "1")))
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id = 0

    def set_imgsz(self, imgsz: Tuple[int, int]) -> None:
        next_size = (max(1, int(imgsz[0])), max(1, int(imgsz[1])))
        with self._imgsz_lock:
            if self._imgsz == next_size:
                return
            self._imgsz = next_size
        LOGGER.info("Capture resize target updated: %sx%s", next_size[0], next_size[1])

    def get_imgsz(self) -> Tuple[int, int]:
        with self._imgsz_lock:
            return self._imgsz

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="capture-worker")
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _open_capture(self) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        src = self.source
        if src.lower().startswith("rtsp://"):
            src = ensure_rtsp_tcp(src)
            # Lower-latency FFmpeg options for RTSP ingest via OpenCV.
            if not os.getenv("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"
                )

        self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.capture_buffer)
        LOGGER.info("RTSP connected: %s", src)
        return True

    def _run(self) -> None:
        delays = [0.5, 1.0, 2.0, 5.0]
        retry_idx = 0

        while not self.stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._open_capture():
                    delay = delays[min(retry_idx, len(delays) - 1)]
                    LOGGER.warning("Capture connect failed, retrying in %.1fs", delay)
                    retry_idx = min(retry_idx + 1, len(delays) - 1)
                    time.sleep(delay)
                    continue
                retry_idx = 0

            ok, frame = self._cap.read()
            if not ok or frame is None:
                delay = delays[min(retry_idx, len(delays) - 1)]
                LOGGER.warning("Capture read failed, reconnecting in %.1fs", delay)
                retry_idx = min(retry_idx + 1, len(delays) - 1)
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                time.sleep(delay)
                continue

            retry_idx = 0
            # Single resize point for the full pipeline.
            target_w, target_h = self.get_imgsz()
            frame_h, frame_w = frame.shape[:2]
            if frame_w == target_w and frame_h == target_h:
                resized = frame
            else:
                resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            self.frame_store.publish(frame_id=self._frame_id, frame=resized, ts=time.time())
            self._frame_id += 1
