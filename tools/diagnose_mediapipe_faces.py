from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path

import cv2
import mediapipe as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MediaPipe diagnostics for multi-face detection on a static image."
    )
    parser.add_argument("--image", required=True, help="Path to an image that contains faces.")
    parser.add_argument(
        "--model-selection",
        type=int,
        default=0,
        choices=[0, 1],
        help="FaceDetection model selection: 0=short-range, 1=full-range.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.3,
        help="FaceDetection min_detection_confidence.",
    )
    parser.add_argument(
        "--check-static-mode",
        action="store_true",
        help="Also run FaceMesh static_image_mode=True/False for comparison.",
    )
    return parser.parse_args()


def pixel_box_from_relative(
    rel_xmin: float, rel_ymin: float, rel_w: float, rel_h: float, width: int, height: int
) -> tuple[int, int, int, int]:
    x1 = max(0, int(rel_xmin * width))
    y1 = max(0, int(rel_ymin * height))
    x2 = min(width, int((rel_xmin + rel_w) * width))
    y2 = min(height, int((rel_ymin + rel_h) * height))
    return x1, y1, x2, y2


def run_face_detection(
    image_bgr, model_selection: int, min_detection_confidence: float
) -> tuple[int, list[dict[str, object]]]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]
    with mp.solutions.face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence,
    ) as detector:
        results = detector.process(rgb)

    output: list[dict[str, object]] = []
    for idx, detection in enumerate(results.detections or []):
        rel_box = detection.location_data.relative_bounding_box
        rel_xmin = float(rel_box.xmin)
        rel_ymin = float(rel_box.ymin)
        rel_w = float(rel_box.width)
        rel_h = float(rel_box.height)
        confidence = float(detection.score[0]) if detection.score else 0.0
        output.append(
            {
                "idx": idx,
                "confidence": confidence,
                "relative_box": (rel_xmin, rel_ymin, rel_w, rel_h),
                "pixel_box": pixel_box_from_relative(rel_xmin, rel_ymin, rel_w, rel_h, w, h),
            }
        )
    return len(output), output


def run_face_mesh_static_mode_check(image_bgr) -> tuple[int, int]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as mesh_static:
        static_results = mesh_static.process(rgb)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as mesh_tracking:
        tracking_results = mesh_tracking.process(rgb)

    static_count = len(static_results.multi_face_landmarks or [])
    tracking_count = len(tracking_results.multi_face_landmarks or [])
    return static_count, tracking_count


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"OpenCV could not decode image: {image_path}")

    signature = inspect.signature(mp.solutions.face_detection.FaceDetection)
    probe = image_bgr[::24, ::24]
    frame_hash = hashlib.md5(probe.tobytes()).hexdigest()[:12]
    mean_bgr = image_bgr.mean(axis=(0, 1))

    count, detections = run_face_detection(
        image_bgr=image_bgr,
        model_selection=args.model_selection,
        min_detection_confidence=args.min_detection_confidence,
    )

    print(f"[MediaPipe FaceDetection signature] {signature}")
    print(f"[Image] path={image_path} shape={image_bgr.shape}")
    print(
        "[Frame diagnostics] hash=%s mean_bgr=(%.1f, %.1f, %.1f)"
        % (frame_hash, float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2]))
    )
    print(
        "[FaceDetection] model_selection=%d min_detection_confidence=%.2f detected_faces=%d"
        % (args.model_selection, args.min_detection_confidence, count)
    )
    for det in detections:
        rel_xmin, rel_ymin, rel_w, rel_h = det["relative_box"]  # type: ignore[misc]
        print(
            "  - idx=%d conf=%.3f rel_bbox=(%.4f, %.4f, %.4f, %.4f) pixel_box=%s"
            % (
                int(det["idx"]),
                float(det["confidence"]),
                rel_xmin,
                rel_ymin,
                rel_w,
                rel_h,
                det["pixel_box"],
            )
        )

    if args.check_static_mode:
        static_count, tracking_count = run_face_mesh_static_mode_check(image_bgr)
        print(
            "[FaceMesh static_mode check] static_image_mode=True faces=%d | static_image_mode=False faces=%d"
            % (static_count, tracking_count)
        )


if __name__ == "__main__":
    main()
