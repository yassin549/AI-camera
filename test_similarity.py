from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from detector import FaceDetector
from embedder import FaceEmbedder
from utils import cosine_similarity

__test__ = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity between two face images."
    )
    parser.add_argument("image_a", help="Path to first image.")
    parser.add_argument("image_b", help="Path to second image.")
    parser.add_argument("--padding-ratio", type=float, default=0.2)
    return parser.parse_args()


def _largest_face(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    return max(boxes, key=lambda b: max(1, (b[2] - b[0]) * (b[3] - b[1])))


def encode_one_face(
    image_path: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    boxes = detector.detect_faces(img)
    if not boxes:
        raise RuntimeError(f"No face detected in {image_path}.")
    target = _largest_face(boxes)
    emb = embedder.compute_embedding(img, target)
    if emb is None:
        raise RuntimeError(f"face_encodings returned no embedding for {image_path}.")
    print(
        f"{Path(image_path).name}: shape={emb.shape} dtype={emb.dtype} "
        f"norm={float(np.linalg.norm(emb)):.6f} face_box={target}"
    )
    return emb


def main() -> None:
    args = parse_args()
    detector = FaceDetector()
    embedder = FaceEmbedder(padding_ratio=args.padding_ratio)
    try:
        emb_a = encode_one_face(args.image_a, detector, embedder)
        emb_b = encode_one_face(args.image_b, detector, embedder)
    finally:
        detector.close()

    sim = cosine_similarity(emb_a, emb_b)
    print(f"cosine_similarity={sim:.6f}")
    print(f"match@0.60={sim >= 0.60}")
    print(f"match@0.80={sim >= 0.80}")


if __name__ == "__main__":
    main()
