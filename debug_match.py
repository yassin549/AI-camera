from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from embedder_body import create_body_embedder
from embedder_face import FaceEmbedder
from utils import cosine_similarity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare face/body similarities between two images")
    p.add_argument("image_a", help="Path to first image")
    p.add_argument("image_b", help="Path to second image")
    p.add_argument("--body-model", default="./models/osnet_x0_25.onnx", help="Body ONNX model path")
    p.add_argument("--onnx-threads", type=int, default=4, help="ONNX Runtime CPU threads")
    return p.parse_args()


def _load(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return img


def main() -> None:
    args = parse_args()

    img_a = _load(args.image_a)
    img_b = _load(args.image_b)

    face = FaceEmbedder()
    body = create_body_embedder(model_path=args.body_model, onnx_threads=args.onnx_threads)

    t0 = time.perf_counter()
    face_a = face.compute_embedding_from_image(img_a)
    t_face_a = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    face_b = face.compute_embedding_from_image(img_b)
    t_face_b = (time.perf_counter() - t1) * 1000.0

    full_box_a = (0, 0, img_a.shape[1], img_a.shape[0])
    full_box_b = (0, 0, img_b.shape[1], img_b.shape[0])

    t2 = time.perf_counter()
    body_a = body.compute_embedding(img_a, full_box_a)
    t_body_a = (time.perf_counter() - t2) * 1000.0

    t3 = time.perf_counter()
    body_b = body.compute_embedding(img_b, full_box_b)
    t_body_b = (time.perf_counter() - t3) * 1000.0

    face_sim = None
    if face_a is not None and face_b is not None:
        face_sim = cosine_similarity(face_a, face_b)

    body_sim = None
    if body_a is not None and body_b is not None:
        body_sim = cosine_similarity(body_a, body_b)

    print("Face model timing (ms):")
    print(f"  image_a: {t_face_a:.2f}")
    print(f"  image_b: {t_face_b:.2f}")
    print("Body model timing (ms):")
    print(f"  image_a: {t_body_a:.2f}")
    print(f"  image_b: {t_body_b:.2f}")

    print("Similarities:")
    print(f"  face_vs_face: {face_sim if face_sim is not None else 'N/A'}")
    print(f"  body_vs_body: {body_sim if body_sim is not None else 'N/A'}")


if __name__ == "__main__":
    main()
