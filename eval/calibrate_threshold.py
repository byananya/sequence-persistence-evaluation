#!/usr/bin/env python3
"""Face-persistence threshold calibration study for Sequence Persistence Score.

This script measures the separation between same-person face-embedding similarities
(source-to-output and output-to-output) and different-person similarities
(source-to-negative-portraits). It reports descriptive statistics, d-prime, and a
face-weight sensitivity sweep that can be used to justify the 0.65/0.70 thresholds.
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate face-persistence thresholds for the SPS pipeline."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the source portrait image.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        required=True,
        help="Directory containing generated video clips for one sequence.",
    )
    parser.add_argument(
        "--negatives",
        type=Path,
        required=True,
        help="Directory containing portrait images of different identities.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=5,
        help="Number of evenly spaced frames to sample per video (default: 5).",
    )
    parser.add_argument(
        "--style-score",
        type=float,
        default=0.0,
        help="Placeholder mean style score for the sensitivity sweep.",
    )
    parser.add_argument(
        "--sps-threshold",
        type=float,
        default=0.70,
        help="SPS pass threshold for the sensitivity sweep (default: 0.70).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for frame sampling.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/calibration.csv"),
        help="Path to the raw per-pair similarity CSV (default: results/calibration.csv).",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("models"),
        help="Root directory for downloaded InsightFace models (default: models).",
    )
    return parser.parse_args()


def load_model(model_root: Path) -> FaceAnalysis:
    """Load the InsightFace buffalo_l model used for face embeddings."""
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(model_root),
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def best_face(faces: list) -> Optional[object]:
    """Return the face with the highest detection score, or None."""
    if not faces:
        return None
    return max(faces, key=lambda f: float(f.det_score))


def extract_image_embedding(
    app: FaceAnalysis,
    image_path: Path,
    strict: bool = True,
) -> Optional[np.ndarray]:
    """Extract the face embedding from a single portrait image.

    Strict images (source and negatives) require exactly one face.
    Frames may use the highest-det_score face if multiple are detected.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Could not load image %s", image_path)
        return None

    faces = app.get(img)
    if not faces:
        logger.warning("No face detected in %s", image_path)
        return None

    if strict and len(faces) > 1:
        logger.warning(
            "Multiple faces detected in %s (%d); skipping strict image",
            image_path,
            len(faces),
        )
        return None

    face = best_face(faces)
    if face is None:
        return None

    if len(faces) > 1:
        logger.info(
            "Using highest-det_score face (%.3f) in %s",
            float(face.det_score),
            image_path,
        )

    return face.normed_embedding


def sample_frame_indices(total_frames: int, n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Return stratified, evenly-spread frame indices for a video.

    If the video has fewer frames than requested, all frames are returned.
    """
    if total_frames <= 0:
        return np.array([], dtype=int)
    if total_frames <= n_frames:
        return np.arange(total_frames, dtype=int)

    edges = np.linspace(0, total_frames, n_frames + 1, dtype=int)
    indices = []
    for i in range(n_frames):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            idx = min(lo, total_frames - 1)
        else:
            idx = int(rng.integers(lo, hi))
        indices.append(idx)
    return np.array(indices, dtype=int)


def extract_clip_embedding(
    app: FaceAnalysis,
    video_path: Path,
    n_frames: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Sample N frames from a video and return an averaged, L2-normalised clip embedding."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video %s", video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sample_frame_indices(total_frames, n_frames, rng)

    embeddings: List[np.ndarray] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning("Could not read frame %d from %s", idx, video_path)
            continue

        faces = app.get(frame)
        if not faces:
            logger.warning("No face in frame %d of %s", idx, video_path)
            continue

        face = best_face(faces)
        if face is None:
            continue

        if len(faces) > 1:
            logger.info(
                "Frame %d of %s: %d faces; using highest det_score (%.3f)",
                idx,
                video_path,
                len(faces),
                float(face.det_score),
            )
        embeddings.append(face.normed_embedding)

    cap.release()

    if not embeddings:
        logger.warning("No valid face frames in %s", video_path)
        return None

    avg = np.mean(np.vstack(embeddings), axis=0)
    norm = np.linalg.norm(avg)
    if norm == 0:
        return None
    return avg / norm


def discover_files(directory: Path, extensions: Iterable[str]) -> List[Path]:
    """Return sorted files in directory matching the given extensions."""
    if not directory.is_dir():
        logger.error("Not a directory: %s", directory)
        return []
    exts = {e.lower() for e in extensions}
    files = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts
    ]
    return sorted(files)


def sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised embeddings."""
    return float(cosine_similarity([a], [b])[0, 0])


def stats(values: np.ndarray) -> dict:
    """Return n/mean/std/min/max for a similarity array."""
    if len(values) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def write_csv(
    output_path: Path,
    source_to_output: List[Tuple[Path, Path, float]],
    output_to_output: List[Tuple[Path, Path, float]],
    negatives: List[Tuple[Path, Path, float]],
) -> None:
    """Write raw per-pair similarities to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_type", "file_a", "file_b", "similarity"])
        for a, b, s in source_to_output:
            writer.writerow(["source_to_output", str(a), str(b), f"{s:.6f}"])
        for a, b, s in output_to_output:
            writer.writerow(["output_to_output", str(a), str(b), f"{s:.6f}"])
        for a, b, s in negatives:
            writer.writerow(["negative", str(a), str(b), f"{s:.6f}"])


def print_markdown_summary(
    source_to_output: np.ndarray,
    output_to_output: np.ndarray,
    negative: np.ndarray,
    d_prime: float,
    style_score: float,
    sps_threshold: float,
) -> None:
    """Print a markdown summary of calibration statistics and the sensitivity sweep."""
    sets = {
        "source_to_output": source_to_output,
        "output_to_output": output_to_output,
        "negative": negative,
    }

    print("\n## Similarity distributions\n")
    print("| set | n | mean | std | min | max |")
    print("|---|---|---|---|---|---|")
    for name, vals in sets.items():
        st = stats(vals)
        print(
            f"| {name} | {st['n']} | {st['mean']:.4f} | {st['std']:.4f} | "
            f"{st['min']:.4f} | {st['max']:.4f} |"
        )

    print(f"\n**d-prime:** {d_prime:.4f}\n")

    positive = np.concatenate([source_to_output, output_to_output])
    mean_face = float(np.mean(positive)) if len(positive) else float("nan")
    weights = np.linspace(0.5, 0.8, 7)

    print("## Face-weight sensitivity sweep\n")
    print(f"mean_face = {mean_face:.4f}, mean_style = {style_score:.4f}, sps_threshold = {sps_threshold:.4f}\n")
    print("| face_weight | sps | pass |")
    print("|---|---|---|")
    passes = []
    for w in weights:
        sps = w * mean_face + (1 - w) * style_score
        passed = sps > sps_threshold
        passes.append(passed)
        verdict = "pass" if passed else "fail"
        print(f"| {w:.2f} | {sps:.4f} | {verdict} |")

    if all(passes) or not any(passes):
        print("\nNo verdict flips across the face-weight sweep.")
    else:
        flip_indices = [
            i for i in range(1, len(passes)) if passes[i] != passes[i - 1]
        ]
        print(f"\nVerdict flips at weight indices: {flip_indices}")


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    # Load model once and reuse for all extractions.
    app = load_model(args.model_root)

    # Source embedding: strict; source must be a clean single-face portrait.
    source_emb = extract_image_embedding(app, args.source, strict=True)
    if source_emb is None:
        logger.error("Could not extract source embedding from %s", args.source)
        return 1

    # Output clip embeddings.
    output_files = discover_files(args.outputs, VIDEO_EXTS)
    if not output_files:
        logger.error("No video files found in %s", args.outputs)
        return 1

    clip_embeddings: List[Tuple[Path, np.ndarray]] = []
    for video_path in output_files:
        emb = extract_clip_embedding(app, video_path, args.n_frames, rng)
        if emb is not None:
            clip_embeddings.append((video_path, emb))
    if not clip_embeddings:
        logger.error("No valid clip embeddings extracted from %s", args.outputs)
        return 1

    # Negative identity embeddings: strict; each negative must be a clean portrait.
    negative_files = discover_files(args.negatives, IMAGE_EXTS)
    if not negative_files:
        logger.warning("No negative images found in %s", args.negatives)

    negative_embeddings: List[Tuple[Path, np.ndarray]] = []
    for img_path in negative_files:
        emb = extract_image_embedding(app, img_path, strict=True)
        if emb is not None:
            negative_embeddings.append((img_path, emb))

    # Build similarity sets.
    source_to_output_rows: List[Tuple[Path, Path, float]] = []
    for clip_path, clip_emb in clip_embeddings:
        s = sim(source_emb, clip_emb)
        source_to_output_rows.append((args.source, clip_path, s))

    output_to_output_rows: List[Tuple[Path, Path, float]] = []
    for i in range(len(clip_embeddings)):
        for j in range(i + 1, len(clip_embeddings)):
            a_path, a_emb = clip_embeddings[i]
            b_path, b_emb = clip_embeddings[j]
            s = sim(a_emb, b_emb)
            output_to_output_rows.append((a_path, b_path, s))

    negative_rows: List[Tuple[Path, Path, float]] = []
    for neg_path, neg_emb in negative_embeddings:
        s = sim(source_emb, neg_emb)
        negative_rows.append((args.source, neg_path, s))

    source_to_output = np.array([s for _, _, s in source_to_output_rows])
    output_to_output = np.array([s for _, _, s in output_to_output_rows])
    negative = np.array([s for _, _, s in negative_rows])

    # d-prime treating source_to_output + output_to_output as positive.
    positive = np.concatenate([source_to_output, output_to_output])
    mu_pos = float(np.mean(positive))
    sd_pos = float(np.std(positive, ddof=0))
    mu_neg = float(np.mean(negative)) if len(negative) else 0.0
    sd_neg = float(np.std(negative, ddof=0)) if len(negative) else 0.0

    pooled_sd = np.sqrt((sd_pos**2 + sd_neg**2) / 2.0)
    if pooled_sd == 0:
        d_prime = float("inf") if mu_pos > mu_neg else 0.0
    else:
        d_prime = (mu_pos - mu_neg) / pooled_sd

    # Write CSV.
    write_csv(args.output_csv, source_to_output_rows, output_to_output_rows, negative_rows)
    logger.info("Wrote raw similarities to %s", args.output_csv)

    # Print markdown summary.
    print_markdown_summary(
        source_to_output,
        output_to_output,
        negative,
        d_prime,
        args.style_score,
        args.sps_threshold,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
