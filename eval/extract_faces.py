"""Face embedding extraction for the Sequence Persistence Score."""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def load_model(root: str = "models") -> FaceAnalysis:
    """Load the default InsightFace face-analysis model.

    The first call downloads the `buffalo_l` model into `root/`.
    """
    app = FaceAnalysis(
        name="buffalo_l",
        root=root,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _largest_face(faces):
    if not faces:
        raise ValueError("No face detected")
    return sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )[0]


def get_face_embedding(
    image_path: Union[str, Path],
    model: FaceAnalysis = None,
) -> np.ndarray:
    """Return the embedding of the largest face in a single image."""
    if model is None:
        model = load_model()
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return _largest_face(model.get(img)).embedding


def get_face_embedding_from_video(
    video_path: Union[str, Path],
    model: FaceAnalysis = None,
    frame_offset: float = 0.5,
) -> np.ndarray:
    """Return the embedding of the largest face at a chosen frame of a video.

    frame_offset is the relative position through the video (0.0 = first frame,
    1.0 = last frame). The default 0.5 picks the middle frame.
    """
    if model is None:
        model = load_model()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    target_frame = int(total_frames * frame_offset)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame from {video_path}")
    return _largest_face(model.get(frame)).embedding
