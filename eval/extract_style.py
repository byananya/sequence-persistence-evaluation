"""Style feature extraction for the Sequence Persistence Score."""

from pathlib import Path
from typing import Union

import cv2
import numpy as np


def _style_vector_from_bgr(bgr: np.ndarray, size: tuple = (64, 64)) -> np.ndarray:
    """Build a lightweight style descriptor from a BGR image.

    The descriptor concatenates the mean and standard deviation of the L*a*b*
    channels after resizing. This captures colour grade and lighting without
    requiring a heavy deep-learning model.
    """
    resized = cv2.resize(bgr, size)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean = lab.mean(axis=(0, 1))
    std = lab.std(axis=(0, 1))
    return np.concatenate([mean, std])


def get_style_vector(
    image_path: Union[str, Path],
    size: tuple = (64, 64),
) -> np.ndarray:
    """Return a style vector for a single image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return _style_vector_from_bgr(img, size)


def get_style_vector_from_video(
    video_path: Union[str, Path],
    size: tuple = (64, 64),
    frame_offset: float = 0.5,
) -> np.ndarray:
    """Return a style vector for a chosen frame of a video."""
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
    return _style_vector_from_bgr(frame, size)
