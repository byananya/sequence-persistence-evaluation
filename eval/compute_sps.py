"""Sequence Persistence Score computation."""

from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


FACE_THRESHOLD = 0.65
SPS_THRESHOLD = 0.70


def face_persistence_score(
    source_embedding: np.ndarray,
    output_embeddings: List[np.ndarray],
) -> float:
    """Average cosine similarity across source-to-output and output-to-output pairs."""
    sims = []
    for emb in output_embeddings:
        sims.append(cosine_similarity([source_embedding], [emb])[0, 0])
    for i in range(len(output_embeddings) - 1):
        sims.append(
            cosine_similarity([output_embeddings[i]], [output_embeddings[i + 1]])[0, 0]
        )
    return float(np.mean(sims)) if sims else 0.0


def style_persistence_score(
    source_vector: np.ndarray,
    output_vectors: List[np.ndarray],
) -> float:
    """Convert average style distance into a 0–1 score.

    A smaller distance means a more consistent style. The score is an inverse
    exponential of the average L2 distance; you can tune the denominator if
    your style vectors use a different scale.
    """
    dists = []
    for vec in output_vectors:
        dists.append(np.linalg.norm(source_vector - vec))
    for i in range(len(output_vectors) - 1):
        dists.append(np.linalg.norm(output_vectors[i] - output_vectors[i + 1]))
    avg_dist = float(np.mean(dists)) if dists else 0.0
    # The denominator is a heuristic based on the L*a*b* mean/std descriptor.
    return float(np.exp(-avg_dist / 50.0))


def compute_sps(
    source_face_embedding: np.ndarray,
    output_face_embeddings: List[np.ndarray],
    source_style_vector: np.ndarray,
    output_style_vectors: List[np.ndarray],
    face_weight: float = 0.6,
    style_weight: float = 0.4,
) -> dict:
    """Compute the Sequence Persistence Score and pass/fail status."""
    face_score = face_persistence_score(source_face_embedding, output_face_embeddings)
    style_score = style_persistence_score(source_style_vector, output_style_vectors)
    sps = face_weight * face_score + style_weight * style_score
    return {
        "face_persistence": round(face_score, 4),
        "style_persistence": round(style_score, 4),
        "sps": round(sps, 4),
        "face_pass": face_score > FACE_THRESHOLD,
        "sps_pass": sps > SPS_THRESHOLD,
    }
