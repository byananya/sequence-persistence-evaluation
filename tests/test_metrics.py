"""Unit tests for the Sequence Persistence metrics."""

import numpy as np

from eval.compute_sps import (
    FACE_THRESHOLD,
    SPS_THRESHOLD,
    compute_sps,
    face_persistence_score,
    style_persistence_score,
)


def test_identical_embeddings_and_styles_score_high():
    emb = np.random.rand(512).astype(np.float32)
    style = np.random.rand(6).astype(np.float32)
    result = compute_sps(emb, [emb, emb], style, [style, style])
    assert result["face_persistence"] > FACE_THRESHOLD
    assert result["sps"] > SPS_THRESHOLD
    assert result["face_pass"] is True
    assert result["sps_pass"] is True


def test_random_outputs_score_low():
    source_emb = np.random.rand(512).astype(np.float32)
    source_style = np.random.rand(6).astype(np.float32)
    out_embs = [np.random.rand(512).astype(np.float32) for _ in range(3)]
    out_styles = [np.random.rand(6).astype(np.float32) for _ in range(3)]
    result = compute_sps(source_emb, out_embs, source_style, out_styles)
    assert result["sps"] < 0.9


def test_face_persistence_with_different_embeddings():
    source = np.random.rand(512).astype(np.float32)
    out = [source + np.random.rand(512).astype(np.float32) for _ in range(2)]
    score = face_persistence_score(source, out)
    assert 0.0 <= score <= 1.0


def test_style_persistence_with_different_vectors():
    source = np.random.rand(6).astype(np.float32)
    out = [source + np.random.rand(6).astype(np.float32) * 100 for _ in range(2)]
    score = style_persistence_score(source, out)
    assert 0.0 <= score <= 1.0
