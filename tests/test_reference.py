"""Tests for the reference bank + reference/hybrid scorers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.hybrid_scorer import hybrid_score
from src.models.reference_scorer import (
    ReferenceScorer,
    ReferenceScorerConfig,
)
from src.reference import FrameRef, ReferenceBank, build_reference_bank


def _unit_vectors(seed: int, n: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def test_reference_bank_roundtrip(tmp_path: Path):
    emb = _unit_vectors(1, 12, 8)
    frames = [
        FrameRef(source_label="ref.mp4", frame_index=i, timestamp_sec=float(i))
        for i in range(12)
    ]
    bank = build_reference_bank(
        embeddings=emb,
        frames=frames,
        backbone="resnet18",
        image_size=224,
        interval_sec=1.0,
        source_videos=["ref.mp4"],
        notes="test bank",
    )
    assert bank.size == 12
    assert bank.embedding_dim == 8

    out_dir = tmp_path / "bank"
    bank.save(out_dir)
    assert (out_dir / "embeddings.npz").exists()
    assert (out_dir / "metadata.json").exists()

    loaded = ReferenceBank.load(out_dir)
    assert loaded.size == bank.size
    assert loaded.embedding_dim == bank.embedding_dim
    assert loaded.backbone == "resnet18"
    assert loaded.source_videos == ["ref.mp4"]
    np.testing.assert_allclose(loaded.embeddings, bank.embeddings, rtol=1e-6)


def test_reference_bank_rejects_mismatched_shapes():
    emb = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        ReferenceBank(embeddings=emb, frames=[])  # type: ignore[arg-type]


def test_reference_scorer_cosine_higher_for_outlier():
    ref = _unit_vectors(0, 50, 16)
    # Candidate 0: an average-ish vector taken from the reference cloud.
    # Candidate 1: its antipode (max cosine distance).
    in_dist = ref[0:1].copy()
    outlier = -in_dist
    candidates = np.concatenate([in_dist, outlier], axis=0)

    scorer = ReferenceScorer(ref, ReferenceScorerConfig(k=5, metric="cosine"))
    scores = scorer.score(candidates)

    assert scores.shape == (2,)
    assert scores[1] > scores[0]
    assert scores[0] >= 0.0


def test_reference_scorer_k_clamps_to_bank_size():
    ref = _unit_vectors(1, 3, 8)
    scorer = ReferenceScorer(ref, ReferenceScorerConfig(k=50, metric="euclidean"))
    scores = scorer.score(ref)
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_reference_scorer_rejects_dim_mismatch():
    ref = _unit_vectors(2, 4, 8)
    cand = _unit_vectors(3, 4, 16)
    scorer = ReferenceScorer(ref)
    with pytest.raises(ValueError):
        scorer.score(cand)


def test_hybrid_score_blend_math():
    within = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    reference = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    res = hybrid_score(within, reference, weight_within=0.5, weight_reference=0.5)

    assert res.hybrid.shape == (4,)
    np.testing.assert_allclose(res.within_norm, [0.0, 1 / 3, 2 / 3, 1.0], atol=1e-5)
    np.testing.assert_allclose(res.reference_norm, [1.0, 2 / 3, 1 / 3, 0.0], atol=1e-5)
    # Symmetric blend with linear inputs -> constant 0.5 per frame.
    np.testing.assert_allclose(res.hybrid, [0.5, 0.5, 0.5, 0.5], atol=1e-5)
    assert res.weight_within == pytest.approx(0.5)
    assert res.weight_reference == pytest.approx(0.5)


def test_hybrid_score_renormalizes_weights():
    within = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    reference = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    # Weights that don't sum to 1 — should be rescaled internally.
    res = hybrid_score(within, reference, weight_within=2.0, weight_reference=2.0)
    assert res.weight_within == pytest.approx(0.5)
    assert res.weight_reference == pytest.approx(0.5)


def test_hybrid_score_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        hybrid_score(np.array([1.0, 2.0]), np.array([1.0]))
