"""Score candidate frames against a reference embedding bank.

The business question this answers is:
    "How far is each candidate frame from the nearest known-good frames?"

We keep the math simple and explainable:

1. Embed candidate frames (done upstream).
2. For each candidate embedding, find its ``k`` nearest neighbors in the
   reference bank.
3. Use the *mean* neighbor distance as the anomaly score
   (higher = further = more anomalous).

Why kNN mean distance instead of a parametric model?
    - It has no training loop and zero hyperparameters beyond ``k``.
    - It is robust when the reference distribution is multi-modal (a
      gameplay capture can legitimately include many visually distinct
      scenes: menu, cutscene, combat, inventory).
    - Distances are interpretable and easy to debug.

We support two distance metrics:
    - ``cosine``  (default; pairs well with L2-normalized embeddings)
    - ``euclidean``

Both produce "higher = more anomalous" scores, which is what the rest of
the pipeline expects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Metric = Literal["cosine", "euclidean"]


@dataclass
class ReferenceScorerConfig:
    k: int = 5
    metric: Metric = "cosine"
    # If the bank is small, we clamp k so it never exceeds bank size.


def _pairwise_cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine *distance* matrix of shape (len(a), len(b)) in [0, 2]."""
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_unit = a / np.maximum(a_norm, 1e-12)
    b_unit = b / np.maximum(b_norm, 1e-12)
    sim = a_unit @ b_unit.T
    # Clip tiny numerical drift so 1 - sim is non-negative.
    sim = np.clip(sim, -1.0, 1.0)
    return (1.0 - sim).astype(np.float32, copy=False)


def _pairwise_euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix of shape (len(a), len(b))."""
    # (x - y)^2 = x^2 + y^2 - 2xy, computed in chunks would save memory but
    # for laptop-scale bank sizes this dense version is fine.
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    a_sq = np.sum(a * a, axis=1, keepdims=True)
    b_sq = np.sum(b * b, axis=1, keepdims=True).T
    sq = a_sq + b_sq - 2.0 * (a @ b.T)
    # Clamp tiny negatives from float error before sqrt.
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq, dtype=np.float32)


class ReferenceScorer:
    """kNN-distance anomaly scorer against a reference embedding bank.

    Example
    -------
    >>> scorer = ReferenceScorer(bank_embeddings, ReferenceScorerConfig(k=5))
    >>> scores = scorer.score(candidate_embeddings)
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        config: ReferenceScorerConfig | None = None,
    ) -> None:
        ref = np.asarray(reference_embeddings, dtype=np.float32)
        if ref.ndim != 2 or ref.shape[0] == 0:
            raise ValueError(
                "Reference embeddings must be a non-empty 2-D array; "
                f"got shape {ref.shape}."
            )
        self.reference = ref
        self.config = config or ReferenceScorerConfig()

    # -------------------------------------------------------------
    # scoring
    # -------------------------------------------------------------
    def score(self, candidate_embeddings: np.ndarray) -> np.ndarray:
        """Return per-candidate anomaly scores (higher = more anomalous)."""
        cand = np.asarray(candidate_embeddings, dtype=np.float32)
        if cand.ndim != 2 or cand.shape[0] == 0:
            raise ValueError(
                "Candidate embeddings must be a non-empty 2-D array; "
                f"got shape {cand.shape}."
            )
        if cand.shape[1] != self.reference.shape[1]:
            raise ValueError(
                "Embedding dim mismatch between candidate "
                f"({cand.shape[1]}) and reference ({self.reference.shape[1]}). "
                "Did you use the same backbone?"
            )

        metric = self.config.metric
        if metric == "cosine":
            dists = _pairwise_cosine_distance(cand, self.reference)
        elif metric == "euclidean":
            dists = _pairwise_euclidean_distance(cand, self.reference)
        else:  # pragma: no cover - dataclass Literal guards this
            raise ValueError(f"Unknown metric: {metric!r}")

        k = max(1, min(int(self.config.k), self.reference.shape[0]))
        # np.partition avoids a full sort; take the k smallest distances.
        if k >= dists.shape[1]:
            neighbor_d = dists
        else:
            neighbor_d = np.partition(dists, k - 1, axis=1)[:, :k]

        return neighbor_d.mean(axis=1).astype(np.float32, copy=False)
