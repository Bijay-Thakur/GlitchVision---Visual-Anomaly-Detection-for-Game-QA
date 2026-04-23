"""Score post-processing helpers (normalization, smoothing, ranking)."""
from __future__ import annotations

from typing import List

import numpy as np


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores into [0, 1]. Constant input returns zeros."""
    if scores.size == 0:
        return scores.astype(np.float32, copy=False)
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi - lo < 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32, copy=False)


def smooth_scores(scores: np.ndarray, window: int = 1) -> np.ndarray:
    """Centered moving average. ``window <= 1`` returns the input unchanged."""
    window = int(window)
    if window <= 1 or scores.size == 0:
        return scores.astype(np.float32, copy=False)

    window = min(window, scores.size)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    # ``same`` keeps length; edges get biased but that's fine for a demo.
    return np.convolve(scores.astype(np.float32), kernel, mode="same")


def rank_top_k(
    scores: np.ndarray,
    k: int,
    min_gap: int = 0,
) -> List[int]:
    """Indices of the top-K scores, optionally enforcing a minimum index gap.

    ``min_gap`` is a cheap temporal dedup heuristic: if two top candidates are
    within ``min_gap`` sampled frames of each other, only the higher-scored
    one is kept. This avoids a gallery full of near-duplicates.
    """
    if scores.size == 0 or k <= 0:
        return []

    k = int(min(k, scores.size))
    # argsort descending
    order = np.argsort(-scores, kind="stable")

    if min_gap <= 0:
        return order[:k].tolist()

    selected: List[int] = []
    for idx in order:
        if all(abs(int(idx) - int(s)) > min_gap for s in selected):
            selected.append(int(idx))
        if len(selected) >= k:
            break
    return selected
