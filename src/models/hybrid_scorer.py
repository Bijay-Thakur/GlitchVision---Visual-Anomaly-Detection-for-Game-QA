"""Combine within-clip and reference-distance scores into a single ranking.

The hybrid score blends two complementary signals:

- **Within-clip** (Isolation Forest): captures frames that are unusual
  *inside the candidate clip itself*. Useful when no reference is
  available, or when the reference isn't perfectly representative.
- **Reference-distance** (kNN): captures frames that are unusual compared
  to a known-good build. This is what "regression triage" literally
  means in practice.

The two scores live on different scales (IF raw score vs. cosine
distance), so we **min-max normalize each independently** before mixing.
That keeps the blend weights interpretable and bounded.

    hybrid_score = w_within * norm(within_score)
                 + w_reference * norm(reference_score)

Default weights (0.5 / 0.5) make it easy to reason about — neither
signal dominates. Users can bias toward one or the other via the
Streamlit UI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.scoring import normalize_scores


@dataclass
class HybridScoreResult:
    """Container for the blended score and its components (for the UI/report)."""

    hybrid: np.ndarray
    within_norm: np.ndarray
    reference_norm: np.ndarray
    weight_within: float
    weight_reference: float


def hybrid_score(
    within_scores: np.ndarray,
    reference_scores: np.ndarray,
    weight_within: float = 0.5,
    weight_reference: float = 0.5,
) -> HybridScoreResult:
    """Blend two per-frame score arrays into a single anomaly score.

    Both inputs must be 1-D arrays of the same length. The output keeps
    the "higher = more anomalous" convention.
    """
    w = np.asarray(within_scores, dtype=np.float32).reshape(-1)
    r = np.asarray(reference_scores, dtype=np.float32).reshape(-1)

    if w.size == 0 or r.size == 0:
        raise ValueError("Both score arrays must be non-empty.")
    if w.size != r.size:
        raise ValueError(
            f"Score length mismatch: within={w.size}, reference={r.size}"
        )

    ww = float(weight_within)
    wr = float(weight_reference)
    total = ww + wr
    if total <= 0:
        raise ValueError("Hybrid weights must sum to a positive number.")
    # Renormalize so the blend weights always sum to 1.
    ww /= total
    wr /= total

    w_norm = normalize_scores(w)
    r_norm = normalize_scores(r)

    hybrid = ww * w_norm + wr * r_norm
    return HybridScoreResult(
        hybrid=hybrid.astype(np.float32, copy=False),
        within_norm=w_norm,
        reference_norm=r_norm,
        weight_within=ww,
        weight_reference=wr,
    )
