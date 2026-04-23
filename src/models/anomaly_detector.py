"""Unsupervised anomaly detection wrapper around ``IsolationForest``.

Why Isolation Forest?
---------------------
It is fast, has no neural training requirements, works well on embedding
spaces, and is a standard baseline for unsupervised anomaly detection. That
makes it an honest choice for an MVP demo on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class DetectorConfig:
    contamination: float = 0.05
    n_estimators: int = 200
    random_state: int = 42
    max_samples: str | int = "auto"


class AnomalyDetector:
    """Fit an Isolation Forest on embeddings and score each sample.

    Scoring convention
    ------------------
    sklearn's ``score_samples`` returns *higher is more normal*. We flip it so
    callers get *higher = more anomalous*, which is what the UI expects.
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self._model: Optional[IsolationForest] = None

    def fit(self, embeddings: np.ndarray) -> "AnomalyDetector":
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2-D; got shape {embeddings.shape}"
            )
        if embeddings.shape[0] < 2:
            raise ValueError(
                "Need at least 2 frames to fit the anomaly detector."
            )

        # Clamp contamination into a range sklearn accepts.
        contamination = float(np.clip(self.config.contamination, 1e-4, 0.5))

        self._model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=contamination,
            random_state=self.config.random_state,
            max_samples=self.config.max_samples,
            n_jobs=1,  # keep it polite on a laptop
        )
        self._model.fit(embeddings)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return an array of anomaly scores (higher = more anomalous)."""
        if self._model is None:
            raise RuntimeError("AnomalyDetector.fit() must be called first.")
        # sklearn: higher score_samples = more normal. Negate it.
        raw = self._model.score_samples(embeddings)
        return (-raw).astype(np.float32, copy=False)

    def fit_score(self, embeddings: np.ndarray) -> np.ndarray:
        self.fit(embeddings)
        return self.score(embeddings)
