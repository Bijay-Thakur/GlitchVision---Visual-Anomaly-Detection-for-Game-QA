"""Lightweight temporal features for gameplay anomaly detection.

These features model frame-to-frame embedding behavior on CPU. They are not a
video transformer; they are compact signals that help catch freezes, stutters,
and sudden jumps while preserving the existing per-frame ResNet workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class TemporalFeatureResult:
    features: np.ndarray
    feature_names: list[str]
    timestamps: np.ndarray


def _rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    mean = np.zeros(n, dtype=np.float32)
    std = np.zeros(n, dtype=np.float32)
    w = max(1, int(window))
    for i in range(n):
        start = max(0, i - w + 1)
        chunk = values[start : i + 1]
        mean[i] = float(np.mean(chunk)) if len(chunk) else 0.0
        std[i] = float(np.std(chunk)) if len(chunk) else 0.0
    return mean, std


def compute_temporal_features(
    embeddings: np.ndarray,
    timestamps: np.ndarray | None = None,
    rolling_window: int = 5,
    freeze_quantile: float = 0.10,
    jump_quantile: float = 0.90,
) -> TemporalFeatureResult:
    """Return temporal features aligned one-to-one with input embeddings."""
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected embeddings shape (N, D), got {emb.shape}")
    n = emb.shape[0]
    ts = (
        np.arange(n, dtype=np.float32)
        if timestamps is None
        else np.asarray(timestamps, dtype=np.float32)
    )
    if ts.shape[0] != n:
        raise ValueError("timestamps length must match embeddings")

    delta_l2 = np.zeros(n, dtype=np.float32)
    cosine_dist = np.zeros(n, dtype=np.float32)
    if n > 1:
        diffs = emb[1:] - emb[:-1]
        delta_l2[1:] = np.linalg.norm(diffs, axis=1).astype(np.float32)
        dots = np.sum(emb[1:] * emb[:-1], axis=1)
        norms = np.linalg.norm(emb[1:], axis=1) * np.linalg.norm(emb[:-1], axis=1)
        sims = dots / np.maximum(norms, 1e-12)
        cosine_dist[1:] = (1.0 - np.clip(sims, -1.0, 1.0)).astype(np.float32)

    roll_mean, roll_std = _rolling_mean_std(delta_l2, rolling_window)
    positive = delta_l2[1:] if n > 1 else np.array([], dtype=np.float32)
    if positive.size:
        freeze_threshold = float(np.quantile(positive, freeze_quantile))
        jump_threshold = float(np.quantile(positive, jump_quantile))
    else:
        freeze_threshold = 0.0
        jump_threshold = 0.0
    freeze_signal = (delta_l2 <= freeze_threshold).astype(np.float32)
    freeze_signal[0] = 0.0
    sudden_jump_signal = (delta_l2 >= jump_threshold).astype(np.float32)
    sudden_jump_signal[0] = 0.0

    features = np.column_stack(
        [
            delta_l2,
            cosine_dist,
            roll_mean,
            roll_std,
            freeze_signal,
            sudden_jump_signal,
        ]
    ).astype(np.float32)
    names = [
        "embedding_delta_l2",
        "cosine_distance_prev",
        "rolling_delta_mean",
        "rolling_delta_std",
        "freeze_stutter_signal",
        "sudden_jump_signal",
    ]
    return TemporalFeatureResult(features=features, feature_names=names, timestamps=ts)


def temporal_feature_schema() -> Dict[str, str]:
    return {
        "embedding_delta_l2": "L2 distance between consecutive frame embeddings.",
        "cosine_distance_prev": "Cosine distance to the previous frame embedding.",
        "rolling_delta_mean": "Rolling mean of embedding deltas.",
        "rolling_delta_std": "Rolling standard deviation of embedding deltas.",
        "freeze_stutter_signal": "1 when frame-to-frame delta is unusually low.",
        "sudden_jump_signal": "1 when frame-to-frame delta is unusually high.",
    }
