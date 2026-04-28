"""Models used by the gameplay benchmark.

ResNet-18 is used only as a frozen pretrained feature extractor. The trained
model here is a lightweight classifier on top of embeddings and temporal
features.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.temporal_features import compute_temporal_features
from src.models.reference_scorer import ReferenceScorer, ReferenceScorerConfig


@dataclass
class ModelScores:
    name: str
    scores: np.ndarray
    latency_sec: float = 0.0
    artifact_size_mb: float = 0.0


def handcrafted_visual_features(frames: np.ndarray) -> np.ndarray:
    """CPU-cheap visual features: histograms, brightness, edges, blur."""
    arr = np.asarray(frames, dtype=np.uint8)
    feats = []
    try:
        import cv2
    except Exception:
        cv2 = None
    for frame in arr:
        hist_parts = []
        for channel in range(3):
            hist, _ = np.histogram(frame[..., channel], bins=8, range=(0, 256))
            hist = hist.astype(np.float32)
            hist = hist / max(float(hist.sum()), 1e-12)
            hist_parts.extend(hist.tolist())
        if cv2 is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            edge_density = float((edges > 0).mean())
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        else:
            gray = frame.mean(axis=2)
            gy, gx = np.gradient(gray.astype(np.float32))
            grad = np.sqrt(gx * gx + gy * gy)
            edge_density = float((grad > 20.0).mean())
            blur = float(np.var(grad))
        brightness_mean = float(gray.mean())
        brightness_std = float(gray.std())
        feats.append(hist_parts + [brightness_mean, brightness_std, edge_density, blur])
    return np.asarray(feats, dtype=np.float32)


def _iforest_scores(features: np.ndarray, seed: int = 42) -> np.ndarray:
    if len(features) < 2:
        return np.zeros(len(features), dtype=np.float32)
    model = IsolationForest(
        n_estimators=160,
        contamination=min(0.25, max(0.02, 1.0 / max(10, len(features)))),
        random_state=seed,
        n_jobs=1,
    )
    model.fit(features)
    return (-model.score_samples(features)).astype(np.float32)


def score_handcrafted_iforest(frames: np.ndarray, seed: int = 42) -> np.ndarray:
    return _iforest_scores(handcrafted_visual_features(frames), seed=seed)


def score_embedding_iforest(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    return _iforest_scores(np.asarray(embeddings, dtype=np.float32), seed=seed)


def score_reference_knn(
    embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    if reference_embeddings is None or len(reference_embeddings) == 0:
        return np.zeros(len(embeddings), dtype=np.float32)
    scorer = ReferenceScorer(
        np.asarray(reference_embeddings, dtype=np.float32),
        ReferenceScorerConfig(k=k, metric="cosine"),
    )
    return scorer.score(np.asarray(embeddings, dtype=np.float32))


def supervised_feature_matrix(
    embeddings: np.ndarray,
    timestamps: np.ndarray | None = None,
    include_temporal: bool = True,
) -> tuple[np.ndarray, list[str]]:
    emb = np.asarray(embeddings, dtype=np.float32)
    names = [f"embedding_{i}" for i in range(emb.shape[1])]
    if not include_temporal:
        return emb, names
    temporal = compute_temporal_features(emb, timestamps=timestamps)
    features = np.concatenate([emb, temporal.features], axis=1).astype(np.float32)
    return features, names + temporal.feature_names


def train_lightweight_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    timestamps: np.ndarray | None = None,
    out_path: str | Path = "data/models/game_anomaly_classifier.joblib",
    model_type: str = "logistic_regression",
    seed: int = 42,
) -> Dict[str, object]:
    """Train a light classifier with group split to avoid frame leakage."""
    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    x, feature_names = supervised_feature_matrix(embeddings, timestamps, include_temporal=True)
    if len(np.unique(y)) < 2 or len(y) < 4:
        raise ValueError("Need at least two classes and four samples to train classifier")

    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        train_idx, test_idx = next(splitter.split(x, y, groups))
    else:
        split = max(1, int(len(y) * 0.7))
        train_idx = np.arange(0, split)
        test_idx = np.arange(split, len(y))
        if len(test_idx) == 0:
            test_idx = train_idx

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=1, class_weight="balanced")
        model = clf
    else:
        clf = LogisticRegression(max_iter=600, class_weight="balanced", random_state=seed)
        model = make_pipeline(StandardScaler(), clf)

    model.fit(x[train_idx], y[train_idx])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    schema_path = out.with_suffix(".schema.json")
    schema_path.write_text(
        json.dumps(
            {
                "model_type": model_type,
                "trained_model": "Lightweight classifier on frozen ResNet-18 embeddings plus temporal features",
                "feature_names": feature_names,
                "split": "GroupShuffleSplit by source video/clip when multiple groups exist",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "model": model,
        "model_path": str(out),
        "schema_path": str(schema_path),
        "feature_names": feature_names,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "features": x,
    }


def classifier_anomaly_scores(model: object, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        if proba.shape[1] == 1:
            return np.zeros(len(features), dtype=np.float32)
        return proba[:, 1].astype(np.float32)
    if hasattr(model, "decision_function"):
        raw = model.decision_function(features)
        raw = np.asarray(raw, dtype=np.float32)
        lo, hi = float(raw.min()), float(raw.max())
        return ((raw - lo) / max(hi - lo, 1e-12)).astype(np.float32)
    pred = np.asarray(model.predict(features), dtype=np.float32)
    return pred


def file_size_mb(path: str | Path | None) -> float:
    if path is None:
        return 0.0
    p = Path(path)
    return p.stat().st_size / (1024 * 1024) if p.exists() else 0.0
