"""Evaluation metrics for gameplay anomaly benchmark runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(fn, default: float = 0.0) -> float:
    try:
        value = fn()
        if value != value:
            return default
        return float(value)
    except Exception:
        return default


def top_k_indices(scores: Sequence[float], k: int) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0 or k <= 0:
        return np.array([], dtype=np.int64)
    k = min(int(k), arr.size)
    order = np.argsort(-arr, kind="mergesort")
    return order[:k].astype(np.int64)


def precision_at_k(y_true: Sequence[int], scores: Sequence[float], k: int) -> float:
    idx = top_k_indices(scores, k)
    if idx.size == 0:
        return 0.0
    labels = np.asarray(y_true, dtype=np.int64)
    return float(labels[idx].sum() / idx.size)


def recall_at_k(y_true: Sequence[int], scores: Sequence[float], k: int) -> float:
    labels = np.asarray(y_true, dtype=np.int64)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    idx = top_k_indices(scores, k)
    return float(labels[idx].sum() / positives) if idx.size else 0.0


def hit_at_k(y_true: Sequence[int], scores: Sequence[float], k: int) -> float:
    idx = top_k_indices(scores, k)
    if idx.size == 0:
        return 0.0
    labels = np.asarray(y_true, dtype=np.int64)
    return 1.0 if int(labels[idx].sum()) > 0 else 0.0


def _interval_mask(intervals: Sequence[Mapping[str, int]], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for iv in intervals:
        s = max(0, int(iv.get("start_frame", 0)))
        e = min(n, int(iv.get("end_frame", 0)))
        if e > s:
            mask[s:e] = True
    return mask


def interval_recall(
    intervals: Sequence[Mapping[str, int]],
    flagged_indices: Sequence[int],
) -> float:
    if not intervals:
        return 0.0
    flagged = {int(i) for i in flagged_indices}
    hit = 0
    for iv in intervals:
        s = int(iv.get("start_frame", 0))
        e = int(iv.get("end_frame", 0))
        if any(s <= idx < e for idx in flagged):
            hit += 1
    return float(hit / len(intervals))


def segment_iou(
    intervals: Sequence[Mapping[str, int]],
    predicted_intervals: Sequence[Mapping[str, int]],
    n_frames: int,
) -> float:
    gt = _interval_mask(intervals, n_frames)
    pred = _interval_mask(predicted_intervals, n_frames)
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(gt, pred).sum() / union)


def compute_metrics(
    y_true: Sequence[int],
    scores: Sequence[float],
    threshold: float | None = None,
    k: int | None = None,
    intervals: Sequence[Mapping[str, int]] = (),
    predicted_intervals: Sequence[Mapping[str, int]] = (),
) -> dict:
    """Compute frame, ranking, and interval metrics.

    Higher ``scores`` must mean more anomalous.
    """
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float32)
    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    n = int(y.shape[0])
    k = int(k or max(1, min(10, n)))

    if threshold is None:
        threshold = float(np.quantile(s, 0.90)) if n else 0.0
    pred = (s >= threshold).astype(np.int64)

    cm = confusion_matrix(y, pred, labels=[0, 1]) if n else np.zeros((2, 2), dtype=int)
    top = top_k_indices(s, k)
    out = {
        "n_samples": n,
        "positive_frames": int(y.sum()),
        "threshold": float(threshold),
        "accuracy": _safe_metric(lambda: accuracy_score(y, pred)),
        "precision": _safe_metric(lambda: precision_score(y, pred, zero_division=0)),
        "recall": _safe_metric(lambda: recall_score(y, pred, zero_division=0)),
        "f1": _safe_metric(lambda: f1_score(y, pred, zero_division=0)),
        "roc_auc": _safe_metric(lambda: roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None,
        "pr_auc": _safe_metric(lambda: average_precision_score(y, s)) if len(np.unique(y)) > 1 else None,
        "precision_at_k": precision_at_k(y, s, k),
        "recall_at_k": recall_at_k(y, s, k),
        "hit_at_k": hit_at_k(y, s, k),
        "k": k,
        "top_k_indices": [int(i) for i in top],
        "confusion_matrix": cm.astype(int).tolist(),
        "interval_recall": interval_recall(intervals, top),
        "segment_iou": segment_iou(intervals, predicted_intervals, n),
    }
    return out


def save_metrics(metrics_by_model: Mapping[str, Mapping[str, object]], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "benchmark_results.json"
    csv_path = out / "benchmark_results.csv"
    json_path.write_text(json.dumps(metrics_by_model, indent=2), encoding="utf-8")

    scalar_keys = sorted(
        {
            key
            for metrics in metrics_by_model.values()
            for key, value in metrics.items()
            if isinstance(value, (int, float, str)) or value is None
        }
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + scalar_keys)
        writer.writeheader()
        for model, metrics in metrics_by_model.items():
            row = {"model": model}
            row.update({k: metrics.get(k) for k in scalar_keys})
            writer.writerow(row)
    return json_path, csv_path
