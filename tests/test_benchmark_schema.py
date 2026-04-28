from __future__ import annotations

import numpy as np

from src.game_benchmark.evaluate import compute_metrics


def test_benchmark_result_schema_for_model_row():
    y = np.array([0, 1, 0, 1, 0])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    metrics = compute_metrics(
        y,
        scores,
        k=2,
        intervals=[{"start_frame": 1, "end_frame": 2}, {"start_frame": 3, "end_frame": 4}],
        predicted_intervals=[{"start_frame": 1, "end_frame": 2}],
    )
    row = {
        "model": "ResNet-18 + temporal + trained LogisticRegression",
        "features_used": "ResNet embeddings + temporal deltas",
        "scorer": "LogisticRegression",
        "f1": metrics["f1"],
        "pr_auc": metrics["pr_auc"],
        "precision_at_k": metrics["precision_at_k"],
        "hit_at_k": metrics["hit_at_k"],
        "latency_sec": 0.01,
        "peak_memory_mb": 1.0,
        "artifact_size_mb": 0.001,
    }
    expected = {
        "model",
        "features_used",
        "scorer",
        "f1",
        "pr_auc",
        "precision_at_k",
        "hit_at_k",
        "latency_sec",
        "peak_memory_mb",
        "artifact_size_mb",
    }
    assert expected.issubset(row)
