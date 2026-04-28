from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features.temporal_features import compute_temporal_features
from src.game_benchmark.evaluate import compute_metrics, hit_at_k, precision_at_k
from src.game_benchmark.glitch_injection import inject_gameplay_glitches, plan_gameplay_glitches
from src.utils.profiling import StageProfiler, write_profile_reports


def test_temporal_feature_shapes():
    emb = np.eye(6, 4, dtype=np.float32)
    res = compute_temporal_features(emb, timestamps=np.arange(6))
    assert res.features.shape == (6, 6)
    assert len(res.feature_names) == 6
    assert res.timestamps.shape == (6,)


def test_gameplay_glitch_labels_and_intervals(tmp_path: Path):
    frames = np.full((20, 32, 32, 3), 80, dtype=np.uint8)
    intervals = plan_gameplay_glitches(20, seed=1, n_intervals=2, interval_length=3)
    intervals[0].kind = "brightness_shift"
    intervals[1].kind = "contrast_shift"
    ds = inject_gameplay_glitches(frames, intervals=intervals, seed=1, debug_dir=None)
    assert ds.frames.shape == frames.shape
    assert ds.labels.sum() == sum(iv.end_frame - iv.start_frame for iv in ds.intervals)


def test_precision_and_hit_at_k():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.2, 0.7])
    assert precision_at_k(y, scores, 2) == 1.0
    assert hit_at_k(y, scores, 2) == 1.0


def test_compute_metrics_schema_handles_single_class():
    metrics = compute_metrics([0, 0, 0], [0.1, 0.2, 0.3], k=2)
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "precision_at_k",
        "recall_at_k",
        "hit_at_k",
        "confusion_matrix",
        "interval_recall",
        "segment_iou",
    ]:
        assert key in metrics
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None


def test_profiler_output_schema(tmp_path: Path):
    profiler = StageProfiler()
    with profiler.stage("training"):
        sum(range(10))
    metrics = profiler.finish(samples=2, frames_sampled=2)
    assert "training_sec" in metrics
    assert "total_sec" in metrics
    assert "peak_memory_mb" in metrics
    json_path, csv_path, cost_path = write_profile_reports(metrics, tmp_path)
    assert json_path.exists()
    assert csv_path.exists()
    assert cost_path.exists()
