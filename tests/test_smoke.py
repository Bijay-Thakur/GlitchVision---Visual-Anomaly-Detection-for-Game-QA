"""End-to-end smoke test using a synthetic local video.

This test covers the full pipeline except YouTube resolution:
    - imports
    - frame sampling
    - ResNet-18 embedding
    - Isolation Forest scoring
    - CSV + image artifact creation

It generates a tiny synthetic MP4 on the fly so CI / local smoke runs don't
need any real footage. Skips gracefully if torch or OpenCV can't be imported.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _write_synthetic_video(path: Path, fps: int = 10, seconds: int = 6) -> bool:
    """Write a tiny synthetic video with a few deliberately odd frames.

    Returns True on success. OpenCV's mp4 writer availability varies by
    platform; if encoding fails, the caller should skip the test.
    """
    try:
        import cv2
    except ImportError:
        return False

    width, height = 96, 96
    n_frames = fps * seconds

    # Use mp4v which is broadly supported on Windows/macOS/Linux builds.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        return False

    rng = np.random.default_rng(0)
    try:
        for i in range(n_frames):
            # "Normal" frames: a calm, slowly drifting gradient.
            r = 40 + (i % 30)
            g = 80
            b = 120
            frame = np.full((height, width, 3), (b, g, r), dtype=np.uint8)
            # Inject a loud outlier every 25 frames.
            if i in {20, 45}:
                frame = (rng.integers(0, 255, frame.shape)).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    return path.exists() and path.stat().st_size > 0


def test_imports_work():
    """Basic sanity check that all public modules import cleanly."""
    from src.ingestion import resolve_youtube_stream, save_uploaded_file  # noqa
    from src.processing import FrameExtractor  # noqa
    from src.features import EmbeddingExtractor  # noqa
    from src.models import AnomalyDetector  # noqa
    from src.pipeline import GlitchVisionPipeline, PipelineConfig  # noqa
    from src.utils import (  # noqa
        normalize_scores,
        rank_top_k,
        smooth_scores,
        ensure_dir,
        new_run_dir,
    )


def test_pipeline_end_to_end(tmp_path: Path):
    """Run the full pipeline on a synthetic clip and check outputs."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("PyTorch not installed; skipping end-to-end smoke test.")

    try:
        import cv2  # noqa: F401
    except ImportError:
        pytest.skip("OpenCV not installed; skipping end-to-end smoke test.")

    video_path = tmp_path / "synthetic.mp4"
    if not _write_synthetic_video(video_path):
        pytest.skip("Could not write synthetic MP4 (OpenCV writer unavailable).")

    from src.pipeline import GlitchVisionPipeline, PipelineConfig

    cfg = PipelineConfig(
        interval_sec=0.5,
        top_k=3,
        contamination=0.1,
        max_frames=40,
        smoothing_window=1,
        min_gap_frames=0,
        batch_size=4,
        output_dir=tmp_path / "outputs",
    )
    pipeline = GlitchVisionPipeline(cfg)
    result = pipeline.run(
        video_source=str(video_path),
        source_type="local_upload",
        source_label="synthetic.mp4",
    )

    assert result.csv_path.exists(), "CSV artifact should be written"
    assert result.run_dir.exists()
    assert result.total_sampled_frames >= 2
    assert len(result.top_records) > 0
    assert len(result.top_records) <= cfg.top_k

    # Every top record should have a saved image on disk.
    for rec in result.top_records:
        img_path = result.run_dir / rec.image_path
        assert img_path.exists(), f"Missing top anomaly image: {img_path}"

    # CSV should contain all sampled frames.
    import pandas as pd

    df = pd.read_csv(result.csv_path)
    assert len(df) == result.total_sampled_frames
    expected_cols = {
        "rank",
        "frame_index",
        "source_frame_idx",
        "timestamp_sec",
        "anomaly_score",
        "normalized_score",
        "image_path",
        "source_type",
        "source_label",
        "mode",
    }
    assert expected_cols.issubset(set(df.columns))

    # Segment CSV + markdown report should be produced too.
    if result.segment_csv_path is not None:
        assert result.segment_csv_path.exists()
    if result.report_path is not None:
        assert result.report_path.exists()
        assert result.report_path.read_text(encoding="utf-8").strip() != ""


if __name__ == "__main__":
    # Allow `python tests/test_smoke.py` as a quick manual check.
    sys.exit(pytest.main([__file__, "-v"]))
