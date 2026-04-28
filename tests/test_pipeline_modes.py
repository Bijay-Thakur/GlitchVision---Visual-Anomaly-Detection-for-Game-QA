"""End-to-end tests for reference and hybrid scoring modes.

These tests build a tiny reference bank from a synthetic clean clip, then
run a second clip (with injected corruption) through the pipeline in
reference / hybrid mode. They are conditional on the torch + OpenCV
stack being available; CI skips them cleanly when it isn't.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_synthetic_video(path: Path, mode: str = "clean", fps: int = 8, seconds: int = 6) -> bool:
    try:
        import cv2
    except ImportError:
        return False

    width, height = 96, 96
    n_frames = fps * seconds
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        return False

    rng = np.random.default_rng(0 if mode == "clean" else 7)
    try:
        for i in range(n_frames):
            # "Normal" frames: a calm, slowly drifting gradient.
            r = 40 + (i % 30)
            g = 80
            b = 120
            frame = np.full((height, width, 3), (b, g, r), dtype=np.uint8)
            # Inject loud outliers in the "candidate" clip only.
            if mode == "candidate" and i in {20, 21, 22, 40, 41}:
                frame = (rng.integers(0, 255, frame.shape)).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path.exists() and path.stat().st_size > 0


@pytest.fixture(scope="module")
def _deps_available() -> bool:
    try:
        import torch  # noqa: F401
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def test_reference_bank_build_and_reference_mode(tmp_path: Path, _deps_available):
    if not _deps_available:
        pytest.skip("torch or opencv missing")

    from src.pipeline import GlitchVisionPipeline, PipelineConfig

    ref_path = tmp_path / "ref.mp4"
    cand_path = tmp_path / "cand.mp4"
    if not _write_synthetic_video(ref_path, mode="clean"):
        pytest.skip("OpenCV writer unavailable")
    if not _write_synthetic_video(cand_path, mode="candidate"):
        pytest.skip("OpenCV writer unavailable")

    cfg = PipelineConfig(
        interval_sec=0.5,
        top_k=3,
        max_frames=40,
        smoothing_window=1,
        segment_window=3,
        segment_top_k=3,
        reference_k=3,
        mode="reference_distance",
        output_dir=tmp_path / "outputs",
    )
    pipeline = GlitchVisionPipeline(cfg)

    bank_dir = tmp_path / "bank"
    bank = pipeline.build_reference(
        [(str(ref_path), "ref.mp4")],
        out_dir=bank_dir,
    )
    assert bank.size >= 2
    assert (bank_dir / "embeddings.npz").exists()
    assert (bank_dir / "metadata.json").exists()

    result = pipeline.run(
        video_source=str(cand_path),
        source_type="local_upload",
        source_label="cand.mp4",
        reference_bank=bank,
    )
    assert result.csv_path.exists()
    assert result.mode == "reference_distance"
    assert len(result.top_records) > 0
    assert result.run_metrics_path is not None
    assert result.run_metrics_path.exists()
    rm = json.loads(result.run_metrics_path.read_text(encoding="utf-8"))
    assert rm["throughput"]["sampled_frames_per_sec_end_to_end"] > 0
    assert result.eval_metrics_path is None
    # All top records should carry a finite reference score.
    for rec in result.top_records:
        assert rec.reference_score == rec.reference_score  # not NaN


def test_hybrid_mode_end_to_end(tmp_path: Path, _deps_available):
    if not _deps_available:
        pytest.skip("torch or opencv missing")

    from src.pipeline import GlitchVisionPipeline, PipelineConfig

    ref_path = tmp_path / "ref.mp4"
    cand_path = tmp_path / "cand.mp4"
    if not _write_synthetic_video(ref_path, mode="clean"):
        pytest.skip("OpenCV writer unavailable")
    if not _write_synthetic_video(cand_path, mode="candidate"):
        pytest.skip("OpenCV writer unavailable")

    pipeline = GlitchVisionPipeline(
        PipelineConfig(
            interval_sec=0.5,
            top_k=3,
            max_frames=40,
            smoothing_window=1,
            segment_window=3,
            mode="hybrid",
            hybrid_weight_within=0.5,
            hybrid_weight_reference=0.5,
            output_dir=tmp_path / "outputs",
        )
    )
    bank = pipeline.build_reference([(str(ref_path), "ref.mp4")])
    result = pipeline.run(
        video_source=str(cand_path),
        source_type="local_upload",
        source_label="cand.mp4",
        reference_bank=bank,
    )
    assert result.mode == "hybrid"
    assert result.csv_path.exists()
    for rec in result.top_records:
        assert rec.within_score == rec.within_score
        assert rec.reference_score == rec.reference_score


def test_pipeline_writes_eval_metrics_with_labeled_indices(tmp_path: Path, _deps_available):
    if not _deps_available:
        pytest.skip("torch or opencv missing")

    from src.pipeline import GlitchVisionPipeline, PipelineConfig

    cand_path = tmp_path / "cand.mp4"
    if not _write_synthetic_video(cand_path, mode="candidate"):
        pytest.skip("OpenCV writer unavailable")

    # Glitches at source frames 20–22 and 40–41; with fps=8 and interval_sec=0.5
    # sampling step is 4, so anomalous sampled indices land at ~5 and ~10.
    pipeline = GlitchVisionPipeline(
        PipelineConfig(
            interval_sec=0.5,
            top_k=5,
            max_frames=48,
            smoothing_window=1,
            mode="within_clip_iforest",
            output_dir=tmp_path / "outputs",
        )
    )
    result = pipeline.run(
        video_source=str(cand_path),
        source_type="local_upload",
        source_label="cand.mp4",
        eval_positive_frame_indices=[5, 10],
        eval_glitch_intervals=(
            {"start_frame": 5, "end_frame": 8},
            {"start_frame": 10, "end_frame": 11},
        ),
    )
    assert result.eval_metrics_path is not None
    assert result.eval_metrics_path.exists()
    ev = json.loads(result.eval_metrics_path.read_text(encoding="utf-8"))
    assert "precision_at_k" in ev
    assert ev["n_samples"] == result.total_sampled_frames


def test_reference_mode_rejects_missing_bank(tmp_path: Path, _deps_available):
    if not _deps_available:
        pytest.skip("torch or opencv missing")

    from src.pipeline import GlitchVisionPipeline, PipelineConfig

    cand_path = tmp_path / "cand.mp4"
    if not _write_synthetic_video(cand_path, mode="candidate"):
        pytest.skip("OpenCV writer unavailable")

    pipeline = GlitchVisionPipeline(
        PipelineConfig(
            interval_sec=0.5,
            max_frames=30,
            mode="reference_distance",
            output_dir=tmp_path / "outputs",
        )
    )
    with pytest.raises(ValueError):
        pipeline.run(
            video_source=str(cand_path),
            source_type="local_upload",
            source_label="cand.mp4",
            reference_bank=None,
        )
