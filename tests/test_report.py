"""Tests for the lightweight run-report builder."""
from __future__ import annotations

from pathlib import Path

from src.reporting import (
    ReportFrame,
    ReportSegment,
    RunReport,
    write_report,
)


def test_write_report_creates_non_empty_markdown(tmp_path: Path):
    run_dir = tmp_path / "run_demo"
    run_dir.mkdir()

    report = RunReport(
        run_dir=run_dir,
        mode="hybrid",
        source_type="local_upload",
        source_label="demo.mp4",
        backbone="resnet18",
        total_sampled_frames=42,
        config_summary={"interval_sec": 1.0, "top_k": 5},
        top_frames=[
            ReportFrame(
                rank=1,
                timestamp_sec=12.5,
                score=0.92,
                normalized_score=1.0,
                image_path="frames/rank01.jpg",
            ),
        ],
        top_segments=[
            ReportSegment(
                rank=1,
                start_time_sec=10.0,
                end_time_sec=15.0,
                mean_score=0.8,
                max_score=0.95,
                representative_frame=12,
            ),
        ],
        csv_path="anomalies.csv",
        segment_csv_path="segments.csv",
        plot_path="score_plot.png",
        hybrid_weights={"weight_within": 0.5, "weight_reference": 0.5},
    )

    out = write_report(report)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "GlitchVision run report" in text
    assert "hybrid" in text
    assert "demo.mp4" in text
    assert "rank01.jpg" in text
    assert "segments.csv" in text
    assert "Limitations" in text


def test_write_report_handles_within_clip_mode(tmp_path: Path):
    run_dir = tmp_path / "run_wc"
    run_dir.mkdir()
    report = RunReport(
        run_dir=run_dir,
        mode="within_clip_iforest",
        source_type="youtube",
        source_label="Some Title",
        backbone="resnet18",
        total_sampled_frames=10,
        config_summary={"interval_sec": 1.0},
        top_frames=[],
        top_segments=[],
        csv_path="anomalies.csv",
    )
    out = write_report(report)
    text = out.read_text(encoding="utf-8")
    assert "Within-clip mode" in text
    assert "No top frames surfaced" in text
