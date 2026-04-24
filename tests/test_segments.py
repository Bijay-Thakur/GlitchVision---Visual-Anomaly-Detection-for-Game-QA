"""Tests for segment-level aggregation."""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.segments import (
    build_segments,
    rank_segments,
    segment_to_row,
)


def test_build_segments_partitions_cleanly():
    scores = np.array([0.0, 0.2, 0.9, 0.8, 0.1, 0.0], dtype=np.float32)
    timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    segs = build_segments(scores, timestamps, window_size=3)
    assert len(segs) == 2
    assert segs[0].start_frame == 0 and segs[0].end_frame == 3
    assert segs[1].start_frame == 3 and segs[1].end_frame == 6
    # Segment 0 should have its max at index 2 (score 0.9).
    assert segs[0].representative_frame == 2
    assert segs[0].max_score == pytest.approx(0.9)
    assert segs[0].mean_score == pytest.approx((0.0 + 0.2 + 0.9) / 3, abs=1e-6)


def test_build_segments_last_partial_window():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9], dtype=np.float32)
    timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
    segs = build_segments(scores, timestamps, window_size=3)
    assert len(segs) == 2
    assert segs[-1].start_frame == 3
    assert segs[-1].end_frame == 5
    assert segs[-1].end_time_sec == pytest.approx(4.0)
    assert segs[-1].representative_frame == 4


def test_rank_segments_orders_and_stamps_rank():
    scores = np.array(
        [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5], dtype=np.float32
    )
    timestamps = list(range(9))
    segs = build_segments(scores, timestamps, window_size=3)
    top = rank_segments(segs, top_k=2)
    assert len(top) == 2
    assert top[0].rank == 1
    assert top[1].rank == 2
    assert top[0].mean_score >= top[1].mean_score


def test_build_segments_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        build_segments(
            np.array([1.0, 2.0], dtype=np.float32),
            [0.0, 1.0, 2.0],
            window_size=2,
        )


def test_segment_to_row_shape():
    scores = np.array([0.1, 0.9, 0.5, 0.2], dtype=np.float32)
    segs = build_segments(scores, [0.0, 1.0, 2.0, 3.0], window_size=2)
    row = segment_to_row(segs[0])
    assert "segment_id" in row
    assert "mean_score" in row
    assert "representative_frame" in row
