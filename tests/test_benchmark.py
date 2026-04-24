"""Tests for synthetic glitch injection and the benchmark evaluator."""
from __future__ import annotations

import numpy as np
import pytest

from src.benchmark import (
    BenchmarkMetrics,
    GlitchInterval,
    available_glitch_kinds,
    evaluate_run,
    inject_glitches,
    plan_glitch_schedule,
)


def _clean_frames(n: int = 30, size: int = 48) -> list[np.ndarray]:
    base = np.full((size, size, 3), (50, 80, 120), dtype=np.uint8)
    return [base.copy() for _ in range(n)]


def test_available_glitch_kinds_includes_freeze():
    kinds = available_glitch_kinds()
    assert "freeze" in kinds
    for k in ("brightness", "color", "occlusion", "noise", "blur", "hud_dropout"):
        assert k in kinds


def test_inject_glitches_produces_visible_change():
    frames = _clean_frames(n=10)
    # Brightness + occlusion cover different pixel-level effects.
    ivs = [
        GlitchInterval(2, 5, "brightness"),
        GlitchInterval(7, 9, "occlusion"),
    ]
    corrupted, cleaned = inject_glitches(frames, ivs, seed=1)

    assert len(corrupted) == len(frames)
    assert all(c.shape == f.shape and c.dtype == f.dtype for c, f in zip(corrupted, frames))
    assert len(cleaned) == 2

    # Untouched frame should be identical to the clean original.
    np.testing.assert_array_equal(corrupted[0], frames[0])
    # At least one corrupted frame should differ from the clean source.
    assert not np.array_equal(corrupted[3], frames[3])
    assert not np.array_equal(corrupted[7], frames[7])


def test_inject_glitches_does_not_mutate_input():
    frames = _clean_frames(n=4)
    snapshot = [f.copy() for f in frames]
    ivs = [GlitchInterval(0, 2, "noise")]
    _corrupted, _ = inject_glitches(frames, ivs, seed=0)
    for a, b in zip(frames, snapshot):
        np.testing.assert_array_equal(a, b)


def test_inject_glitches_freeze_uses_previous_frame():
    frames = [
        np.full((16, 16, 3), (10, 10, 10), dtype=np.uint8),
        np.full((16, 16, 3), (20, 20, 20), dtype=np.uint8),
        np.full((16, 16, 3), (30, 30, 30), dtype=np.uint8),
        np.full((16, 16, 3), (40, 40, 40), dtype=np.uint8),
    ]
    ivs = [GlitchInterval(2, 4, "freeze")]
    corrupted, _ = inject_glitches(frames, ivs, seed=0)
    # Frames 2 and 3 should now equal frame 1 (the frame just before start).
    np.testing.assert_array_equal(corrupted[2], frames[1])
    np.testing.assert_array_equal(corrupted[3], frames[1])


def test_plan_glitch_schedule_non_overlapping():
    schedule = plan_glitch_schedule(n_frames=40, n_intervals=3, interval_length=4, seed=7)
    assert 1 <= len(schedule) <= 3
    # Sort by start and verify no overlaps.
    schedule.sort(key=lambda s: s.start_frame)
    for a, b in zip(schedule, schedule[1:]):
        assert a.end_frame <= b.start_frame


def test_evaluate_run_perfect_hit():
    ivs = [GlitchInterval(5, 8, "brightness"), GlitchInterval(15, 17, "noise")]
    # All top frames fall inside ground truth.
    metrics = evaluate_run(
        top_frame_indices=[5, 6, 15, 16],
        ground_truth=ivs,
        n_sampled_frames=20,
        top_segment_ranges=[(5, 8), (15, 17)],
    )
    assert isinstance(metrics, BenchmarkMetrics)
    assert metrics.precision_at_k == pytest.approx(1.0)
    assert metrics.hit_at_k == pytest.approx(1.0)
    assert metrics.interval_recall == pytest.approx(1.0)
    assert metrics.segment_overlap_iou == pytest.approx(1.0)
    assert metrics.hit_intervals == 2


def test_evaluate_run_all_misses():
    ivs = [GlitchInterval(5, 8, "noise")]
    metrics = evaluate_run(
        top_frame_indices=[0, 1, 2],
        ground_truth=ivs,
        n_sampled_frames=20,
    )
    assert metrics.precision_at_k == 0.0
    assert metrics.hit_at_k == 0.0
    assert metrics.interval_recall == 0.0


def test_evaluate_run_empty_top_returns_zeros():
    ivs = [GlitchInterval(1, 2, "noise")]
    metrics = evaluate_run([], ivs, n_sampled_frames=10)
    assert metrics.precision_at_k == 0.0
    assert metrics.hit_at_k == 0.0
    assert metrics.total_intervals == 1
