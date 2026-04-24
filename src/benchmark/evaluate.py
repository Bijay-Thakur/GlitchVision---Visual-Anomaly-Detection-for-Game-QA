"""Metrics for the synthetic benchmark.

We keep metrics honest and few:

- **Precision@K** — of the top K frames flagged by the pipeline, what
  fraction fell inside an injected glitch interval?
- **Hit@K** — is *any* of the top K frames inside an injected interval?
- **Recall on intervals** — what fraction of injected glitch intervals
  were "hit" (had at least one of the top-K frames inside them)?
- **Segment overlap** — IoU-style overlap between the union of the top
  anomalous segments and the union of injected glitch intervals.

Everything operates over sampled-frame indices, not source frame
indices. This matches the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .glitch_injection import GlitchInterval


@dataclass
class BenchmarkMetrics:
    """Bundle of honest benchmark numbers for a single run."""

    k: int
    n_frames: int
    n_flagged: int
    precision_at_k: float
    hit_at_k: float
    interval_recall: float
    segment_overlap_iou: float
    hit_intervals: int
    total_intervals: int


def _interval_contains(iv: GlitchInterval, frame_idx: int) -> bool:
    return iv.start_frame <= int(frame_idx) < iv.end_frame


def _union_mask(intervals: Iterable[Tuple[int, int]], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for s, e in intervals:
        s = max(0, int(s))
        e = min(n, int(e))
        if e > s:
            mask[s:e] = True
    return mask


def evaluate_run(
    top_frame_indices: Sequence[int],
    ground_truth: Sequence[GlitchInterval],
    n_sampled_frames: int,
    top_segment_ranges: Sequence[Tuple[int, int]] = (),
) -> BenchmarkMetrics:
    """Compute honest metrics for one run.

    Parameters
    ----------
    top_frame_indices:
        Sampled-frame indices produced by the pipeline (already de-duped
        and ranked). Any length; typical is 10-20.
    ground_truth:
        The list of injected :class:`GlitchInterval`.
    n_sampled_frames:
        Total number of sampled frames analyzed.
    top_segment_ranges:
        Optional list of ``(start_frame, end_frame)`` for the pipeline's
        top segments. Used for IoU-style overlap.
    """
    top = [int(i) for i in top_frame_indices]
    k = len(top)

    if k == 0 or n_sampled_frames <= 0:
        return BenchmarkMetrics(
            k=k,
            n_frames=int(n_sampled_frames),
            n_flagged=0,
            precision_at_k=0.0,
            hit_at_k=0.0,
            interval_recall=0.0,
            segment_overlap_iou=0.0,
            hit_intervals=0,
            total_intervals=len(list(ground_truth)),
        )

    gts = list(ground_truth)

    # --- frame-level precision@k ---
    hits = 0
    for idx in top:
        if any(_interval_contains(iv, idx) for iv in gts):
            hits += 1
    precision = hits / k if k > 0 else 0.0
    hit_at_k = 1.0 if hits > 0 else 0.0

    # --- interval-level recall ---
    hit_ivs = 0
    for iv in gts:
        if any(_interval_contains(iv, idx) for idx in top):
            hit_ivs += 1
    interval_recall = (hit_ivs / len(gts)) if gts else 0.0

    # --- segment IoU vs. glitch intervals ---
    gt_mask = _union_mask(
        [(iv.start_frame, iv.end_frame) for iv in gts], n_sampled_frames
    )
    seg_mask = _union_mask(top_segment_ranges, n_sampled_frames)
    inter = int(np.logical_and(gt_mask, seg_mask).sum())
    union = int(np.logical_or(gt_mask, seg_mask).sum())
    iou = (inter / union) if union > 0 else 0.0

    return BenchmarkMetrics(
        k=k,
        n_frames=int(n_sampled_frames),
        n_flagged=k,
        precision_at_k=float(precision),
        hit_at_k=float(hit_at_k),
        interval_recall=float(interval_recall),
        segment_overlap_iou=float(iou),
        hit_intervals=int(hit_ivs),
        total_intervals=int(len(gts)),
    )
