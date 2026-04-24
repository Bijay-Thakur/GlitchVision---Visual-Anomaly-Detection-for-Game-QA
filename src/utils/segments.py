"""Segment-level aggregation over per-frame anomaly scores.

QA reviewers usually care about *suspicious intervals*, not isolated
frames. This module takes per-frame scores and computes:

1. A rolling-mean score curve over a configurable window.
2. Non-overlapping "segments" of ``window_size`` sampled frames.
3. A ranked list of the top anomalous segments with timestamps and a
   representative frame (the highest-scoring frame inside the segment).

We intentionally keep segments non-overlapping so segment IDs map 1:1 to
regions on disk and so the UI can render them without confusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class Segment:
    """A contiguous run of sampled frames and its aggregate score."""

    segment_id: int
    start_frame: int        # inclusive index into the sampled-frame list
    end_frame: int          # exclusive
    start_time_sec: float
    end_time_sec: float
    mean_score: float
    max_score: float
    representative_frame: int  # sampled-frame index of the max-scoring frame
    rank: int = -1             # filled in after ranking


def build_segments(
    scores: np.ndarray,
    timestamps: Sequence[float],
    window_size: int,
) -> List[Segment]:
    """Partition sampled frames into non-overlapping segments and score them.

    Parameters
    ----------
    scores:
        Per-sampled-frame anomaly scores (higher = more anomalous).
    timestamps:
        Timestamps (seconds) for each sampled frame.
    window_size:
        Number of sampled frames per segment. Values < 2 produce a single
        segment per frame, which is rarely useful — we clamp to >= 2.
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    ts = np.asarray(list(timestamps), dtype=np.float32).reshape(-1)
    if scores.size != ts.size:
        raise ValueError(
            f"scores/timestamps length mismatch: {scores.size} vs {ts.size}"
        )
    n = int(scores.size)
    if n == 0:
        return []

    w = max(2, int(window_size))
    w = min(w, n)

    segments: List[Segment] = []
    for seg_id, start in enumerate(range(0, n, w)):
        end = min(n, start + w)
        chunk = scores[start:end]
        rep = int(start + int(np.argmax(chunk)))
        segments.append(
            Segment(
                segment_id=seg_id,
                start_frame=int(start),
                end_frame=int(end),
                start_time_sec=float(ts[start]),
                # ``end`` is exclusive; use last included timestamp for display.
                end_time_sec=float(ts[end - 1]),
                mean_score=float(np.mean(chunk)),
                max_score=float(np.max(chunk)),
                representative_frame=rep,
            )
        )
    return segments


def rank_segments(
    segments: List[Segment],
    top_k: int,
    score_key: str = "mean_score",
) -> List[Segment]:
    """Return the top ``top_k`` segments sorted by ``score_key`` desc.

    Also stamps ``rank`` on the returned copies so callers can write it
    directly to CSV.
    """
    if not segments or top_k <= 0:
        return []
    if score_key not in {"mean_score", "max_score"}:
        raise ValueError("score_key must be 'mean_score' or 'max_score'")

    ordered = sorted(segments, key=lambda s: getattr(s, score_key), reverse=True)
    top = ordered[: int(top_k)]
    ranked: List[Segment] = []
    for rank, seg in enumerate(top, start=1):
        ranked.append(
            Segment(
                segment_id=seg.segment_id,
                start_frame=seg.start_frame,
                end_frame=seg.end_frame,
                start_time_sec=seg.start_time_sec,
                end_time_sec=seg.end_time_sec,
                mean_score=seg.mean_score,
                max_score=seg.max_score,
                representative_frame=seg.representative_frame,
                rank=rank,
            )
        )
    return ranked


def segment_to_row(seg: Segment) -> dict:
    """Flatten a ``Segment`` into a dict suitable for CSV writing."""
    return {
        "rank": seg.rank,
        "segment_id": seg.segment_id,
        "start_frame": seg.start_frame,
        "end_frame": seg.end_frame,
        "start_time_sec": round(seg.start_time_sec, 3),
        "end_time_sec": round(seg.end_time_sec, 3),
        "mean_score": round(seg.mean_score, 6),
        "max_score": round(seg.max_score, 6),
        "representative_frame": seg.representative_frame,
    }
