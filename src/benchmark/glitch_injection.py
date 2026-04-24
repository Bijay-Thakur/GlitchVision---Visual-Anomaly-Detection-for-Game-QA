"""Synthetic glitch injection utilities for honest self-evaluation.

We do not have labeled game-QA data, so we approximate it: take a clean
(or at least clean-ish) clip, inject visually recognizable artifacts at
known timestamps, and use those known intervals as ground truth when
evaluating the pipeline.

This is a **proxy benchmark**. Real glitches are rarer, subtler, and
correlated with the rendering engine. Treat numbers from this benchmark
as a sanity check on the detector's behavior, not as a production claim.

Supported glitch kinds (all CPU-only, OpenCV/NumPy):

- ``brightness``  - strong brightness/contrast shift
- ``color``       - channel swap / color cast
- ``occlusion``   - random black rectangle covering part of the frame
- ``noise``       - heavy additive Gaussian noise
- ``blur``        - strong Gaussian blur
- ``freeze``      - repeat the previous frame (stuck animation)
- ``hud_dropout`` - knock out a HUD-shaped region (bottom/top band)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class GlitchInterval:
    """Ground-truth annotation: [start_frame, end_frame) was corrupted."""

    start_frame: int
    end_frame: int
    kind: str


# ---------------------------------------------------------------
# individual per-frame glitch operators
# ---------------------------------------------------------------
def _glitch_brightness(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    delta = int(rng.choice([-90, -70, 70, 90]))
    out = frame.astype(np.int16) + delta
    return np.clip(out, 0, 255).astype(np.uint8)


def _glitch_color(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Channel swap. BGR -> permuted.
    perms: List[Tuple[int, int, int]] = [(2, 1, 0), (1, 2, 0), (2, 0, 1), (0, 2, 1)]
    order = perms[int(rng.integers(0, len(perms)))]
    return frame[..., list(order)].copy()


def _glitch_occlusion(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    # Cover 20-45% of the frame with a black rectangle.
    bw = int(rng.integers(max(4, w // 5), max(5, int(w * 0.45)) + 1))
    bh = int(rng.integers(max(4, h // 5), max(5, int(h * 0.45)) + 1))
    bx = int(rng.integers(0, max(1, w - bw)))
    by = int(rng.integers(0, max(1, h - bh)))
    out[by : by + bh, bx : bx + bw] = 0
    return out


def _glitch_noise(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 45.0, size=frame.shape).astype(np.int16)
    out = frame.astype(np.int16) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _glitch_blur(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    import cv2

    # Kernel size scales with frame size so the blur is visibly strong.
    h, w = frame.shape[:2]
    k = max(7, (min(h, w) // 10) | 1)  # force odd
    return cv2.GaussianBlur(frame, (k, k), 0)


def _glitch_hud_dropout(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    # Paint a black HUD-shaped band along bottom or top.
    band_h = max(6, h // 6)
    if rng.random() < 0.5:
        out[-band_h:, :] = 0
    else:
        out[:band_h, :] = 0
    return out


# ``freeze`` is handled separately in :func:`inject_glitches` because it
# needs the previous frame.

_GLITCH_OPS: Dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "brightness": _glitch_brightness,
    "color": _glitch_color,
    "occlusion": _glitch_occlusion,
    "noise": _glitch_noise,
    "blur": _glitch_blur,
    "hud_dropout": _glitch_hud_dropout,
}


def available_glitch_kinds() -> List[str]:
    """Return every glitch kind this module knows how to inject."""
    return list(_GLITCH_OPS.keys()) + ["freeze"]


# ---------------------------------------------------------------
# batch injection
# ---------------------------------------------------------------
def inject_glitches(
    frames: Sequence[np.ndarray],
    intervals: Sequence[GlitchInterval],
    seed: Optional[int] = 0,
) -> Tuple[List[np.ndarray], List[GlitchInterval]]:
    """Apply ``intervals`` to a list of BGR uint8 frames.

    Returns
    -------
    (corrupted_frames, cleaned_intervals)
        ``corrupted_frames`` is a new list; input frames are not mutated.
        ``cleaned_intervals`` drops any requests that were clipped to an
        empty range after bounds checking.
    """
    if not frames:
        return [], []

    rng = np.random.default_rng(seed)
    out = [f.copy() for f in frames]
    n = len(out)
    cleaned: List[GlitchInterval] = []

    for iv in intervals:
        kind = iv.kind.lower()
        start = max(0, int(iv.start_frame))
        end = min(n, int(iv.end_frame))
        if end <= start:
            continue

        if kind == "freeze":
            # Replace each frame in the interval with the frame *just before*
            # the interval (or the first frame if interval starts at 0).
            src_idx = max(0, start - 1)
            ref = out[src_idx].copy()
            for i in range(start, end):
                out[i] = ref.copy()
        else:
            op = _GLITCH_OPS.get(kind)
            if op is None:
                raise ValueError(f"Unknown glitch kind: {iv.kind!r}")
            for i in range(start, end):
                out[i] = op(out[i], rng)

        cleaned.append(
            GlitchInterval(start_frame=start, end_frame=end, kind=kind)
        )

    return out, cleaned


def plan_glitch_schedule(
    n_frames: int,
    kinds: Sequence[str] = ("brightness", "occlusion", "noise", "blur"),
    n_intervals: int = 3,
    interval_length: int = 4,
    seed: Optional[int] = 0,
) -> List[GlitchInterval]:
    """Deterministically pick non-overlapping glitch intervals.

    Parameters
    ----------
    n_frames:
        Total number of candidate frames (sampled, not source).
    kinds:
        Pool of glitch kinds to cycle through.
    n_intervals:
        How many glitch intervals to schedule.
    interval_length:
        How many sampled frames each glitch covers.
    seed:
        RNG seed for reproducibility.
    """
    if n_frames <= 0 or n_intervals <= 0 or interval_length <= 0 or not kinds:
        return []

    rng = np.random.default_rng(seed)
    interval_length = min(int(interval_length), max(1, n_frames // max(1, n_intervals * 2)))
    if interval_length < 1:
        interval_length = 1

    # Carve n_intervals evenly-spaced slots and jitter each a bit.
    slot_size = n_frames // (n_intervals + 1)
    if slot_size < interval_length + 1:
        # Fall back to contiguous placement if the clip is short.
        slot_size = interval_length + 1

    out: List[GlitchInterval] = []
    used_ranges: List[Tuple[int, int]] = []
    for i in range(n_intervals):
        target = (i + 1) * slot_size
        jitter = int(rng.integers(-slot_size // 4, slot_size // 4 + 1))
        start = max(0, min(n_frames - interval_length, target + jitter))
        end = start + interval_length

        # Prevent overlap with previous intervals.
        for us, ue in used_ranges:
            if not (end <= us or start >= ue):
                start = ue
                end = start + interval_length
        if end > n_frames:
            break
        used_ranges.append((start, end))

        kind = str(kinds[i % len(kinds)])
        out.append(GlitchInterval(start_frame=start, end_frame=end, kind=kind))

    return out
