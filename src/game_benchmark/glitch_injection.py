"""Synthetic gameplay glitch dataset generation.

The generated labels are a proxy benchmark for gameplay anomaly triage. They
are useful for side-by-side model comparisons, but they are not a substitute for
real engine bug labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np


@dataclass
class GameGlitchInterval:
    start_frame: int
    end_frame: int
    kind: str


@dataclass
class GlitchDataset:
    frames: np.ndarray
    labels: np.ndarray
    intervals: List[GameGlitchInterval]
    timestamps: np.ndarray
    groups: np.ndarray

    def interval_dicts(self) -> list[dict]:
        return [asdict(iv) for iv in self.intervals]


def _brightness_shift(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    delta = int(rng.choice([-85, -60, 60, 85]))
    return np.clip(frame.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def _contrast_shift(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = float(rng.choice([0.45, 0.6, 1.65, 1.9]))
    beta = float(rng.choice([-20, 0, 20]))
    return np.clip(frame.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _blur(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    try:
        import cv2

        k = max(5, (min(frame.shape[:2]) // 9) | 1)
        return cv2.GaussianBlur(frame, (k, k), 0)
    except Exception:
        # Tiny box blur fallback for environments where OpenCV is not installed.
        padded = np.pad(frame.astype(np.float32), ((1, 1), (1, 1), (0, 0)), mode="edge")
        out = np.zeros_like(frame, dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                out += padded[dy : dy + frame.shape[0], dx : dx + frame.shape[1]]
        return np.clip(out / 9.0, 0, 255).astype(np.uint8)


def _gaussian_noise(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 38.0, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _black_frame(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.zeros_like(frame)


def _hud_occlusion(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    block_w = int(rng.integers(max(8, w // 6), max(9, w // 3)))
    block_h = int(rng.integers(max(8, h // 10), max(9, h // 4)))
    x = int(rng.choice([0, max(0, w - block_w), rng.integers(0, max(1, w - block_w))]))
    y = int(rng.choice([0, max(0, h - block_h)]))
    out[y : y + block_h, x : x + block_w] = 0
    return out


_FRAME_GLITCHES = {
    "brightness_shift": _brightness_shift,
    "contrast_shift": _contrast_shift,
    "blur": _blur,
    "gaussian_noise": _gaussian_noise,
    "black_frame": _black_frame,
    "hud_occlusion": _hud_occlusion,
}
_TEMPORAL_GLITCHES = {"freeze_stutter", "temporal_jump"}


def available_gameplay_glitches() -> list[str]:
    return list(_FRAME_GLITCHES) + sorted(_TEMPORAL_GLITCHES)


def plan_gameplay_glitches(
    n_frames: int,
    seed: int = 42,
    n_intervals: int | None = None,
    interval_length: int | None = None,
) -> List[GameGlitchInterval]:
    if n_frames <= 1:
        return []
    kinds = available_gameplay_glitches()
    n_intervals = n_intervals or min(6, max(1, n_frames // 12))
    interval_length = interval_length or max(1, min(5, n_frames // 12))
    rng = np.random.default_rng(seed)
    slot = max(interval_length + 1, n_frames // (n_intervals + 1))
    intervals: List[GameGlitchInterval] = []
    cursor_end = 0
    for i in range(n_intervals):
        center = (i + 1) * slot
        jitter = int(rng.integers(-max(1, slot // 4), max(2, slot // 4 + 1)))
        start = max(cursor_end, min(n_frames - interval_length, center + jitter))
        end = min(n_frames, start + interval_length)
        if end <= start:
            continue
        intervals.append(GameGlitchInterval(start, end, kinds[i % len(kinds)]))
        cursor_end = end + 1
    return intervals


def inject_gameplay_glitches(
    frames: Sequence[np.ndarray] | np.ndarray,
    timestamps: Sequence[float] | np.ndarray | None = None,
    groups: Sequence[str] | np.ndarray | None = None,
    intervals: Sequence[GameGlitchInterval] | None = None,
    seed: int = 42,
    debug_dir: str | Path | None = "data/outputs/game_benchmark/debug_glitches",
) -> GlitchDataset:
    """Inject labeled glitches and return frames, frame labels, and intervals."""
    arr = np.asarray(frames, dtype=np.uint8).copy()
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected frames shaped (N, H, W, 3), got {arr.shape}")
    n = arr.shape[0]
    ts = np.arange(n, dtype=np.float32) if timestamps is None else np.asarray(timestamps, dtype=np.float32)
    if len(ts) != n:
        raise ValueError("timestamps length must match frames")
    grp = (
        np.array(["clip_0"] * n, dtype=object)
        if groups is None
        else np.asarray(groups, dtype=object)
    )
    if len(grp) != n:
        raise ValueError("groups length must match frames")

    rng = np.random.default_rng(seed)
    planned = list(intervals or plan_gameplay_glitches(n, seed=seed))
    labels = np.zeros(n, dtype=np.int64)
    clean = arr.copy()
    applied: List[GameGlitchInterval] = []

    for iv in planned:
        start = max(0, int(iv.start_frame))
        end = min(n, int(iv.end_frame))
        if end <= start:
            continue
        kind = iv.kind
        if kind == "freeze_stutter":
            source = clean[max(0, start - 1)].copy()
            arr[start:end] = source
        elif kind == "temporal_jump":
            chunk = arr[start:end].copy()
            arr[start:end] = chunk[::-1]
        else:
            op = _FRAME_GLITCHES.get(kind)
            if op is None:
                raise ValueError(f"Unknown glitch kind: {kind!r}")
            for idx in range(start, end):
                arr[idx] = op(arr[idx], rng)
        labels[start:end] = 1
        applied.append(GameGlitchInterval(start, end, kind))

    if debug_dir is not None:
        _write_debug_examples(clean, arr, applied, Path(debug_dir))

    return GlitchDataset(frames=arr, labels=labels, intervals=applied, timestamps=ts, groups=grp)


def _write_debug_examples(
    clean: np.ndarray,
    corrupted: np.ndarray,
    intervals: Sequence[GameGlitchInterval],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        writer = lambda path, image: cv2.imwrite(str(path), image)
    except Exception:
        from PIL import Image

        writer = lambda path, image: Image.fromarray(image[..., ::-1]).save(path)
    for iv in intervals[:8]:
        idx = min(max(0, iv.start_frame), len(corrupted) - 1)
        writer(out_dir / f"{idx:04d}_{iv.kind}_clean.jpg", clean[idx])
        writer(out_dir / f"{idx:04d}_{iv.kind}_glitched.jpg", corrupted[idx])
