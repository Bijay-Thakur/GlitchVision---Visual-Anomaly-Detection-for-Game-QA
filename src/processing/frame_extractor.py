"""Sample frames from a video source at a fixed interval and resize to 224x224.

The extractor is a small generator-style class so the pipeline can stream
frames instead of loading everything into RAM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass
class SampledFrame:
    """One sampled frame plus its metadata."""

    index: int                 # sequential index among *sampled* frames
    source_frame_idx: int      # frame number in the original stream
    timestamp_sec: float
    image: np.ndarray          # BGR uint8 image, resized to target size


class FrameExtractor:
    """Iterate over frames from a video source (file path or stream URL).

    Parameters
    ----------
    source:
        A local file path or an HTTP(S) stream URL that OpenCV can open.
    interval_sec:
        Seconds between sampled frames. Default 1.0 (i.e. 1 fps).
    image_size:
        Output side length for the square resized frame.
    max_frames:
        Hard cap on the number of sampled frames (prevents runaway runs).
    """

    def __init__(
        self,
        source: str,
        interval_sec: float = 1.0,
        image_size: int = 224,
        max_frames: Optional[int] = None,
        duration_fraction: float = 1.0,
    ) -> None:
        if interval_sec <= 0:
            raise ValueError("interval_sec must be > 0")
        if image_size <= 0:
            raise ValueError("image_size must be > 0")
        if not (0.0 < duration_fraction <= 1.0):
            raise ValueError("duration_fraction must be in (0, 1]")

        self.source = source
        self.interval_sec = float(interval_sec)
        self.image_size = int(image_size)
        self.max_frames = max_frames
        self.duration_fraction = float(duration_fraction)

        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._total_frames: int = 0
        self._max_duration_sec: Optional[float] = None

    # ---------------------------------------------------------------
    # lifecycle
    # ---------------------------------------------------------------
    def open(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                "OpenCV could not open the video source. "
                "If this is a YouTube stream, try the local upload fallback."
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        # Some streams report bogus fps (0 or NaN); fall back to 30.
        if not fps or fps != fps or fps < 1.0 or fps > 240.0:
            fps = 30.0

        self._cap = cap
        self._fps = float(fps)
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self._max_duration_sec: Optional[float] = None
        if (
            self.duration_fraction < 0.999
            and self.duration_fraction > 0.0
            and self._total_frames > 0
            and self._fps > 0
        ):
            total_dur = self._total_frames / self._fps
            self._max_duration_sec = total_dur * self.duration_fraction

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameExtractor":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ---------------------------------------------------------------
    # properties
    # ---------------------------------------------------------------
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def estimated_sample_count(self) -> int:
        """Rough estimate of how many frames we'll emit."""
        if self._fps <= 0:
            return 0
        duration = 0.0
        if self._total_frames > 0:
            duration = self._total_frames / self._fps
        if self._max_duration_sec is not None:
            duration = min(duration, self._max_duration_sec)
        if duration <= 0:
            return 0
        est = int(duration / self.interval_sec) + 1
        if self.max_frames is not None:
            est = min(est, self.max_frames)
        return max(0, est)

    # ---------------------------------------------------------------
    # iteration
    # ---------------------------------------------------------------
    def iter_frames(self) -> Iterator[SampledFrame]:
        """Yield ``SampledFrame``s one at a time."""
        import cv2

        if self._cap is None:
            raise RuntimeError("FrameExtractor.open() must be called first.")

        step = max(1, int(round(self._fps * self.interval_sec)))
        src_frame_idx = 0
        emitted = 0

        while True:
            if self.max_frames is not None and emitted >= self.max_frames:
                break

            ok, frame = self._cap.read()
            if not ok or frame is None:
                break

            timestamp = src_frame_idx / self._fps if self._fps else 0.0
            if self._max_duration_sec is not None and timestamp >= self._max_duration_sec:
                break

            if src_frame_idx % step == 0:
                resized = cv2.resize(
                    frame,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_AREA,
                )
                yield SampledFrame(
                    index=emitted,
                    source_frame_idx=src_frame_idx,
                    timestamp_sec=float(timestamp),
                    image=resized,
                )
                emitted += 1

            src_frame_idx += 1
