from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_frame_extractor_duration_fraction_truncates(tmp_path: Path):
    try:
        import cv2
    except ImportError:
        pytest.skip("opencv missing")

    from src.processing.frame_extractor import FrameExtractor

    path = tmp_path / "t.mp4"
    w, h, fps, n = 32, 32, 10, 100
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened()
    try:
        for i in range(n):
            frame = np.full((h, w, 3), (i % 255, 40, 80), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    with FrameExtractor(
        str(path),
        interval_sec=1.0 / fps,
        image_size=32,
        max_frames=500,
        duration_fraction=0.25,
    ) as ex:
        frames = list(ex.iter_frames())

    assert len(frames) < n
    cap = (n / fps) * 0.25
    assert frames[-1].timestamp_sec <= cap + (1.0 / fps) + 1e-6
