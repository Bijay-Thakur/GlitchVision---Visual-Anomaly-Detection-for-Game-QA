"""Simple visualization helpers for the demo outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def make_contact_sheet(
    images_bgr: Sequence[np.ndarray],
    cols: int = 4,
    tile_size: int = 160,
    gap: int = 6,
    bg: int = 20,
) -> np.ndarray:
    """Arrange images into a simple grid contact sheet (BGR uint8)."""
    if not images_bgr:
        return np.full((tile_size, tile_size, 3), bg, dtype=np.uint8)

    n = len(images_bgr)
    cols = max(1, int(cols))
    rows = (n + cols - 1) // cols

    h = rows * tile_size + (rows + 1) * gap
    w = cols * tile_size + (cols + 1) * gap
    sheet = np.full((h, w, 3), bg, dtype=np.uint8)

    for i, img in enumerate(images_bgr):
        r, c = divmod(i, cols)
        y = gap + r * (tile_size + gap)
        x = gap + c * (tile_size + gap)
        tile = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        sheet[y : y + tile_size, x : x + tile_size] = tile

    return sheet


def make_score_plot(
    scores: np.ndarray,
    timestamps: np.ndarray,
    out_path: str | Path,
    highlights: Sequence[int] | None = None,
    title: str = "Anomaly score over time",
) -> Path:
    """Save a PNG plot of anomaly score vs. time with optional highlights."""
    # Lazy import so matplotlib is only required when plotting.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=110)
    ax.plot(timestamps, scores, color="#3b82f6", linewidth=1.4, label="score")
    if highlights:
        hl_t = timestamps[list(highlights)]
        hl_s = scores[list(highlights)]
        ax.scatter(hl_t, hl_s, color="#ef4444", s=36, zorder=3, label="top anomalies")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Anomaly score")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
