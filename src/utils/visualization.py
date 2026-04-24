"""Plot helpers for the demo outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


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
