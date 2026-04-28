"""IO helpers: output directories, saving frames, writing CSV."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_run_dir(base_dir: str | Path, prefix: str = "run") -> Path:
    """Create a unique, timestamped output directory for a single run."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = Path(base_dir) / f"{prefix}_{ts}"
    # If the user clicks twice in the same second, append a counter.
    counter = 1
    while run.exists():
        run = Path(base_dir) / f"{prefix}_{ts}_{counter}"
        counter += 1
    run.mkdir(parents=True, exist_ok=True)
    (run / "frames").mkdir(parents=True, exist_ok=True)
    return run


def save_frame_image(image_bgr: np.ndarray, out_path: str | Path) -> Path:
    """Write a BGR image to disk as JPEG/PNG inferred from extension."""
    import cv2

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), image_bgr)
    if not ok:
        raise IOError(f"Failed to write image to {out}")
    return out


def write_results_csv(
    rows: Sequence[Mapping[str, object]],
    out_path: str | Path,
    fieldnames: Iterable[str] | None = None,
) -> Path:
    """Write a list of dict rows to CSV. Returns the written path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        if not rows:
            fieldnames = []
        else:
            fieldnames = list(rows[0].keys())

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out
