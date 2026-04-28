"""Per-run operational and optional supervised-evaluation metrics (JSON artifacts)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (float, int)) and (obj == obj):  # not NaN
        return obj
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(v, float) and v != v:
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return str(obj)


def build_run_metrics_payload(
    *,
    total_wall_sec: float,
    sample_wall_sec: float,
    embed_wall_sec: float,
    post_wall_sec: float,
    n_sampled_frames: int,
    mode: str,
    backbone: str,
    top_k: int,
    score_stats: Mapping[str, float],
    top_ranked_indices: Sequence[int],
) -> dict[str, Any]:
    """Numbers suitable for dashboards, README, and résumé bullets."""
    n = max(1, int(n_sampled_frames))
    out: dict[str, Any] = {
        "timing_sec": {
            "total": round(float(total_wall_sec), 4),
            "frame_sampling": round(float(sample_wall_sec), 4),
            "embedding": round(float(embed_wall_sec), 4),
            "score_rank_export": round(float(post_wall_sec), 4),
        },
        "throughput": {
            "sampled_frames_per_sec_end_to_end": round(n / max(total_wall_sec, 1e-9), 2),
            "embeddings_per_sec": round(n / max(embed_wall_sec, 1e-9), 2),
        },
        "scale": {
            "sampled_frames": n,
            "top_k_surface": int(top_k),
        },
        "model": {
            "mode": mode,
            "backbone": backbone,
        },
        "score_distribution": {k: round(float(v), 6) for k, v in score_stats.items()},
        "detection": {
            "top_anomaly_indices": [int(i) for i in top_ranked_indices],
        },
    }
    return out


def summarize_scores(smoothed: np.ndarray) -> dict[str, float]:
    """Descriptive stats on per-frame (smoothed) anomaly scores."""
    if smoothed.size == 0:
        return {}
    s = smoothed.astype(np.float64, copy=False)
    p90 = float(np.quantile(s, 0.90))
    p95 = float(np.quantile(s, 0.95))
    p99 = float(np.quantile(s, 0.99))
    med = float(np.median(s))
    mx = float(np.max(s))
    return {
        "min": float(np.min(s)),
        "mean": float(np.mean(s)),
        "median": med,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": mx,
        "top1_minus_median": float(mx - med),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2),
        encoding="utf-8",
    )
    return path
