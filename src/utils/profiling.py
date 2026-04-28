"""Small profiling helpers for benchmark latency, memory, and cost reports."""
from __future__ import annotations

import csv
import json
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional


class StageProfiler:
    """Track stage durations plus lightweight memory/storage counters."""

    def __init__(self) -> None:
        self.metrics: Dict[str, float] = {}
        self._started = False
        self._total_start = 0.0

    def start(self) -> None:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._started = True
        self._total_start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self._started:
            self.start()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.metrics[f"{name}_sec"] = self.metrics.get(f"{name}_sec", 0.0) + (
                time.perf_counter() - start
            )
            self._update_memory()

    def _update_memory(self) -> None:
        current, peak = tracemalloc.get_traced_memory()
        self.metrics["peak_memory_mb"] = max(
            self.metrics.get("peak_memory_mb", 0.0),
            peak / (1024 * 1024),
        )
        try:
            import psutil

            rss = psutil.Process().memory_info().rss / (1024 * 1024)
            self.metrics["rss_memory_mb"] = max(self.metrics.get("rss_memory_mb", 0.0), rss)
        except Exception:
            self.metrics.setdefault("rss_memory_mb", 0.0)
        self.metrics["current_tracemalloc_mb"] = current / (1024 * 1024)

    def finish(
        self,
        samples: int = 0,
        model_artifact: str | Path | None = None,
        reference_bank: str | Path | None = None,
        videos_probed: int = 0,
        frames_sampled: int = 0,
    ) -> Dict[str, float]:
        self._update_memory()
        total = time.perf_counter() - self._total_start if self._started else 0.0
        self.metrics["total_sec"] = total
        if samples:
            self.metrics["samples_per_sec"] = float(samples / max(total, 1e-12))
            self.metrics["ms_per_sample"] = float((total * 1000.0) / samples)
        else:
            self.metrics.setdefault("samples_per_sec", 0.0)
            self.metrics.setdefault("ms_per_sample", 0.0)
        self.metrics["model_artifact_size_mb"] = _path_size_mb(model_artifact)
        self.metrics["reference_bank_size_mb"] = _path_size_mb(reference_bank)
        self.metrics["videos_probed"] = float(videos_probed)
        self.metrics["frames_sampled"] = float(frames_sampled)
        return dict(self.metrics)


def _path_size_mb(path: str | Path | None) -> float:
    if path is None:
        return 0.0
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)


def write_profile_reports(
    metrics: Dict[str, float],
    out_dir: str | Path,
    before_training: Optional[Dict[str, float]] = None,
    after_training: Optional[Dict[str, float]] = None,
) -> tuple[Path, Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "profiling_report.json"
    csv_path = out / "profiling_report.csv"
    cost_path = out / "cost_report.md"

    payload = {
        "overall": metrics,
        "before_training": before_training or {},
        "after_training": after_training or {},
        "api_cost_usd": 0.0,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    keys = sorted(metrics)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scope"] + keys)
        writer.writeheader()
        writer.writerow({"scope": "overall", **metrics})

    lines = [
        "# GlitchVision Benchmark Cost Report",
        "",
        "- API cost: $0.00 (no external paid APIs are used by default).",
        f"- Runtime: {metrics.get('total_sec', 0.0):.2f} sec.",
        f"- Peak RAM: {metrics.get('peak_memory_mb', 0.0):.2f} MB (tracemalloc).",
        f"- RSS RAM: {metrics.get('rss_memory_mb', 0.0):.2f} MB when psutil is available.",
        f"- Disk storage, model artifact: {metrics.get('model_artifact_size_mb', 0.0):.3f} MB.",
        f"- Disk storage, reference bank: {metrics.get('reference_bank_size_mb', 0.0):.3f} MB.",
        f"- Videos probed: {int(metrics.get('videos_probed', 0.0))}.",
        f"- Frames sampled: {int(metrics.get('frames_sampled', 0.0))}.",
        "",
        "Before/after training numbers compare baseline scoring with the lightweight trained classifier when available.",
    ]
    cost_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, cost_path
