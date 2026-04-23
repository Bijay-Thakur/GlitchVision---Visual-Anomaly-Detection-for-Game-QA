"""End-to-end orchestration for GlitchVision.

The pipeline is intentionally small and linear:

    source -> frame sampling -> ResNet-18 embeddings ->
    Isolation Forest -> ranked top-K frames + CSV + plot
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ..features import EmbeddingExtractor
from ..models import AnomalyDetector
from ..models.anomaly_detector import DetectorConfig
from ..processing import FrameExtractor, SampledFrame
from ..utils import (
    ensure_dir,
    make_score_plot,
    new_run_dir,
    normalize_scores,
    rank_top_k,
    save_frame_image,
    smooth_scores,
    write_results_csv,
)


ProgressCallback = Callable[[float, str], None]


@dataclass
class PipelineConfig:
    """All knobs for a single pipeline run."""

    interval_sec: float = 1.0
    image_size: int = 224
    max_frames: int = 600
    top_k: int = 12
    contamination: float = 0.05
    smoothing_window: int = 3
    min_gap_frames: int = 2
    batch_size: int = 8
    output_dir: Path = field(default_factory=lambda: Path("data/outputs"))


@dataclass
class AnomalyRecord:
    """One frame's worth of result metadata (mirrors a CSV row)."""

    rank: int
    frame_index: int
    source_frame_idx: int
    timestamp_sec: float
    anomaly_score: float
    normalized_score: float
    image_path: str
    source_type: str
    source_label: str


@dataclass
class PipelineResult:
    """Container for everything we want to surface to the UI."""

    run_dir: Path
    csv_path: Path
    plot_path: Optional[Path]
    top_records: List[AnomalyRecord]
    all_records: List[AnomalyRecord]
    total_sampled_frames: int
    score_min: float
    score_max: float
    source_type: str
    source_label: str


class GlitchVisionPipeline:
    """Builder-style pipeline. Instantiate once, call ``run(...)`` per video."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._extractor: Optional[EmbeddingExtractor] = None

    # ---------------------------------------------------------------
    # lazy-loaded heavy components
    # ---------------------------------------------------------------
    def _get_embedding_extractor(self) -> EmbeddingExtractor:
        if self._extractor is None:
            self._extractor = EmbeddingExtractor(device="cpu")
        return self._extractor

    # ---------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------
    def run(
        self,
        video_source: str,
        source_type: str,
        source_label: str,
        progress: Optional[ProgressCallback] = None,
    ) -> PipelineResult:
        """Execute the full pipeline on a single video source."""
        cfg = self.config
        report = progress or (lambda p, m: None)

        report(0.02, "Preparing output directory")
        ensure_dir(cfg.output_dir)
        run_dir = new_run_dir(cfg.output_dir, prefix="run")

        # -----------------------------------------------------------
        # 1. Sample frames
        # -----------------------------------------------------------
        report(0.05, "Opening video source")
        with FrameExtractor(
            source=video_source,
            interval_sec=cfg.interval_sec,
            image_size=cfg.image_size,
            max_frames=cfg.max_frames,
        ) as extractor:
            report(0.10, "Sampling frames")
            frames: List[SampledFrame] = []
            est_total = max(1, extractor.estimated_sample_count())

            for sf in extractor.iter_frames():
                frames.append(sf)
                # Cap progress at 45% during sampling.
                frac = 0.10 + 0.35 * min(1.0, len(frames) / est_total)
                if len(frames) % 5 == 0:
                    report(frac, f"Sampled {len(frames)} frames")

        if len(frames) < 2:
            raise RuntimeError(
                "Not enough frames were sampled from the video. "
                "Try a longer clip or a smaller frame interval."
            )

        report(0.45, f"Sampled {len(frames)} frames total")

        # -----------------------------------------------------------
        # 2. Extract embeddings
        # -----------------------------------------------------------
        report(0.48, "Loading ResNet-18 feature extractor")
        embedder = self._get_embedding_extractor()

        report(0.52, "Computing embeddings")
        embeddings = self._embed_in_batches(
            [f.image for f in frames],
            embedder,
            batch_size=cfg.batch_size,
            report=report,
            progress_start=0.52,
            progress_end=0.82,
        )

        # -----------------------------------------------------------
        # 3. Fit anomaly detector and score
        # -----------------------------------------------------------
        report(0.84, "Fitting Isolation Forest")
        detector = AnomalyDetector(
            DetectorConfig(contamination=cfg.contamination)
        )
        raw_scores = detector.fit_score(embeddings)

        smoothed = smooth_scores(raw_scores, window=cfg.smoothing_window)
        norm = normalize_scores(smoothed)

        # -----------------------------------------------------------
        # 4. Rank and save top-K
        # -----------------------------------------------------------
        report(0.90, "Ranking top anomalies")
        top_indices = rank_top_k(
            smoothed, k=cfg.top_k, min_gap=cfg.min_gap_frames
        )

        frames_dir = run_dir / "frames"
        top_records: List[AnomalyRecord] = []
        saved_image_paths: dict[int, str] = {}
        for rank, idx in enumerate(top_indices, start=1):
            sf = frames[idx]
            fname = (
                f"rank{rank:02d}_idx{sf.index:04d}_t{sf.timestamp_sec:07.2f}s.jpg"
            )
            img_path = frames_dir / fname
            save_frame_image(sf.image, img_path)
            rel = str(img_path.relative_to(run_dir))
            saved_image_paths[idx] = rel
            top_records.append(
                AnomalyRecord(
                    rank=rank,
                    frame_index=sf.index,
                    source_frame_idx=sf.source_frame_idx,
                    timestamp_sec=sf.timestamp_sec,
                    anomaly_score=float(smoothed[idx]),
                    normalized_score=float(norm[idx]),
                    image_path=rel,
                    source_type=source_type,
                    source_label=source_label,
                )
            )

        # Build an "all frames" record list for the full CSV export.
        all_records: List[AnomalyRecord] = []
        # Rank order for every frame (dense ranking by smoothed score desc).
        rank_by_index = {
            int(idx): r + 1
            for r, idx in enumerate(np.argsort(-smoothed, kind="stable").tolist())
        }
        for i, sf in enumerate(frames):
            all_records.append(
                AnomalyRecord(
                    rank=rank_by_index.get(i, -1),
                    frame_index=sf.index,
                    source_frame_idx=sf.source_frame_idx,
                    timestamp_sec=sf.timestamp_sec,
                    anomaly_score=float(smoothed[i]),
                    normalized_score=float(norm[i]),
                    image_path=saved_image_paths.get(i, ""),
                    source_type=source_type,
                    source_label=source_label,
                )
            )

        # -----------------------------------------------------------
        # 5. Persist CSV + score plot
        # -----------------------------------------------------------
        report(0.94, "Writing CSV and score plot")
        csv_path = run_dir / "anomalies.csv"
        write_results_csv(
            [record_to_row(r) for r in all_records],
            csv_path,
            fieldnames=[
                "rank",
                "frame_index",
                "source_frame_idx",
                "timestamp_sec",
                "anomaly_score",
                "normalized_score",
                "image_path",
                "source_type",
                "source_label",
            ],
        )

        plot_path: Optional[Path] = None
        try:
            timestamps = np.array([f.timestamp_sec for f in frames], dtype=np.float32)
            plot_path = make_score_plot(
                scores=smoothed,
                timestamps=timestamps,
                out_path=run_dir / "score_plot.png",
                highlights=top_indices,
                title="GlitchVision anomaly scores",
            )
        except Exception:
            # Plotting is a nice-to-have; never let it kill the demo.
            plot_path = None

        report(1.0, "Done")
        return PipelineResult(
            run_dir=run_dir,
            csv_path=csv_path,
            plot_path=plot_path,
            top_records=top_records,
            all_records=all_records,
            total_sampled_frames=len(frames),
            score_min=float(np.min(smoothed)),
            score_max=float(np.max(smoothed)),
            source_type=source_type,
            source_label=source_label,
        )

    # ---------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------
    @staticmethod
    def _embed_in_batches(
        images_bgr: List[np.ndarray],
        embedder: EmbeddingExtractor,
        batch_size: int,
        report: ProgressCallback,
        progress_start: float,
        progress_end: float,
    ) -> np.ndarray:
        n = len(images_bgr)
        if n == 0:
            return np.zeros((0, embedder.embedding_dim), dtype=np.float32)

        span = max(1e-6, progress_end - progress_start)
        out_chunks: List[np.ndarray] = []
        done = 0
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            batch = np.stack(images_bgr[start:end], axis=0)
            out_chunks.append(embedder.embed(batch))
            done = end
            report(
                progress_start + span * (done / n),
                f"Embedding frames {done}/{n}",
            )
        return np.concatenate(out_chunks, axis=0)


def record_to_row(r: AnomalyRecord) -> dict:
    """Convert an ``AnomalyRecord`` to a plain dict for CSV writing."""
    return {
        "rank": r.rank,
        "frame_index": r.frame_index,
        "source_frame_idx": r.source_frame_idx,
        "timestamp_sec": round(r.timestamp_sec, 3),
        "anomaly_score": round(r.anomaly_score, 6),
        "normalized_score": round(r.normalized_score, 6),
        "image_path": r.image_path,
        "source_type": r.source_type,
        "source_label": r.source_label,
    }
