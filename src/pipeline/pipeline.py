"""End-to-end orchestration for GlitchVision.

The pipeline is intentionally small and linear. Three anomaly-scoring
modes are supported behind a single :class:`GlitchVisionPipeline` API:

- ``within_clip_iforest`` (the original baseline MVP mode)
- ``reference_distance`` (candidate vs. reference embedding bank)
- ``hybrid``              (normalized blend of the two)

Each run produces a timestamped folder under ``data/outputs/`` with:

    run_<ts>/
        frames/                 # top-K thumbnail JPEGs
        anomalies.csv           # per-frame scores
        segments.csv            # top anomalous segments
        score_plot.png          # per-frame score curve (if matplotlib)
        report.md               # human-readable summary
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Literal, Optional

import numpy as np

from ..features import EmbeddingExtractor
from ..models import AnomalyDetector
from ..models.anomaly_detector import DetectorConfig
from ..models.hybrid_scorer import HybridScoreResult, hybrid_score
from ..models.reference_scorer import ReferenceScorer, ReferenceScorerConfig
from ..processing import FrameExtractor, SampledFrame
from ..reference import FrameRef, ReferenceBank, build_reference_bank
from ..reporting import (
    ReportFrame,
    ReportSegment,
    RunReport,
    write_report,
)
from ..utils import (
    build_segments,
    ensure_dir,
    make_score_plot,
    new_run_dir,
    normalize_scores,
    rank_segments,
    rank_top_k,
    save_frame_image,
    segment_to_row,
    smooth_scores,
    write_results_csv,
)


ProgressCallback = Callable[[float, str], None]

ScoringMode = Literal["within_clip_iforest", "reference_distance", "hybrid"]


@dataclass
class PipelineConfig:
    """All knobs for a single pipeline run."""

    # -- frame sampling / model basics --
    interval_sec: float = 1.0
    image_size: int = 224
    max_frames: int = 600
    batch_size: int = 8
    backbone: str = "resnet18"

    # -- scoring --
    mode: ScoringMode = "within_clip_iforest"
    top_k: int = 12
    contamination: float = 0.05
    smoothing_window: int = 3
    min_gap_frames: int = 2

    # -- reference / hybrid mode --
    reference_k: int = 5
    reference_metric: str = "cosine"
    hybrid_weight_within: float = 0.5
    hybrid_weight_reference: float = 0.5

    # -- segment analysis --
    segment_window: int = 5
    segment_top_k: int = 5

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
    mode: str
    within_score: float = float("nan")
    reference_score: float = float("nan")


@dataclass
class PipelineResult:
    """Container for everything we want to surface to the UI."""

    run_dir: Path
    csv_path: Path
    segment_csv_path: Optional[Path]
    plot_path: Optional[Path]
    report_path: Optional[Path]
    top_records: List[AnomalyRecord]
    all_records: List[AnomalyRecord]
    top_segments: List
    total_sampled_frames: int
    score_min: float
    score_max: float
    source_type: str
    source_label: str
    mode: str


class GlitchVisionPipeline:
    """Builder-style pipeline. Instantiate once, call ``run(...)`` per video."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._extractor: Optional[EmbeddingExtractor] = None
        self._extractor_backbone: Optional[str] = None

    # ---------------------------------------------------------------
    # lazy-loaded heavy components
    # ---------------------------------------------------------------
    def _get_embedding_extractor(self) -> EmbeddingExtractor:
        # Rebuild when the caller switches backbone between runs.
        if (
            self._extractor is None
            or self._extractor_backbone != self.config.backbone
        ):
            self._extractor = EmbeddingExtractor(
                device="cpu", backbone=self.config.backbone
            )
            self._extractor_backbone = self._extractor.backbone_name
        return self._extractor

    # ---------------------------------------------------------------
    # reference bank construction
    # ---------------------------------------------------------------
    def build_reference(
        self,
        video_sources: List[tuple[str, str]],
        *,
        out_dir: Optional[Path] = None,
        progress: Optional[ProgressCallback] = None,
        notes: str = "",
    ) -> ReferenceBank:
        """Build (and optionally persist) a reference bank from videos.

        Parameters
        ----------
        video_sources:
            List of ``(path_or_url, source_label)`` tuples.
        out_dir:
            If provided, the bank is saved under ``out_dir``.
        """
        cfg = self.config
        report = progress or (lambda p, m: None)
        embedder = self._get_embedding_extractor()

        all_embeddings: List[np.ndarray] = []
        all_frames: List[FrameRef] = []
        labels: List[str] = []

        for video_idx, (src, label) in enumerate(video_sources):
            labels.append(label)
            report(
                0.02 + 0.9 * (video_idx / max(1, len(video_sources))),
                f"Reading reference video {video_idx + 1}/{len(video_sources)}: {label}",
            )

            with FrameExtractor(
                source=src,
                interval_sec=cfg.interval_sec,
                image_size=cfg.image_size,
                max_frames=cfg.max_frames,
            ) as extractor:
                frames = list(extractor.iter_frames())

            if len(frames) < 1:
                continue

            embeddings = self._embed_in_batches(
                [f.image for f in frames],
                embedder,
                batch_size=cfg.batch_size,
                report=lambda p, m: None,
                progress_start=0.0,
                progress_end=0.0,
            )
            all_embeddings.append(embeddings)
            for sf in frames:
                all_frames.append(
                    FrameRef(
                        source_label=label,
                        frame_index=sf.index,
                        timestamp_sec=sf.timestamp_sec,
                    )
                )

        if not all_embeddings:
            raise RuntimeError(
                "No frames were extracted from any reference video."
            )

        matrix = np.concatenate(all_embeddings, axis=0)
        bank = build_reference_bank(
            embeddings=matrix,
            frames=all_frames,
            backbone=embedder.backbone_name,
            image_size=cfg.image_size,
            interval_sec=cfg.interval_sec,
            source_videos=labels,
            notes=notes,
        )

        if out_dir is not None:
            ensure_dir(out_dir)
            bank.save(out_dir)

        report(1.0, f"Built reference bank with {bank.size} frames.")
        return bank

    # ---------------------------------------------------------------
    # main run
    # ---------------------------------------------------------------
    def run(
        self,
        video_source: str,
        source_type: str,
        source_label: str,
        progress: Optional[ProgressCallback] = None,
        reference_bank: Optional[ReferenceBank] = None,
    ) -> PipelineResult:
        """Execute the full pipeline on a single video source."""
        cfg = self.config
        report = progress or (lambda p, m: None)

        mode = cfg.mode
        if mode in {"reference_distance", "hybrid"} and reference_bank is None:
            raise ValueError(
                f"Mode '{mode}' requires a reference bank. "
                "Pass `reference_bank=...` or switch to within_clip_iforest."
            )

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
                frac = 0.10 + 0.30 * min(1.0, len(frames) / est_total)
                if len(frames) % 5 == 0:
                    report(frac, f"Sampled {len(frames)} frames")

        if len(frames) < 2:
            raise RuntimeError(
                "Not enough frames were sampled from the video. "
                "Try a longer clip or a smaller frame interval."
            )

        report(0.40, f"Sampled {len(frames)} frames total")

        # -----------------------------------------------------------
        # 2. Extract embeddings
        # -----------------------------------------------------------
        report(0.45, "Loading feature extractor")
        embedder = self._get_embedding_extractor()

        if reference_bank is not None and reference_bank.embedding_dim != embedder.embedding_dim:
            raise RuntimeError(
                "Reference bank embedding dim "
                f"({reference_bank.embedding_dim}) does not match candidate "
                f"({embedder.embedding_dim}). Rebuild the bank with the same "
                "backbone."
            )

        report(0.48, "Computing embeddings")
        embeddings = self._embed_in_batches(
            [f.image for f in frames],
            embedder,
            batch_size=cfg.batch_size,
            report=report,
            progress_start=0.48,
            progress_end=0.75,
        )

        # -----------------------------------------------------------
        # 3. Score according to the selected mode
        # -----------------------------------------------------------
        within_raw: Optional[np.ndarray] = None
        reference_raw: Optional[np.ndarray] = None

        if mode in {"within_clip_iforest", "hybrid"}:
            report(0.78, "Fitting Isolation Forest (within-clip)")
            detector = AnomalyDetector(
                DetectorConfig(contamination=cfg.contamination)
            )
            within_raw = detector.fit_score(embeddings)

        if mode in {"reference_distance", "hybrid"}:
            assert reference_bank is not None  # guarded above
            report(0.82, "Scoring candidates against reference bank")
            scorer = ReferenceScorer(
                reference_embeddings=reference_bank.embeddings,
                config=ReferenceScorerConfig(
                    k=cfg.reference_k, metric=cfg.reference_metric  # type: ignore[arg-type]
                ),
            )
            reference_raw = scorer.score(embeddings)

        if mode == "within_clip_iforest":
            assert within_raw is not None
            raw_scores = within_raw
            hybrid_result: Optional[HybridScoreResult] = None
        elif mode == "reference_distance":
            assert reference_raw is not None
            raw_scores = reference_raw
            hybrid_result = None
        elif mode == "hybrid":
            assert within_raw is not None and reference_raw is not None
            hybrid_result = hybrid_score(
                within_raw,
                reference_raw,
                weight_within=cfg.hybrid_weight_within,
                weight_reference=cfg.hybrid_weight_reference,
            )
            raw_scores = hybrid_result.hybrid
        else:
            raise ValueError(f"Unknown scoring mode: {mode!r}")

        smoothed = smooth_scores(raw_scores, window=cfg.smoothing_window)
        norm = normalize_scores(smoothed)

        # -----------------------------------------------------------
        # 4. Rank top-K frames
        # -----------------------------------------------------------
        report(0.88, "Ranking top anomalies")
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
            rel = str(img_path.relative_to(run_dir)).replace("\\", "/")
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
                    mode=mode,
                    within_score=(
                        float(within_raw[idx]) if within_raw is not None else float("nan")
                    ),
                    reference_score=(
                        float(reference_raw[idx]) if reference_raw is not None else float("nan")
                    ),
                )
            )

        # Full per-frame records for CSV export.
        all_records: List[AnomalyRecord] = []
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
                    mode=mode,
                    within_score=(
                        float(within_raw[i]) if within_raw is not None else float("nan")
                    ),
                    reference_score=(
                        float(reference_raw[i]) if reference_raw is not None else float("nan")
                    ),
                )
            )

        # -----------------------------------------------------------
        # 5. Segment analysis
        # -----------------------------------------------------------
        report(0.92, "Building segments")
        timestamps = np.array([f.timestamp_sec for f in frames], dtype=np.float32)
        segments = build_segments(
            scores=smoothed,
            timestamps=timestamps,
            window_size=cfg.segment_window,
        )
        top_segments = rank_segments(segments, top_k=cfg.segment_top_k)

        # -----------------------------------------------------------
        # 6. Persist CSVs + plot
        # -----------------------------------------------------------
        report(0.95, "Writing CSVs and plot")
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
                "within_score",
                "reference_score",
                "image_path",
                "source_type",
                "source_label",
                "mode",
            ],
        )

        segment_csv_path: Optional[Path] = None
        if segments:
            segment_csv_path = run_dir / "segments.csv"
            write_results_csv(
                [segment_to_row(s) for s in top_segments or segments],
                segment_csv_path,
                fieldnames=[
                    "rank",
                    "segment_id",
                    "start_frame",
                    "end_frame",
                    "start_time_sec",
                    "end_time_sec",
                    "mean_score",
                    "max_score",
                    "representative_frame",
                ],
            )

        plot_path: Optional[Path] = None
        try:
            plot_path = make_score_plot(
                scores=smoothed,
                timestamps=timestamps,
                out_path=run_dir / "score_plot.png",
                highlights=top_indices,
                title=f"GlitchVision anomaly scores ({mode})",
            )
        except Exception:
            plot_path = None

        # -----------------------------------------------------------
        # 7. Report
        # -----------------------------------------------------------
        report_path: Optional[Path] = None
        try:
            report_obj = RunReport(
                run_dir=run_dir,
                mode=mode,
                source_type=source_type,
                source_label=source_label,
                backbone=embedder.backbone_name,
                total_sampled_frames=len(frames),
                config_summary={
                    "interval_sec": cfg.interval_sec,
                    "image_size": cfg.image_size,
                    "max_frames": cfg.max_frames,
                    "top_k": cfg.top_k,
                    "contamination": cfg.contamination,
                    "smoothing_window": cfg.smoothing_window,
                    "min_gap_frames": cfg.min_gap_frames,
                    "segment_window": cfg.segment_window,
                    "segment_top_k": cfg.segment_top_k,
                    "reference_k": cfg.reference_k,
                    "reference_metric": cfg.reference_metric,
                },
                top_frames=[
                    ReportFrame(
                        rank=r.rank,
                        timestamp_sec=r.timestamp_sec,
                        score=r.anomaly_score,
                        normalized_score=r.normalized_score,
                        image_path=r.image_path,
                    )
                    for r in top_records
                ],
                top_segments=[
                    ReportSegment(
                        rank=s.rank,
                        start_time_sec=s.start_time_sec,
                        end_time_sec=s.end_time_sec,
                        mean_score=s.mean_score,
                        max_score=s.max_score,
                        representative_frame=s.representative_frame,
                    )
                    for s in top_segments
                ],
                csv_path=str(csv_path.relative_to(run_dir)).replace("\\", "/"),
                segment_csv_path=(
                    str(segment_csv_path.relative_to(run_dir)).replace("\\", "/")
                    if segment_csv_path
                    else None
                ),
                plot_path=(
                    str(plot_path.relative_to(run_dir)).replace("\\", "/")
                    if plot_path
                    else None
                ),
                reference_bank_path=(
                    ",".join(reference_bank.source_videos) if reference_bank else None
                ),
                hybrid_weights=(
                    {
                        "weight_within": hybrid_result.weight_within,
                        "weight_reference": hybrid_result.weight_reference,
                    }
                    if hybrid_result is not None
                    else None
                ),
            )
            report_path = write_report(report_obj)
        except Exception:
            # Report writing is a nice-to-have; do not fail the run over it.
            report_path = None

        report(1.0, "Done")
        return PipelineResult(
            run_dir=run_dir,
            csv_path=csv_path,
            segment_csv_path=segment_csv_path,
            plot_path=plot_path,
            report_path=report_path,
            top_records=top_records,
            all_records=all_records,
            top_segments=top_segments,
            total_sampled_frames=len(frames),
            score_min=float(np.min(smoothed)),
            score_max=float(np.max(smoothed)),
            source_type=source_type,
            source_label=source_label,
            mode=mode,
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
            if span > 0:
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
        "within_score": (
            round(r.within_score, 6) if r.within_score == r.within_score else ""
        ),
        "reference_score": (
            round(r.reference_score, 6) if r.reference_score == r.reference_score else ""
        ),
        "image_path": r.image_path,
        "source_type": r.source_type,
        "source_label": r.source_label,
        "mode": r.mode,
    }
