"""Build a lightweight per-run markdown report.

The report lives next to the CSV/plot artifacts inside each run
directory, so a human reviewer (or CI artifact viewer) can open one file
and immediately understand what the pipeline did, what it found, and
what the caveats are.

It is deliberately plain markdown — no HTML, no JS — so it renders on
GitHub, in VS Code, and anywhere else markdown is supported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


@dataclass
class ReportFrame:
    rank: int
    timestamp_sec: float
    score: float
    normalized_score: float
    image_path: str   # relative to run_dir


@dataclass
class ReportSegment:
    rank: int
    start_time_sec: float
    end_time_sec: float
    mean_score: float
    max_score: float
    representative_frame: int


@dataclass
class RunReport:
    """All the information the report writer needs."""

    run_dir: Path
    mode: str
    source_type: str
    source_label: str
    backbone: str
    total_sampled_frames: int
    config_summary: dict
    top_frames: List[ReportFrame] = field(default_factory=list)
    top_segments: List[ReportSegment] = field(default_factory=list)
    csv_path: Optional[str] = None
    segment_csv_path: Optional[str] = None
    plot_path: Optional[str] = None
    reference_bank_path: Optional[str] = None
    hybrid_weights: Optional[dict] = None
    notes: str = ""


def _fmt_time(t: float) -> str:
    t = max(0.0, float(t))
    m = int(t // 60)
    s = t - 60 * m
    return f"{m:02d}:{s:05.2f}"


def write_report(report: RunReport, filename: str = "report.md") -> Path:
    """Write a markdown report into ``report.run_dir`` and return its path."""
    out = Path(report.run_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []

    lines.append(f"# GlitchVision run report")
    lines.append("")
    lines.append(f"- **Generated:** {now}")
    lines.append(f"- **Mode:** `{report.mode}`")
    lines.append(f"- **Backbone:** `{report.backbone}`")
    lines.append(f"- **Source type:** `{report.source_type}`")
    lines.append(f"- **Source label:** {report.source_label}")
    lines.append(f"- **Sampled frames:** {report.total_sampled_frames}")
    if report.reference_bank_path:
        lines.append(f"- **Reference bank:** `{report.reference_bank_path}`")
    if report.hybrid_weights:
        w = report.hybrid_weights
        lines.append(
            "- **Hybrid weights:** "
            f"within-clip = {w.get('weight_within', 0):.2f}, "
            f"reference = {w.get('weight_reference', 0):.2f}"
        )
    lines.append("")

    # --- Config ---
    lines.append("## Run configuration")
    lines.append("")
    lines.append("| Key | Value |")
    lines.append("| --- | --- |")
    for k in sorted(report.config_summary.keys()):
        lines.append(f"| `{k}` | {report.config_summary[k]} |")
    lines.append("")

    # --- Artifacts ---
    lines.append("## Output artifacts")
    lines.append("")
    if report.csv_path:
        lines.append(f"- Per-frame CSV: `{report.csv_path}`")
    if report.segment_csv_path:
        lines.append(f"- Segment CSV: `{report.segment_csv_path}`")
    if report.plot_path:
        lines.append(f"- Score plot: `{report.plot_path}`")
    lines.append(f"- Report: `{filename}`")
    lines.append("")

    # --- Top frames ---
    lines.append("## Top anomalous frames")
    lines.append("")
    if report.top_frames:
        lines.append("| Rank | Time | Score | Norm. score | Thumbnail |")
        lines.append("| ---: | ---: | ---: | ---: | --- |")
        for f in report.top_frames:
            thumb = f"![frame]({f.image_path})" if f.image_path else ""
            lines.append(
                f"| {f.rank} | {_fmt_time(f.timestamp_sec)} | "
                f"{f.score:.4f} | {f.normalized_score:.3f} | {thumb} |"
            )
    else:
        lines.append("_No top frames surfaced._")
    lines.append("")

    # --- Top segments ---
    lines.append("## Top anomalous segments")
    lines.append("")
    if report.top_segments:
        lines.append("| Rank | Interval | Mean | Max | Representative frame |")
        lines.append("| ---: | --- | ---: | ---: | ---: |")
        for s in report.top_segments:
            lines.append(
                f"| {s.rank} | {_fmt_time(s.start_time_sec)} – "
                f"{_fmt_time(s.end_time_sec)} | "
                f"{s.mean_score:.4f} | {s.max_score:.4f} | "
                f"{s.representative_frame} |"
            )
    else:
        lines.append("_Segment analysis not available for this run._")
    lines.append("")

    # --- Interpretation + limitations ---
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append(
        "- Higher scores are *more anomalous*. Scores are produced by "
        "the selected mode and then min-max normalized for display."
    )
    if report.mode == "within_clip_iforest":
        lines.append(
            "- **Within-clip mode** flags frames that look different from "
            "the rest of the same clip. Cutscenes, menus, and fade-to-black "
            "frames can score high without being bugs."
        )
    elif report.mode == "reference_distance":
        lines.append(
            "- **Reference-distance mode** flags frames that look different "
            "from the known-good reference bank. A score spike here means "
            "the content has drifted from the reference distribution."
        )
    elif report.mode == "hybrid":
        lines.append(
            "- **Hybrid mode** blends within-clip novelty and reference "
            "drift. Intended for triage when you have both a new build and "
            "a reference baseline."
        )
    if report.notes:
        lines.append("")
        lines.append(report.notes)
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Unsupervised scoring identifies *statistical* outliers, not "
        "confirmed bugs. This report is an aid to human review."
    )
    lines.append(
        "- Frame sampling is fixed-interval; short-duration glitches "
        "(< one interval) can be missed."
    )
    lines.append(
        "- The reference bank only generalizes as well as its source "
        "footage. If the reference doesn't cover a scene, that scene will "
        "look anomalous."
    )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
