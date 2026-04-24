"""GlitchVision — Streamlit demo UI.

Input modes:
    1. YouTube URL (primary)   — resolve a stream via yt-dlp, no persistent
       download of the source video.
    2. Local file upload (fallback) — for reliability when a YouTube stream
       isn't OpenCV-compatible, or when offline.

Scoring modes:
    * within_clip_iforest  — the original MVP baseline.
    * reference_distance   — compare candidate frames to a known-good bank.
    * hybrid               — normalized blend of both.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import DEFAULTS, OUTPUTS_DIR, REFERENCE_BANKS_DIR  # noqa: E402
from src.ingestion import (  # noqa: E402
    YouTubeStreamError,
    resolve_youtube_stream,
    save_uploaded_file,
)
from src.pipeline import GlitchVisionPipeline, PipelineConfig  # noqa: E402
from src.reference import ReferenceBank  # noqa: E402


st.set_page_config(
    page_title="GlitchVision — Visual Regression Triage",
    page_icon="🎮",
    layout="wide",
)


# ---------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading backbone…")
def get_pipeline(backbone: str) -> GlitchVisionPipeline:
    """One pipeline per (backbone) so the model stays warm between runs."""
    cfg = PipelineConfig(backbone=backbone)
    return GlitchVisionPipeline(cfg)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("GlitchVision")
st.caption(
    "Visual regression triage for gameplay / simulation builds. "
    "ResNet-18 embeddings + Isolation Forest + optional reference baseline. "
    "CPU-first."
)

with st.expander("What is this?", expanded=False):
    st.markdown(
        """
        **GlitchVision** samples frames from a video, embeds them with a
        pretrained backbone (default ResNet-18), and surfaces suspicious
        frames and segments using one of three modes:

        - **Within-clip (Isolation Forest)** — flags frames that look
          different from the rest of the *same* clip. Good for one-off
          captures with no reference.
        - **Reference distance (kNN)** — compares each candidate frame
          to a saved "known-good" reference embedding bank. This is
          the business path: *visual regression triage*.
        - **Hybrid** — normalized blend of both scores, for when you
          want novelty *and* drift detection in one pass.

        This is an honest, CPU-first MVP. It surfaces suspicious moments
        for a human to eyeball — not a drop-in bug classifier.
        """
    )


# ---------------------------------------------------------------------
# Sidebar: tabs for Run / Reference / Benchmark
# ---------------------------------------------------------------------
st.sidebar.header("Run controls")

mode_label_to_id = {
    "Within-clip (baseline)": "within_clip_iforest",
    "Reference distance": "reference_distance",
    "Hybrid (within + reference)": "hybrid",
}
mode_label = st.sidebar.selectbox(
    "Scoring mode",
    options=list(mode_label_to_id.keys()),
    index=0,
    help="Reference / hybrid modes require a reference embedding bank.",
)
mode_id = mode_label_to_id[mode_label]

input_mode = st.sidebar.radio(
    "Input source",
    options=["YouTube URL", "Local upload (fallback)"],
    index=0,
    help="YouTube is the primary path. Local upload is a reliable fallback.",
)

backbone = st.sidebar.selectbox(
    "Backbone",
    options=["resnet18", "dino", "clip"],
    index=0,
    help="resnet18 is the default. dino/clip are optional and require their "
    "own dependencies; if missing, the app falls back to resnet18.",
)

frame_interval = st.sidebar.slider(
    "Frame interval (seconds)",
    min_value=0.25,
    max_value=5.0,
    value=float(DEFAULTS.frame_interval_sec),
    step=0.25,
    help="Lower = more frames analyzed, slower run.",
)

top_k = st.sidebar.slider(
    "Top-K anomalies",
    min_value=3,
    max_value=30,
    value=int(DEFAULTS.top_k),
    step=1,
)

contamination = st.sidebar.slider(
    "Contamination (within-clip only)",
    min_value=0.01,
    max_value=0.20,
    value=float(DEFAULTS.contamination),
    step=0.01,
)

max_frames = st.sidebar.slider(
    "Max frames to analyze",
    min_value=60,
    max_value=1200,
    value=int(DEFAULTS.max_frames),
    step=60,
)

smoothing = st.sidebar.slider(
    "Temporal smoothing window",
    min_value=1,
    max_value=9,
    value=int(DEFAULTS.smoothing_window),
    step=2,
    help="1 disables smoothing.",
)

segment_window = st.sidebar.slider(
    "Segment window (sampled frames)",
    min_value=2,
    max_value=20,
    value=int(DEFAULTS.segment_window),
    step=1,
)
segment_top_k = st.sidebar.slider(
    "Segment top-K",
    min_value=1,
    max_value=15,
    value=int(DEFAULTS.segment_top_k),
    step=1,
)

reference_k = st.sidebar.slider(
    "Reference kNN (k)",
    min_value=1,
    max_value=20,
    value=int(DEFAULTS.reference_k),
    step=1,
)

if mode_id == "hybrid":
    hybrid_w_within = st.sidebar.slider(
        "Hybrid weight — within",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULTS.hybrid_weight_within),
        step=0.05,
    )
    hybrid_w_ref = st.sidebar.slider(
        "Hybrid weight — reference",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULTS.hybrid_weight_reference),
        step=0.05,
    )
else:
    hybrid_w_within = float(DEFAULTS.hybrid_weight_within)
    hybrid_w_ref = float(DEFAULTS.hybrid_weight_reference)


# ---------------------------------------------------------------------
# Reference bank management
# ---------------------------------------------------------------------
REFERENCE_BANKS_DIR.mkdir(parents=True, exist_ok=True)


def _list_reference_banks() -> list[Path]:
    if not REFERENCE_BANKS_DIR.exists():
        return []
    return sorted(
        [p for p in REFERENCE_BANKS_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


selected_bank: Optional[ReferenceBank] = None
selected_bank_path: Optional[Path] = None

if mode_id in {"reference_distance", "hybrid"}:
    st.subheader("Reference bank")

    banks = _list_reference_banks()
    bank_names = [b.name for b in banks]

    if bank_names:
        choice = st.selectbox(
            "Load an existing reference bank",
            options=["<none>"] + bank_names,
            index=0,
        )
        if choice != "<none>":
            selected_bank_path = REFERENCE_BANKS_DIR / choice
            try:
                selected_bank = ReferenceBank.load(selected_bank_path)
                st.caption(
                    f"Loaded `{choice}` · {selected_bank.size} frames · "
                    f"backbone = `{selected_bank.backbone}`"
                )
            except Exception as exc:
                st.error(f"Could not load reference bank: {exc}")
    else:
        st.info(
            "No reference banks saved yet. Use the builder below to create one."
        )

    with st.expander("Build a new reference bank", expanded=(not bank_names)):
        bank_name = st.text_input(
            "Name for the new bank",
            value="known_good_v1",
            help="Stored under `data/reference_banks/<name>/`.",
        )
        ref_upload = st.file_uploader(
            "Upload one or more known-good reference videos",
            type=["mp4", "mov", "mkv", "webm", "avi", "m4v"],
            accept_multiple_files=True,
            key="ref_upload",
        )
        build_clicked = st.button("Build reference bank", type="secondary")

        if build_clicked:
            if not ref_upload:
                st.warning("Please upload at least one reference video.")
            elif not bank_name.strip():
                st.warning("Please name the bank.")
            else:
                pipeline_for_ref = get_pipeline(backbone)
                pipeline_for_ref.config = PipelineConfig(
                    interval_sec=float(frame_interval),
                    image_size=int(DEFAULTS.image_size),
                    max_frames=int(max_frames),
                    backbone=backbone,
                )
                cleanups = []
                sources = []
                for up in ref_upload:
                    lv = save_uploaded_file(up, original_name=up.name)
                    cleanups.append(lv.cleanup)
                    sources.append((str(lv.path), up.name))

                progress_bar = st.progress(0.0, text="Building bank…")
                try:
                    bank = pipeline_for_ref.build_reference(
                        sources,
                        out_dir=REFERENCE_BANKS_DIR / bank_name.strip(),
                        progress=lambda p, m: progress_bar.progress(
                            min(max(p, 0.0), 1.0), text=m
                        ),
                    )
                    st.success(
                        f"Saved reference bank '{bank_name}' with "
                        f"{bank.size} frames."
                    )
                except Exception as exc:
                    st.error(f"Reference bank build failed: {exc}")
                finally:
                    progress_bar.empty()
                    for c in cleanups:
                        try:
                            c()
                        except Exception:
                            pass


# ---------------------------------------------------------------------
# Input widgets
# ---------------------------------------------------------------------
youtube_url: str = ""
uploaded = None

if input_mode == "YouTube URL":
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=…",
        help="We resolve a playable stream via yt-dlp. The full source video "
        "is NOT saved to disk.",
    )
else:
    uploaded = st.file_uploader(
        "Upload a candidate video",
        type=["mp4", "mov", "mkv", "webm", "avi", "m4v"],
        accept_multiple_files=False,
        key="candidate_upload",
    )
    st.caption(
        "Uploaded files are written to a temp location for OpenCV, then "
        "cleaned up after the run."
    )

run_clicked = st.button(
    "Run anomaly detection",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------
def _prepare_source() -> Optional[dict]:
    """Resolve the chosen input into a (source, type, label) dict."""
    if input_mode == "YouTube URL":
        url = (youtube_url or "").strip()
        if not url:
            st.warning("Please paste a YouTube URL first.")
            return None

        try:
            with st.spinner("Resolving YouTube stream via yt-dlp…"):
                resolved = resolve_youtube_stream(url)
        except YouTubeStreamError as exc:
            st.error(
                f"Could not resolve a readable stream: {exc}\n\n"
                "Please switch to **Local upload (fallback)** in the sidebar."
            )
            return None

        st.success(f"Resolved stream: **{resolved.title}**")
        meta_cols = st.columns(4)
        meta_cols[0].metric("Duration (s)", f"{resolved.duration_sec or 0:.0f}")
        meta_cols[1].metric("Format", resolved.ext or "—")
        meta_cols[2].metric("Width", resolved.width or "—")
        meta_cols[3].metric("Height", resolved.height or "—")
        return {
            "source": resolved.stream_url,
            "type": "youtube",
            "label": resolved.title or url,
            "cleanup": None,
        }

    if uploaded is None:
        st.warning("Please upload a video file first.")
        return None

    local = save_uploaded_file(uploaded, original_name=uploaded.name)
    return {
        "source": str(local.path),
        "type": "local_upload",
        "label": local.display_name,
        "cleanup": local.cleanup,
    }


def _run_pipeline(source_info: dict) -> None:
    if mode_id in {"reference_distance", "hybrid"} and selected_bank is None:
        st.error(
            "This scoring mode needs a reference bank. Build or load one "
            "first, or switch to the within-clip mode."
        )
        return

    cfg = PipelineConfig(
        interval_sec=float(frame_interval),
        top_k=int(top_k),
        contamination=float(contamination),
        max_frames=int(max_frames),
        smoothing_window=int(smoothing),
        segment_window=int(segment_window),
        segment_top_k=int(segment_top_k),
        reference_k=int(reference_k),
        hybrid_weight_within=float(hybrid_w_within),
        hybrid_weight_reference=float(hybrid_w_ref),
        backbone=backbone,
        mode=mode_id,  # type: ignore[arg-type]
        output_dir=OUTPUTS_DIR,
    )
    pipeline = get_pipeline(backbone)
    pipeline.config = cfg

    progress_bar = st.progress(0.0, text="Starting…")
    status = st.empty()
    start = time.time()

    def on_progress(p: float, msg: str) -> None:
        progress_bar.progress(min(max(p, 0.0), 1.0), text=msg)
        status.caption(msg)

    try:
        result = pipeline.run(
            video_source=source_info["source"],
            source_type=source_info["type"],
            source_label=source_info["label"],
            progress=on_progress,
            reference_bank=selected_bank,
        )
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        return
    finally:
        cleanup = source_info.get("cleanup")
        if callable(cleanup):
            try:
                cleanup()
            except Exception:
                pass

    elapsed = time.time() - start
    progress_bar.empty()
    status.empty()
    st.success(
        f"[{result.mode}] Analyzed {result.total_sampled_frames} frames in "
        f"{elapsed:.1f}s."
    )

    # --- Summary ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampled frames", result.total_sampled_frames)
    c2.metric("Top-K returned", len(result.top_records))
    c3.metric("Score min", f"{result.score_min:.3f}")
    c4.metric("Score max", f"{result.score_max:.3f}")

    st.markdown(f"**Output folder:** `{result.run_dir}`")
    if result.report_path:
        st.markdown(f"**Report:** `{result.report_path.name}`")

    # --- Score plot ---
    if result.plot_path and Path(result.plot_path).exists():
        st.image(str(result.plot_path), caption="Anomaly score over time")

    # --- Top anomalous frames ---
    st.subheader("Top anomalous frames")
    if not result.top_records:
        st.info("No top frames to display.")
    else:
        cols_per_row = 4
        for row_start in range(0, len(result.top_records), cols_per_row):
            row = result.top_records[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, rec in zip(cols, row):
                img_full = result.run_dir / rec.image_path
                if img_full.exists():
                    col.image(str(img_full), use_container_width=True)
                col.caption(
                    f"**Rank #{rec.rank}** · t = {rec.timestamp_sec:.2f}s\n\n"
                    f"score = {rec.anomaly_score:.3f} "
                    f"(norm {rec.normalized_score:.2f})"
                )

    # --- Top anomalous segments ---
    st.subheader("Top anomalous segments")
    if not result.top_segments:
        st.info("Segment analysis not available for this run.")
    else:
        seg_rows = [
            {
                "rank": s.rank,
                "start (s)": round(s.start_time_sec, 2),
                "end (s)": round(s.end_time_sec, 2),
                "mean_score": round(s.mean_score, 4),
                "max_score": round(s.max_score, 4),
                "representative_frame": s.representative_frame,
            }
            for s in result.top_segments
        ]
        st.dataframe(pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)

    # --- Downloads ---
    # Artifacts are persisted to the run folder on disk. Nothing is pushed
    # to the browser automatically; the user decides what (if anything) to
    # download by clicking a button.
    st.subheader("Download artifacts")
    st.caption(
        f"All files are saved under `{result.run_dir}` on disk. "
        "Use the buttons below to download individual artifacts."
    )

    download_targets: list[tuple[str, Path | None, str, str]] = [
        ("anomalies.csv (per-frame scores)", result.csv_path, "anomalies.csv", "text/csv"),
        ("segments.csv (top anomalous segments)", result.segment_csv_path, "segments.csv", "text/csv"),
        ("report.md (run summary)", result.report_path, "report.md", "text/markdown"),
        ("score_plot.png", result.plot_path, "score_plot.png", "image/png"),
    ]

    dl_cols = st.columns(2)
    for i, (label, path, fname, mime) in enumerate(download_targets):
        target_col = dl_cols[i % 2]
        if path and Path(path).exists():
            with open(path, "rb") as f:
                target_col.download_button(
                    label=f"Download {label}",
                    data=f.read(),
                    file_name=fname,
                    mime=mime,
                    use_container_width=True,
                    key=f"dl_{fname}",
                )
        else:
            target_col.caption(f"— {label} not available")


if run_clicked:
    info = _prepare_source()
    if info is not None:
        _run_pipeline(info)
else:
    st.info(
        "Pick a scoring mode, choose an input source, set your parameters, "
        "then click **Run anomaly detection**. Outputs are saved under "
        "`data/outputs/`."
    )


st.markdown("---")
st.caption(
    "Professionalized MVP. Unsupervised, frame- and segment-level anomaly "
    "detection. Not a production QA replacement."
)
