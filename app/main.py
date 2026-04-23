"""GlitchVision — Streamlit demo UI.

Two input modes:
    1. YouTube URL (primary)   — resolve a stream via yt-dlp, no persistent
       download of the source video.
    2. Local file upload (fallback) — for reliability when a YouTube stream
       isn't OpenCV-compatible, or when offline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Make ``src`` importable when running via ``streamlit run app/main.py``.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import DEFAULTS, OUTPUTS_DIR  # noqa: E402
from src.ingestion import (  # noqa: E402
    YouTubeStreamError,
    resolve_youtube_stream,
    save_uploaded_file,
)
from src.pipeline import GlitchVisionPipeline, PipelineConfig  # noqa: E402


st.set_page_config(
    page_title="GlitchVision — Visual Anomaly Detection for Game QA",
    page_icon="🎮",
    layout="wide",
)


# ---------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ResNet-18 backbone…")
def get_pipeline() -> GlitchVisionPipeline:
    """A single pipeline instance per Streamlit session keeps the model warm."""
    return GlitchVisionPipeline()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("GlitchVision")
st.caption(
    "Unsupervised visual anomaly detection for gameplay footage. "
    "ResNet-18 embeddings + Isolation Forest, CPU-only."
)

with st.expander("What is this?", expanded=False):
    st.markdown(
        """
        **GlitchVision** samples frames from a video, embeds each frame with a
        pretrained **ResNet-18**, and fits an **Isolation Forest** over the
        embeddings. Frames that look statistically unusual compared to the
        rest of the clip are surfaced as candidate anomalies — the kind of
        frames a QA engineer might want to eyeball.

        This is an MVP. It does **frame-level, unsupervised** detection. It
        doesn't know what a "glitch" is; it flags *outliers* in feature space.
        """
    )


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
st.sidebar.header("Run controls")

input_mode = st.sidebar.radio(
    "Input source",
    options=["YouTube URL", "Local upload (fallback)"],
    index=0,
    help="YouTube is the primary path. Local upload is a reliable fallback.",
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
    "Contamination (expected anomaly fraction)",
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
    help="Hard cap for responsive CPU runs.",
)

smoothing = st.sidebar.slider(
    "Temporal smoothing window",
    min_value=1,
    max_value=9,
    value=int(DEFAULTS.smoothing_window),
    step=2,
    help="1 disables smoothing. Larger windows = calmer score curve.",
)


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
        "Upload a video file",
        type=["mp4", "mov", "mkv", "webm", "avi", "m4v"],
        accept_multiple_files=False,
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
                "Some YouTube formats aren't OpenCV-compatible on this "
                "machine. Please switch to **Local upload (fallback)** in "
                "the sidebar."
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
    cfg = PipelineConfig(
        interval_sec=float(frame_interval),
        top_k=int(top_k),
        contamination=float(contamination),
        max_frames=int(max_frames),
        smoothing_window=int(smoothing),
        output_dir=OUTPUTS_DIR,
    )
    pipeline = get_pipeline()
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
    st.success(f"Analyzed {result.total_sampled_frames} frames in {elapsed:.1f}s.")

    # --- Summary ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampled frames", result.total_sampled_frames)
    c2.metric("Top-K returned", len(result.top_records))
    c3.metric("Score min", f"{result.score_min:.3f}")
    c4.metric("Score max", f"{result.score_max:.3f}")

    st.markdown(f"**Output folder:** `{result.run_dir}`")

    # --- Score plot ---
    if result.plot_path and Path(result.plot_path).exists():
        st.image(str(result.plot_path), caption="Anomaly score over time")

    # --- Top anomaly gallery ---
    st.subheader("Top anomalies")
    if not result.top_records:
        st.info("No top anomalies to display.")
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

    # --- CSV download ---
    st.subheader("Full results")
    try:
        df = pd.read_csv(result.csv_path)
        st.dataframe(df, use_container_width=True, hide_index=True)
        with open(result.csv_path, "rb") as f:
            st.download_button(
                "Download anomalies.csv",
                data=f.read(),
                file_name="anomalies.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception as exc:
        st.warning(f"CSV preview unavailable: {exc}")


if run_clicked:
    info = _prepare_source()
    if info is not None:
        _run_pipeline(info)
else:
    st.info(
        "Pick an input source in the sidebar, set your parameters, then click "
        "**Run anomaly detection**. Outputs are saved under `data/outputs/`."
    )


st.markdown("---")
st.caption(
    "MVP built for a CodePath showcase. Unsupervised, frame-level anomaly "
    "detection. Not a research-grade system."
)
