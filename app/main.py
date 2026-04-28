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

from app.config import (  # noqa: E402
    DEFAULTS,
    GAME_BENCHMARK_DIR,
    LEGACY_REFERENCE_VIDEOS_FILE,
    OUTPUTS_DIR,
    REFERENCE_BANKS_DIR,
    REFERENCE_VIDEOS_FILE,
    REPO_ROOT,
)
from src.ingestion import (  # noqa: E402
    YouTubeStreamError,
    opera_gx_profile_dir,
    resolve_youtube_stream,
    save_uploaded_file,
)
from src.pipeline import GlitchVisionPipeline, PipelineConfig  # noqa: E402
from src.reference import ReferenceBank, youtube_sources  # noqa: E402

from app import output_view  # noqa: E402


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
st.caption(
    "**See output** (first tab below) shows every saved artifact inline: report, tables, "
    "plots, thumbnails, and metrics. Use **Run** to process a video."
)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.header("Run controls")
st.sidebar.caption("After a run, open the **See output** tab for full results.")

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

with st.sidebar.expander("YouTube cookies (anti-bot)", expanded=False):
    st.caption(
        "Use this when YouTube returns **Sign in to confirm you're not a bot**. "
        "A real **Netscape `cookies.txt` file** is the most reliable on Windows."
    )
    if sys.platform == "win32":
        st.warning(
            "On **Windows**, **Chrome**, **Edge**, **Brave**, and **Opera GX** use "
            "Chromium-style cookie encryption and may hit **Failed to decrypt with DPAPI** "
            "when yt-dlp reads them. **Most reliable:** export **`cookies.txt`** (link below). "
            "**Often works without DPAPI issues:** **Firefox**. **Opera GX:** pick "
            "**opera gx** below (not plain **opera**) so we point at the GX profile folder."
        )
    st.markdown(
        "[Export YouTube cookies for yt-dlp](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies)"
    )
    youtube_cookies_path = st.text_input(
        "Path to cookies.txt (recommended on Windows)",
        value="",
        placeholder=r"e.g. C:\Users\YourName\Downloads\youtube-cookies.txt",
        key="gv_yt_cookie_path",
        help="Must be a real path to an exported Netscape cookies file. "
        "If this file is missing, the path is ignored and yt-dlp uses the browser option below.",
    )
    _p_raw = (youtube_cookies_path or "").strip()
    if _p_raw and not Path(_p_raw).is_file():
        st.error(
            f"**File not found:** `{_p_raw}` — fix the path, or clear the field. "
            "Until the file exists, only the **browser** option below is used."
        )
    yt_browser_pick = st.selectbox(
        "Or read cookies from browser",
        [
            "— none —",
            "firefox",
            "opera gx",
            "opera (stable)",
            "edge",
            "chrome",
            "brave",
        ],
        index=0,
        key="gv_yt_browser",
        help="**Edge** uses your Gmail login but can still fail with DPAPI on Windows. "
        "For **Opera GX**, choose **opera gx**. "
        "**Important:** quit the browser completely before resolve — yt-dlp copies the "
        "cookie DB; if the browser is open you may get *Could not copy … cookie database*. "
        "Check Task Manager for stray processes, or use **cookies.txt** only (browser = — none —).",
    )
    if yt_browser_pick == "opera gx":
        _gx = opera_gx_profile_dir()
        if _gx is None:
            st.error(
                "**Opera GX profile not found.** Install/sign in to Opera GX first, or set "
                "environment variable **GLITCHVISION_OPERA_GX_USER_DATA** to your GX "
                "user-data folder (usually contains `Local State`)."
            )
        else:
            st.caption(
                f"GX folder: `{_gx}`. **Quit Opera GX fully** (including systray/background). "
                "If you still see *Could not copy cookie database*, open **Task Manager** → "
                "end all **Opera** tasks, then retry — or use **`cookies.txt`** with browser "
                "set to **— none —**."
            )
    elif yt_browser_pick == "edge" and sys.platform == "win32":
        st.caption(
            "**Edge** often triggers the same **DPAPI** error as Chrome. If it does, use "
            "**cookies.txt** or **Firefox** for YouTube."
        )
_browser_internal = {
    "— none —": None,
    "firefox": "firefox",
    "opera gx": "opera_gx",
    "opera (stable)": "opera",
    "edge": "edge",
    "chrome": "chrome",
    "brave": "brave",
}
youtube_cookies_browser: str | None = _browser_internal.get(yt_browser_pick or "")

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

video_portion_pct = st.sidebar.radio(
    "Video portion to analyze",
    options=[25, 50, 100],
    index=2,
    horizontal=True,
    help="Uses the first N%% of the clip when OpenCV knows duration (typical for local files). "
    "Many streams report unknown length — the full stream is used.",
)
duration_fraction = float(video_portion_pct) / 100.0

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
tab_see, tab_run = st.tabs(["See output", "Run"])

with tab_see:
    output_view.render_see_output_tab(
        OUTPUTS_DIR,
        GAME_BENCHMARK_DIR,
        preferred_run_name=st.session_state.get("gv_last_run_name"),
    )

with tab_run:
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

    # Reference bank management
    # ---------------------------------------------------------------------
    REFERENCE_BANKS_DIR.mkdir(parents=True, exist_ok=True)


    def _effective_reference_urls_file() -> Path:
        """Prefer ``data/reference_videos.txt``, then legacy ``reference_banks/…``."""
        if REFERENCE_VIDEOS_FILE.exists():
            return REFERENCE_VIDEOS_FILE
        if LEGACY_REFERENCE_VIDEOS_FILE.exists():
            return LEGACY_REFERENCE_VIDEOS_FILE
        return REFERENCE_VIDEOS_FILE


    def _parse_non_comment_urls(path: Path) -> list[str]:
        if not path.exists():
            return []
        urls: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
        return urls


    def _list_reference_banks() -> list[Path]:
        if not REFERENCE_BANKS_DIR.exists():
            return []
        valid = []
        for p in REFERENCE_BANKS_DIR.iterdir():
            if not p.is_dir():
                continue
            if (p / "embeddings.npz").exists() and (p / "metadata.json").exists():
                valid.append(p)
        return sorted(valid, key=lambda x: x.stat().st_mtime, reverse=True)


    selected_bank: Optional[ReferenceBank] = None
    selected_bank_path: Optional[Path] = None

    if mode_id in {"reference_distance", "hybrid"}:
        st.subheader("Reference bank")

        if "ref_bank_built_notice" in st.session_state:
            st.success(st.session_state.pop("ref_bank_built_notice"))

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
                "No reference banks are on disk yet. "
                "Expand **Build a new reference bank** and use **From YouTube URL list** "
                "(not a local upload unless you have files)."
            )

        ref_url_path = _effective_reference_urls_file()
        url_list = _parse_non_comment_urls(ref_url_path)
        try:
            url_display = ref_url_path.relative_to(REPO_ROOT)
        except ValueError:
            url_display = ref_url_path

        with st.expander("Build a new reference bank", expanded=(not bank_names)):
            bank_name = st.text_input(
                "Name for the new bank",
                value="known_good_v1",
                help="Stored under `data/reference_banks/<name>/`.",
            )
            tab_urls, tab_upload = st.tabs(
                ["From YouTube URL list (recommended)", "From file upload"]
            )

            with tab_urls:
                if not ref_url_path.exists():
                    st.caption(
                        f"Create `{url_display}` in the repo (one YouTube URL per line). "
                        "Lines starting with `#` are ignored."
                    )
                else:
                    st.caption(
                        f"Reads **`{url_display}`** — **{len(url_list)}** URL(s). "
                        "Resolves each with yt-dlp and samples frames like uploads."
                    )
                build_from_urls_clicked = st.button(
                    "Build bank from URL list file",
                    type="primary",
                    disabled=not url_list,
                    key="build_from_urls",
                    help="Uses every line in the file; failed URLs are skipped with a warning.",
                )

                if build_from_urls_clicked:
                    if not bank_name.strip():
                        st.warning("Please name the bank.")
                    else:
                        pipeline_for_ref = get_pipeline(backbone)
                        pipeline_for_ref.config = PipelineConfig(
                            interval_sec=float(frame_interval),
                            image_size=int(DEFAULTS.image_size),
                            max_frames=int(max_frames),
                            backbone=backbone,
                            duration_fraction=float(duration_fraction),
                        )
                        progress_bar = st.progress(0.0, text="Resolving reference URLs…")

                        def _on_resolve(cur: int, tot: int) -> None:
                            progress_bar.progress(
                                min(cur / max(1, tot), 0.99),
                                text=f"Resolving URL {cur}/{tot}…",
                            )

                        try:
                            ca = _youtube_cookie_args()
                            sources, resolve_errors = youtube_sources.page_urls_to_stream_sources(
                                url_list,
                                on_progress=_on_resolve,
                                cookies_file=ca.get("cookies_file"),
                                cookies_from_browser=ca.get("cookies_from_browser"),
                            )
                            if resolve_errors:
                                st.warning(
                                    "Some URLs could not be resolved (others are still used):\n\n"
                                    + "\n".join(resolve_errors)
                                )
                            if not sources:
                                st.error(
                                    "No URLs could be opened. Fix errors above or check the list file."
                                )
                            else:
                                progress_bar.progress(0.0, text="Building embedding bank…")
                                try:
                                    bank = pipeline_for_ref.build_reference(
                                        sources,
                                        out_dir=REFERENCE_BANKS_DIR / bank_name.strip(),
                                        progress=lambda p, m: progress_bar.progress(
                                            min(max(p, 0.0), 1.0), text=m
                                        ),
                                        notes=(
                                            f"from {url_display}; "
                                            f"{len(sources)}/{len(url_list)} URLs OK"
                                        ),
                                    )
                                    st.session_state["ref_bank_built_notice"] = (
                                        f"Saved **{bank_name.strip()}** — {bank.size} frames "
                                        f"from {len(sources)} video(s). Select it above."
                                    )
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Reference bank build failed: {exc}")
                        finally:
                            progress_bar.empty()

            with tab_upload:
                ref_upload = st.file_uploader(
                    "Upload one or more known-good reference videos",
                    type=["mp4", "mov", "mkv", "webm", "avi", "m4v"],
                    accept_multiple_files=True,
                    key="ref_upload",
                )
                build_clicked = st.button(
                    "Build from uploaded files",
                    type="secondary",
                    key="build_from_upload",
                )

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
                            duration_fraction=float(duration_fraction),
                        )
                        cleanups = []
                        sources: list[tuple[str, str]] = []
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
                            st.session_state["ref_bank_built_notice"] = (
                                f"Saved **{bank_name.strip()}** — {bank.size} frames. "
                                "Select it above."
                            )
                            st.rerun()
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
    def _youtube_cookie_args() -> dict:
        """Arguments for :func:`~src.ingestion.youtube_stream.resolve_youtube_stream`."""
        raw = (youtube_cookies_path or "").strip()
        cf: Path | None = Path(raw) if raw else None
        if cf is not None and not cf.is_file():
            cf = None
        out: dict = {}
        if cf is not None:
            out["cookies_file"] = cf
        if youtube_cookies_browser:
            out["cookies_from_browser"] = youtube_cookies_browser
        return out


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
                    resolved = resolve_youtube_stream(url, **_youtube_cookie_args())
            except YouTubeStreamError as exc:
                err_low = str(exc).lower()
                cookie_hint = ""
                if "dpapi" in err_low or ("decrypt" in err_low and "cookie" in err_low):
                    cookie_hint = (
                        "\n\n**DPAPI (Windows + Chrome/Edge):** yt-dlp often cannot read "
                        "those browser cookies. Set **Or read cookies from browser** to "
                        "**— none —**, export **`cookies.txt`** while logged into YouTube "
                        "(sidebar link), and paste the **full path** to that file. Or use "
                        "**Firefox** for YouTube and pick **firefox** here. "
                        "[Details](https://github.com/yt-dlp/yt-dlp/issues/10927)"
                    )
                elif "could not copy" in err_low and "cookie" in err_low:
                    cookie_hint = (
                        "\n\n**Cookie database is locked:** yt-dlp must **copy** your "
                        "browser’s cookie file. **Fully quit** the browser (Opera GX, Chrome, "
                        "Edge, etc.) — check **Task Manager** for leftover **Opera** / "
                        "**opera** processes and **End task**. Then run again. "
                        "Or avoid the lock entirely: export **`cookies.txt`**, set "
                        "**— none —** for browser, paste only the file path. "
                        "[yt-dlp #7271](https://github.com/yt-dlp/yt-dlp/issues/7271)"
                    )
                elif any(
                    x in err_low
                    for x in ("sign in", "not a bot", "bot", "cookie", "authentication")
                ):
                    cookie_hint = (
                        "\n\n**If YouTube blocked the request:** open **YouTube cookies (anti-bot)** "
                        "in the sidebar and add a valid `cookies.txt` path or choose your browser, "
                        "then run again. Or switch to **Local upload (fallback)**."
                    )
                st.error(
                    f"Could not resolve a readable stream: {exc}{cookie_hint}"
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
            duration_fraction=float(duration_fraction),
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
        st.session_state["gv_last_run_name"] = result.run_dir.name
        st.success(
            f"[{result.mode}] Analyzed {result.total_sampled_frames} frames in "
            f"{elapsed:.1f}s → `{result.run_dir.name}`"
        )
        st.info(
            "Switch to the **See output** tab (first tab) for **report**, **anomalies.csv**, "
            "**segments**, **score plot**, thumbnails, and metrics — plus CLI benchmark output if present."
        )


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
