"""Streamlit "See output" tab: inline reports, CSVs, plots, optional benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def list_pipeline_runs(outputs_dir: Path) -> list[Path]:
    if not outputs_dir.exists():
        return []
    runs = [
        p
        for p in outputs_dir.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    ]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def render_see_output_tab(
    outputs_dir: Path,
    benchmark_dir: Path,
    *,
    preferred_run_name: str | None = None,
) -> None:
    st.subheader("See output")
    st.caption(
        "Everything is shown inline (no downloads): pipeline runs under `data/outputs/run_*`, "
        "and optional CLI benchmark under `data/outputs/game_benchmark/`."
    )

    runs = list_pipeline_runs(outputs_dir)
    names = [p.name for p in runs]
    idx = 0
    if preferred_run_name and preferred_run_name in names:
        idx = names.index(preferred_run_name)

    st.markdown("#### Pipeline run")
    if not names:
        st.info(
            "No `run_*` folders yet. Switch to the **Run** tab and click **Run anomaly detection**, "
            "or drop artifacts into `data/outputs/`."
        )
        chosen: Path | None = None
    else:
        pick = st.selectbox("Run folder", options=names, index=idx, key="gv_see_run_pick")
        chosen = outputs_dir / pick

    if chosen is not None:
        _show_pipeline_run_artifacts(chosen)

    st.markdown("---")
    st.markdown("#### Gameplay benchmark (CLI: `scripts/run_game_benchmark.py`)")
    _show_benchmark_artifacts(benchmark_dir)


def _show_benchmark_artifacts(benchmark_dir: Path) -> None:
    if not benchmark_dir.exists():
        st.info(
            f"No folder **`{benchmark_dir}`** yet. From the repo root run e.g. "
            f"`python scripts/run_game_benchmark.py` — results will appear here."
        )
        return

    bmd = benchmark_dir / "benchmark_table.md"
    csv_path = benchmark_dir / "benchmark_results.csv"
    json_path = benchmark_dir / "benchmark_results.json"
    profile_json = benchmark_dir / "profiling_report.json"
    profile_csv = benchmark_dir / "profiling_report.csv"
    cost_md = benchmark_dir / "cost_report.md"
    ablation_csv = benchmark_dir / "ablation_table.csv"

    if bmd.exists():
        st.markdown(bmd.read_text(encoding="utf-8"))
    elif not csv_path.exists():
        st.caption("No `benchmark_table.md` or `benchmark_results.csv` in this folder yet.")

    if csv_path.exists():
        st.markdown("**benchmark_results.csv**")
        st.dataframe(pd.read_csv(csv_path), use_container_width=True, hide_index=True)

    if ablation_csv.exists():
        st.markdown("**ablation_table.csv**")
        st.dataframe(pd.read_csv(ablation_csv), use_container_width=True, hide_index=True)

    if json_path.exists():
        st.markdown("**Evaluation metrics (benchmark_results.json)**")
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            st.json(data)
        except json.JSONDecodeError:
            st.code(json_path.read_text(encoding="utf-8"), language="json")

    if profile_json.exists():
        st.markdown("**Profiling (`profiling_report.json`)**")
        try:
            st.json(json.loads(profile_json.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            st.code(profile_json.read_text(encoding="utf-8"), language="json")
    if profile_csv.exists():
        st.markdown("**profiling_report.csv**")
        st.dataframe(pd.read_csv(profile_csv), use_container_width=True, hide_index=True)
    if cost_md.exists():
        st.markdown("**cost_report.md**")
        st.markdown(cost_md.read_text(encoding="utf-8"))

    chart_names = [
        ("metric_bar_chart.png", "Metric comparison"),
        ("roc_pr_curves.png", "ROC / PR curves"),
        ("latency_memory_chart.png", "Latency & memory"),
        ("confusion_matrices.png", "Confusion matrices"),
        ("sample_predictions_grid.png", "Sample predictions"),
    ]
    any_chart = any((benchmark_dir / n).exists() for n, _ in chart_names)
    if any_chart:
        st.markdown("**Benchmark figures**")
        for fname, title in chart_names:
            p = benchmark_dir / fname
            if p.exists():
                st.caption(title)
                st.image(str(p), use_container_width=True)


def _show_headline_metrics(run_dir: Path) -> None:
    """Surface JSON metrics for recruiters: latency, throughput, optional eval."""
    run_json = run_dir / "run_metrics.json"
    if run_json.exists():
        try:
            block = json.loads(run_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st.warning("Could not parse `run_metrics.json`.")
        else:
            st.markdown("**Operational metrics** *(from `run_metrics.json`)*")
            tt = block.get("timing_sec") or {}
            th = block.get("throughput") or {}
            sd = block.get("score_distribution") or {}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Wall time (s)", f"{float(tt.get('total', 0.0)):.2f}")
            c2.metric("Frames sampled", f"{int((block.get('scale') or {}).get('sampled_frames', 0))}")
            c3.metric("Throughput (frames/s)", f"{float(th.get('sampled_frames_per_sec_end_to_end', 0.0)):.1f}")
            c4.metric("Embedding throughput (emb/s)", f"{float(th.get('embeddings_per_sec', 0.0)):.1f}")
            sep = sd.get("top1_minus_median")
            if sep is not None:
                st.caption(f"Score separation (max − median): **{float(sep):.4f}** — larger often means clearer outliers.")
            with st.expander("Full `run_metrics.json`"):
                st.json(block)

    eval_json = run_dir / "eval_metrics.json"
    if eval_json.exists():
        try:
            ev = json.loads(eval_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st.warning("Could not parse `eval_metrics.json`.")
        else:
            st.markdown("**Supervised evaluation** *(needs labeled glitch frames; from `eval_metrics.json`)*")
            k_used = ev.get("k", "?")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Precision@{k_used}", f"{float(ev.get('precision_at_k', 0.0)):.3f}")
            c2.metric(f"Recall@{k_used}", f"{float(ev.get('recall_at_k', 0.0)):.3f}")
            c3.metric(f"Hit@{k_used}", f"{float(ev.get('hit_at_k', 0.0)):.3f}")
            pr = ev.get("pr_auc")
            roc = ev.get("roc_auc")
            c4.metric("PR-AUC", "—" if pr is None else f"{float(pr):.3f}")
            d1, d2, d3 = st.columns(3)
            d1.metric("ROC-AUC", "—" if roc is None else f"{float(roc):.3f}")
            d2.metric("F1 (thresholded)", f"{float(ev.get('f1', 0.0)):.3f}")
            d3.metric("Interval recall@top-k", f"{float(ev.get('interval_recall', 0.0)):.3f}")
            with st.expander("Full `eval_metrics.json`"):
                st.json(ev)


def _show_pipeline_run_artifacts(run_dir: Path) -> None:
    report = run_dir / "report.md"
    anomalies = run_dir / "anomalies.csv"
    segments = run_dir / "segments.csv"
    plot_path = run_dir / "score_plot.png"

    _show_headline_metrics(run_dir)
    if report.exists():
        st.markdown("**Report**")
        st.markdown(report.read_text(encoding="utf-8"))

    if anomalies.exists():
        st.markdown("**anomalies.csv**")
        df = pd.read_csv(anomalies)
        st.dataframe(df, use_container_width=True, hide_index=True)
        _metrics_from_anomalies(df)
        _show_top_frame_thumbnails(run_dir, df)
    else:
        st.warning("No `anomalies.csv` in this run folder.")

    if segments.exists():
        st.markdown("**segments.csv**")
        st.dataframe(pd.read_csv(segments), use_container_width=True, hide_index=True)

    if plot_path.exists():
        st.markdown("**score_plot.png**")
        st.image(str(plot_path), use_container_width=True)


def _show_top_frame_thumbnails(run_dir: Path, df: pd.DataFrame, limit: int = 12) -> None:
    if "image_path" not in df.columns or "rank" not in df.columns:
        return
    r_col = pd.to_numeric(df["rank"], errors="coerce")
    ranked = df.loc[r_col >= 1].copy()
    ranked = ranked[ranked["image_path"].astype(str).str.strip() != ""]
    if ranked.empty:
        return
    sub = ranked.sort_values("rank", ascending=True).head(int(limit))
    paths = [run_dir / str(p) for p in sub["image_path"].tolist() if str(p).strip()]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        return
    st.markdown(f"**Top {len(paths)} anomaly frames (thumbnails)**")
    cols_per_row = 4
    for i in range(0, len(paths), cols_per_row):
        row_paths = paths[i : i + cols_per_row]
        cols = st.columns(len(row_paths))
        for col, path in zip(cols, row_paths):
            col.image(str(path), use_container_width=True)


def _metrics_from_anomalies(df: pd.DataFrame) -> None:
    st.markdown("**Summary metrics (from this run)**")
    if "anomaly_score" in df.columns:
        s = pd.to_numeric(df["anomaly_score"], errors="coerce").dropna()
        if len(s):
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean anomaly score", f"{float(s.mean()):.4f}")
            c2.metric("Std anomaly score", f"{float(s.std()):.4f}")
            c3.metric("Max anomaly score", f"{float(s.max()):.4f}")
    if "normalized_score" in df.columns:
        n = pd.to_numeric(df["normalized_score"], errors="coerce").dropna()
        if len(n):
            st.metric("Mean normalized score", f"{float(n.mean()):.4f}")

    st.markdown("**Within-clip vs reference channels** (hybrid / reference modes fill these)")
    rows = []
    if "within_score" in df.columns:
        w = pd.to_numeric(df["within_score"], errors="coerce").dropna()
        if len(w):
            rows.append(
                {
                    "channel": "within-clip (Isolation Forest)",
                    "mean": float(w.mean()),
                    "std": float(w.std()),
                    "max": float(w.max()),
                }
            )
    if "reference_score" in df.columns:
        r = pd.to_numeric(df["reference_score"], errors="coerce").dropna()
        if len(r):
            rows.append(
                {
                    "channel": "reference (kNN distance)",
                    "mean": float(r.mean()),
                    "std": float(r.std()),
                    "max": float(r.max()),
                }
            )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption(
            "Only the combined `anomaly_score` is present for this mode (within-only or reference-only)."
        )
