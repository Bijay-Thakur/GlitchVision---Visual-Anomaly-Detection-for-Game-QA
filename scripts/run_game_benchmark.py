"""Run the gameplay-reference benchmark end to end.

Run from repo root::

    python scripts/run_game_benchmark.py

If you see ``No module named 'scripts'``, you were likely not in the project
root, or a stale interpreter cwd — this file adds the repo root to
``sys.path`` automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.build_gameplay_reference_bank import build_gameplay_reference_bank
from src.features import EmbeddingExtractor, compute_temporal_features
from src.game_benchmark.evaluate import compute_metrics, save_metrics
from src.game_benchmark.glitch_injection import inject_gameplay_glitches
from src.game_benchmark.models import (
    classifier_anomaly_scores,
    file_size_mb,
    score_embedding_iforest,
    score_handcrafted_iforest,
    score_reference_knn,
    supervised_feature_matrix,
    train_lightweight_classifier,
)
from src.ingestion.url_reference_loader import DEFAULT_OUT_DIR, verify_reference_urls
from src.processing import FrameExtractor
from src.utils.profiling import StageProfiler, write_profile_reports


def _load_reference_bank(path: Path) -> np.ndarray:
    bank_path = path / "reference_bank.npz"
    if not bank_path.exists():
        return np.zeros((0, 512), dtype=np.float32)
    with np.load(bank_path) as data:
        return np.asarray(data["embeddings"], dtype=np.float32)


def _sample_verified_frames(args, verified_results) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    timestamps = []
    groups = []
    for video_idx, item in enumerate([r for r in verified_results if r.ok][: args.max_videos]):
        try:
            with FrameExtractor(
                item.stream_url,
                interval_sec=args.eval_interval_sec,
                image_size=args.image_size,
                max_frames=args.max_samples_per_video,
            ) as extractor:
                for sample in extractor.iter_frames():
                    frames.append(sample.image)
                    timestamps.append(sample.timestamp_sec)
                    groups.append(f"video_{video_idx}")
        except Exception:
            continue
    if frames:
        return np.asarray(frames, dtype=np.uint8), np.asarray(timestamps, dtype=np.float32), np.asarray(groups, dtype=object)
    return _synthetic_gameplay_frames(args.max_samples_per_video, args.image_size, args.seed)


def _synthetic_gameplay_frames(n: int, image_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = max(24, int(n))
    frames = []
    for i in range(n):
        base = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        x = np.linspace(0, 1, image_size, dtype=np.float32)
        y = np.linspace(0, 1, image_size, dtype=np.float32)[:, None]
        base[..., 0] = np.clip(45 + 55 * x + (i % 20), 0, 255).astype(np.uint8)
        base[..., 1] = np.clip(80 + 45 * y + rng.normal(0, 2, (image_size, 1)), 0, 255).astype(np.uint8)
        base[..., 2] = np.clip(120 + 25 * np.sin((x * 6) + i / 8), 0, 255).astype(np.uint8)
        base[image_size - 28 : image_size - 10, 8 : image_size // 2] = (35, 35, 35)
        cx, cy = (i * 5) % image_size, image_size // 2
        yy, xx = np.ogrid[:image_size, :image_size]
        base[(xx - cx) ** 2 + (yy - cy) ** 2 <= 100] = (220, 190, 80)
        frames.append(base)
    timestamps = np.arange(n, dtype=np.float32) * 1.5
    groups = np.array([f"synthetic_{i // max(1, n // 3)}" for i in range(n)], dtype=object)
    return np.asarray(frames, dtype=np.uint8), timestamps, groups


def _embed_frames(frames: np.ndarray, batch_size: int = 8) -> np.ndarray:
    embedder = EmbeddingExtractor(device="cpu", backbone="resnet18")
    parts = []
    for start in range(0, len(frames), batch_size):
        parts.append(embedder.embed(frames[start : start + batch_size]))
    return np.concatenate(parts, axis=0).astype(np.float32)


def _threshold_intervals(scores: np.ndarray, quantile: float = 0.90) -> list[dict]:
    if len(scores) == 0:
        return []
    mask = scores >= float(np.quantile(scores, quantile))
    intervals = []
    start = None
    for idx, flag in enumerate(mask.tolist() + [False]):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            intervals.append({"start_frame": start, "end_frame": idx})
            start = None
    return intervals


def _write_table(rows: list[dict], out_path: Path) -> None:
    headers = ["model", "f1", "pr_auc", "precision_at_k", "hit_at_k", "latency_sec", "peak_memory_mb", "artifact_size_mb"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plots(rows: list[dict], metrics_by_model: Dict[str, dict], y_true: np.ndarray, scores_by_model: Dict[str, np.ndarray], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    names = [r["model"] for r in rows]
    f1 = [float(r.get("f1") or 0.0) for r in rows]
    pr = [float(r.get("pr_auc") or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=120)
    x = np.arange(len(names))
    ax.bar(x - 0.18, f1, 0.36, label="F1")
    ax.bar(x + 0.18, pr, 0.36, label="PR-AUC")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "metric_bar_chart.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    if len(np.unique(y_true)) > 1:
        for name, scores in scores_by_model.items():
            RocCurveDisplay.from_predictions(y_true, scores, name=name, ax=axes[0])
            PrecisionRecallDisplay.from_predictions(y_true, scores, name=name, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    xpos = np.arange(len(names))
    ax.bar(xpos, [float(r.get("latency_sec") or 0.0) for r in rows], label="latency sec")
    ax.set_xticks(xpos, labels=names, rotation=20, ha="right")
    ax.set_ylabel("seconds")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_memory_chart.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), dpi=120)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        cm = np.asarray(metrics_by_model[name]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, int(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png")
    plt.close(fig)


def _sample_predictions_grid(frames: np.ndarray, y: np.ndarray, scores: np.ndarray, out_path: Path, k: int = 12) -> None:
    import cv2

    idx = np.argsort(-scores)[: min(k, len(scores))]
    if len(idx) == 0:
        return
    thumbs = frames[idx]
    h, w = thumbs[0].shape[:2]
    cols = min(4, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for j, frame in enumerate(thumbs):
        r, c = divmod(j, cols)
        tile = frame.copy()
        color = (0, 255, 0) if y[idx[j]] else (0, 0, 255)
        cv2.rectangle(tile, (0, 0), (w - 1, h - 1), color, 3)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    try:
        cv2.imwrite(str(out_path), grid)
    except Exception:
        from PIL import Image

        Image.fromarray(grid[..., ::-1]).save(out_path)


def run_benchmark(args) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiler = StageProfiler()
    profiler.start()
    verified_results = []

    with profiler.stage("url_verification"):
        if args.skip_url_verification:
            verified_results = []
        else:
            verified_results = verify_reference_urls(args.urls_file, DEFAULT_OUT_DIR, args.max_videos)

    bank_dir = Path("data/reference_banks/gameplay_reference")
    with profiler.stage("frame_sampling"):
        if not args.reuse_reference_bank:
            build_gameplay_reference_bank(
                urls_file=args.urls_file,
                out_dir=bank_dir,
                interval_sec=args.interval_sec,
                max_samples_per_video=args.max_samples_per_video,
                max_videos=args.max_videos,
                image_size=args.image_size,
            )
        clean_frames, timestamps, groups = _sample_verified_frames(args, verified_results)

    with profiler.stage("feature_extraction"):
        dataset = inject_gameplay_glitches(clean_frames, timestamps, groups, seed=args.seed, debug_dir=out_dir / "debug_glitches")
        embeddings = _embed_frames(dataset.frames)
        reference_embeddings = _load_reference_bank(bank_dir)
        if len(reference_embeddings) == 0:
            reference_embeddings = embeddings[dataset.labels == 0][: max(1, min(30, len(embeddings)))]

    intervals = dataset.interval_dicts()
    k = max(1, min(20, int(dataset.labels.sum()) or 10))
    scores_by_model: Dict[str, np.ndarray] = {}
    latency: Dict[str, float] = {}

    with profiler.stage("scoring"):
        for name, fn in [
            ("Handcrafted + IsolationForest", lambda: score_handcrafted_iforest(dataset.frames, args.seed)),
            ("ResNet-18 + IsolationForest", lambda: score_embedding_iforest(embeddings, args.seed)),
            ("ResNet-18 + reference kNN", lambda: score_reference_knn(embeddings, reference_embeddings, k=5)),
        ]:
            start = time.perf_counter()
            scores_by_model[name] = fn()
            latency[name] = time.perf_counter() - start
    before = {"baseline_scoring_sec": sum(latency.values())}

    with profiler.stage("training"):
        trained = train_lightweight_classifier(
            embeddings,
            dataset.labels,
            dataset.groups,
            timestamps=dataset.timestamps,
            out_path="data/models/game_anomaly_classifier.joblib",
            seed=args.seed,
        )
    with profiler.stage("scoring"):
        start = time.perf_counter()
        features, _names = supervised_feature_matrix(embeddings, dataset.timestamps, include_temporal=True)
        scores_by_model["ResNet-18 + temporal + trained LogisticRegression"] = classifier_anomaly_scores(trained["model"], features)
        latency["ResNet-18 + temporal + trained LogisticRegression"] = time.perf_counter() - start
    after = {"trained_classifier_scoring_sec": latency["ResNet-18 + temporal + trained LogisticRegression"]}

    with profiler.stage("evaluation"):
        metrics_by_model = {}
        rows = []
        for name, scores in scores_by_model.items():
            pred_intervals = _threshold_intervals(scores)
            metrics = compute_metrics(dataset.labels, scores, k=k, intervals=intervals, predicted_intervals=pred_intervals)
            metrics["latency_sec"] = latency.get(name, 0.0)
            metrics["artifact_size_mb"] = file_size_mb(trained.get("model_path")) if "trained" in name.lower() else 0.0
            metrics_by_model[name] = metrics
            rows.append(
                {
                    "model": name,
                    "features_used": "see model name",
                    "scorer": name.split("+")[-1].strip(),
                    "f1": metrics["f1"],
                    "pr_auc": metrics["pr_auc"],
                    "precision_at_k": metrics["precision_at_k"],
                    "hit_at_k": metrics["hit_at_k"],
                    "latency_sec": metrics["latency_sec"],
                    "peak_memory_mb": 0.0,
                    "artifact_size_mb": metrics["artifact_size_mb"],
                }
            )
        save_metrics(metrics_by_model, out_dir)
        pd.DataFrame(rows).to_csv(out_dir / "ablation_table.csv", index=False)
        _write_table(rows, out_dir / "benchmark_table.md")
        _write_table(rows, out_dir / "ablation_table.md")

    with profiler.stage("report_generation"):
        _plots(rows, metrics_by_model, dataset.labels, scores_by_model, out_dir)
        _sample_predictions_grid(dataset.frames, dataset.labels, scores_by_model[rows[-1]["model"]], out_dir / "sample_predictions_grid.png")

    profile = profiler.finish(
        samples=len(dataset.labels),
        model_artifact=trained.get("model_path"),
        reference_bank=bank_dir,
        videos_probed=len(verified_results),
        frames_sampled=len(clean_frames),
    )
    for row in rows:
        row["peak_memory_mb"] = profile.get("peak_memory_mb", 0.0)
    pd.DataFrame(rows).to_csv(out_dir / "benchmark_results.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "ablation_table.csv", index=False)
    _write_table(rows, out_dir / "benchmark_table.md")
    _write_table(rows, out_dir / "ablation_table.md")
    write_profile_reports(profile, out_dir, before_training=before, after_training=after)
    (out_dir / "benchmark_results.json").write_text(json.dumps(metrics_by_model, indent=2), encoding="utf-8")
    return {"out_dir": str(out_dir), "metrics": metrics_by_model, "profile": profile}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GlitchVision gameplay benchmark.")
    parser.add_argument("--urls-file", default="data/reference_videos.txt")
    parser.add_argument("--max-videos", type=int, default=10)
    parser.add_argument("--max-samples-per-video", type=int, default=80)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--eval-interval-sec", type=float, default=1.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/outputs/game_benchmark")
    parser.add_argument("--skip-url-verification", action="store_true")
    parser.add_argument("--reuse-reference-bank", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run_benchmark(parse_args()), indent=2))
