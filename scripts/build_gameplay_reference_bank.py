"""Build a gameplay reference bank from verified stream URLs.

Only derived artifacts are stored: embeddings, metadata, and an optional
thumbnail grid. Source videos are streamed/sampled and never downloaded.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

import numpy as np

from src.features import EmbeddingExtractor
from src.ingestion.url_reference_loader import DEFAULT_OUT_DIR, DEFAULT_URLS_FILE, verify_reference_urls
from src.processing import FrameExtractor


def _is_near_black(frame: np.ndarray, threshold: float = 8.0) -> bool:
    return float(frame.mean()) < threshold


def _pixel_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def _looks_like_static_menu(frame: np.ndarray) -> bool:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge_density = float((cv2.Canny(gray, 80, 160) > 0).mean())
    # Conservative heuristic: large, almost flat image with very sparse edges.
    return float(gray.std()) < 7.5 and edge_density < 0.01


def _embed_batches(embedder: EmbeddingExtractor, frames: Sequence[np.ndarray], batch_size: int = 8) -> np.ndarray:
    chunks = []
    for start in range(0, len(frames), batch_size):
        chunks.append(embedder.embed(np.asarray(frames[start : start + batch_size], dtype=np.uint8)))
    return np.concatenate(chunks, axis=0).astype(np.float32) if chunks else np.zeros((0, embedder.embedding_dim), dtype=np.float32)


def _thumbnail_grid(frames: Sequence[np.ndarray], out_path: Path, max_images: int = 25) -> None:
    if not frames:
        return
    take = list(frames[:max_images])
    h, w = take[0].shape[:2]
    cols = min(5, len(take))
    rows = int(np.ceil(len(take) / cols))
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, frame in enumerate(take):
        r, c = divmod(idx, cols)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        cv2.imwrite(str(out_path), grid)
    except Exception:
        from PIL import Image

        Image.fromarray(grid[..., ::-1]).save(out_path)


def build_gameplay_reference_bank(
    urls_file: str | Path = DEFAULT_URLS_FILE,
    out_dir: str | Path = "data/reference_banks/gameplay_reference",
    interval_sec: float = 5.0,
    max_samples_per_video: int = 120,
    max_videos: int = 20,
    image_size: int = 224,
    verified_urls_file: str | Path | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if verified_urls_file is None:
        results = verify_reference_urls(urls_file, DEFAULT_OUT_DIR, max_urls=max_videos)
        verified = [r for r in results if r.ok]
    else:
        verified = []
        for url in Path(verified_urls_file).read_text(encoding="utf-8").splitlines():
            url = url.strip()
            if url:
                # Re-verification is safer because stream URLs expire.
                verified.extend([r for r in verify_reference_urls(urls_file, DEFAULT_OUT_DIR, max_urls=max_videos) if r.url == url and r.ok])
    verified = verified[: max(0, int(max_videos))]

    embedder = EmbeddingExtractor(device="cpu", backbone="resnet18")
    all_frames: List[np.ndarray] = []
    metadata_frames: list[dict] = []
    embeddings_parts: list[np.ndarray] = []

    for video_idx, item in enumerate(verified):
        kept: List[np.ndarray] = []
        last_frame: np.ndarray | None = None
        try:
            with FrameExtractor(
                item.stream_url,
                interval_sec=interval_sec,
                image_size=image_size,
                max_frames=max_samples_per_video * 4,
            ) as extractor:
                for sample in extractor.iter_frames():
                    if len(kept) >= max_samples_per_video:
                        break
                    frame = sample.image
                    if _is_near_black(frame) or _looks_like_static_menu(frame):
                        continue
                    if last_frame is not None and _pixel_distance(frame, last_frame) < 0.012:
                        continue
                    kept.append(frame)
                    all_frames.append(frame)
                    metadata_frames.append(
                        {
                            "source_url": item.url,
                            "title": item.title,
                            "video_index": video_idx,
                            "frame_index": sample.index,
                            "source_frame_idx": sample.source_frame_idx,
                            "timestamp_sec": sample.timestamp_sec,
                        }
                    )
                    last_frame = frame
        except Exception as exc:
            metadata_frames.append({"source_url": item.url, "video_index": video_idx, "skipped_reason": str(exc)[:300]})
            continue
        if kept:
            embeddings_parts.append(_embed_batches(embedder, kept))

    embeddings = (
        np.concatenate(embeddings_parts, axis=0).astype(np.float32)
        if embeddings_parts
        else np.zeros((0, embedder.embedding_dim), dtype=np.float32)
    )
    np.savez_compressed(out / "reference_bank.npz", embeddings=embeddings)
    metadata = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "backbone": embedder.backbone_name,
        "image_size": int(image_size),
        "interval_sec": float(interval_sec),
        "max_samples_per_video": int(max_samples_per_video),
        "max_videos": int(max_videos),
        "n_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "source_urls": [r.url for r in verified],
        "filtering": {
            "near_black": "mean pixel intensity < 8",
            "near_identical": "mean absolute pixel distance from previous kept frame < 0.012",
            "menu_loading": "conservative low-variance/low-edge heuristic; not a robust menu detector",
        },
        "frames": metadata_frames,
    }
    (out / "reference_bank_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _thumbnail_grid(all_frames, out / "thumbnail_grid.jpg")
    return {"out_dir": str(out), "n_embeddings": int(embeddings.shape[0]), "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gameplay reference bank from verified URLs.")
    parser.add_argument("--urls-file", default=str(DEFAULT_URLS_FILE))
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--max-samples-per-video", type=int, default=120)
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--out-dir", default="data/reference_banks/gameplay_reference")
    args = parser.parse_args()
    result = build_gameplay_reference_bank(**vars(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
