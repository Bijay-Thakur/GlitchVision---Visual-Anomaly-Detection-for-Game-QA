"""Build a Streamlit-compatible reference bank from ``data/reference_videos.txt``.

Writes ``embeddings.npz`` + ``metadata.json`` under ``data/reference_banks/<name>/``.
Run from repo root::

    python scripts/build_reference_bank_from_urls.py --bank-name known_good_v1

Optional: ``--max-urls N`` for a quick smoke test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import (  # noqa: E402
    DEFAULTS,
    LEGACY_REFERENCE_VIDEOS_FILE,
    REFERENCE_BANKS_DIR,
    REFERENCE_VIDEOS_FILE,
)
from src.pipeline import GlitchVisionPipeline, PipelineConfig  # noqa: E402
from src.reference import youtube_sources  # noqa: E402


def _effective_urls_file() -> Path:
    if REFERENCE_VIDEOS_FILE.exists():
        return REFERENCE_VIDEOS_FILE
    if LEGACY_REFERENCE_VIDEOS_FILE.exists():
        return LEGACY_REFERENCE_VIDEOS_FILE
    return REFERENCE_VIDEOS_FILE


def _parse_urls(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reference bank from URL list.")
    parser.add_argument("--bank-name", default="known_good_v1")
    parser.add_argument("--urls-file", type=Path, default=None)
    parser.add_argument("--interval-sec", type=float, default=float(DEFAULTS.frame_interval_sec))
    parser.add_argument("--max-frames", type=int, default=int(DEFAULTS.max_frames))
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--max-urls", type=int, default=None, help="Only use N URLs (for testing).")
    parser.add_argument(
        "--cookies_file",
        type=Path,
        default=None,
        help="Netscape cookies.txt for yt-dlp (YouTube bot check).",
    )
    parser.add_argument(
        "--cookies-from-browser",
        dest="cookies_from_browser",
        default=None,
        metavar="BROWSER",
        help="e.g. firefox, edge, chrome, opera_gx (Opera GX), opera — yt-dlp reads browser cookies.",
    )
    args = parser.parse_args()

    url_path = args.urls_file or _effective_urls_file()
    urls = _parse_urls(url_path)
    if args.max_urls is not None:
        urls = urls[: max(0, args.max_urls)]

    if not urls:
        print(f"No URLs in {url_path}. Add one YouTube URL per line.", file=sys.stderr)
        return 1

    print(f"Resolving {len(urls)} URL(s) from {url_path}...")
    cf = args.cookies_file if args.cookies_file and args.cookies_file.is_file() else None
    sources, errors = youtube_sources.page_urls_to_stream_sources(
        urls,
        on_progress=lambda cur, tot: print(f"  resolve {cur}/{tot}", flush=True),
        cookies_file=cf,
        cookies_from_browser=args.cookies_from_browser,
    )
    for err in errors:
        print(f"  SKIP {err}", file=sys.stderr)
    if not sources:
        print("No streams could be opened.", file=sys.stderr)
        return 1

    out_dir = REFERENCE_BANKS_DIR / args.bank_name.strip()
    cfg = PipelineConfig(
        interval_sec=float(args.interval_sec),
        image_size=int(DEFAULTS.image_size),
        max_frames=int(args.max_frames),
        backbone=args.backbone,
    )
    pipeline = GlitchVisionPipeline(cfg)
    print(f"Building bank -> {out_dir} ({len(sources)} video(s))...")
    bank = pipeline.build_reference(
        sources,
        out_dir=out_dir,
        progress=lambda p, m: print(f"  {100.0 * p:.0f}% {m}", flush=True),
        notes=f"CLI from {url_path.name}; {len(sources)}/{len(urls)} URLs OK",
    )
    print(f"Done. {bank.size} embeddings, backbone={bank.backbone}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
