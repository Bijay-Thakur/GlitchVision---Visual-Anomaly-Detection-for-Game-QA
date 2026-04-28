"""Verify gameplay reference URLs without downloading full videos.

This module reads a small URL list, resolves each URL with ``yt-dlp``, and
performs a one-frame OpenCV probe against the resolved media stream. Failures
are recorded per URL so one bad reference does not break the rest of the
pipeline.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .youtube_stream import YouTubeStreamError, resolve_youtube_stream


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URLS_FILE = _REPO_ROOT / "data" / "reference_videos.txt"
LEGACY_URLS_FILE = _REPO_ROOT / "data" / "reference_banks" / "reference_videos.txt"
DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "outputs" / "reference_verification"


@dataclass
class UrlVerificationResult:
    url: str
    ok: bool
    reason: str = ""
    title: str = ""
    duration_sec: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    stream_url: str = ""
    first_frame_shape: str = ""


def ensure_url_file(path: str | Path = DEFAULT_URLS_FILE) -> Path:
    """Create a small commented URL file if it does not exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(
            "# Paste known-good gameplay YouTube URLs below, one per line.\n"
            "# Lines starting with # are ignored. Videos are probed/streamed;\n"
            "# GlitchVision does not download full source videos.\n"
            "# Example:\n"
            "# https://www.youtube.com/watch?v=dQw4w9WgXcQ\n",
            encoding="utf-8",
        )
    return p


def resolve_url_file(path: str | Path = DEFAULT_URLS_FILE) -> Path:
    """Return the requested URL file, falling back to the older bank path."""
    p = Path(path)
    if p.exists():
        return p
    try:
        refers_to_default = p.resolve() == DEFAULT_URLS_FILE.resolve()
    except OSError:
        refers_to_default = str(p).replace("\\", "/") == str(DEFAULT_URLS_FILE).replace(
            "\\", "/"
        )
    if refers_to_default and LEGACY_URLS_FILE.exists():
        return LEGACY_URLS_FILE
    return ensure_url_file(p)


def read_reference_urls(path: str | Path = DEFAULT_URLS_FILE) -> List[str]:
    """Read non-empty, non-comment URLs from ``path``."""
    p = resolve_url_file(path)
    urls: List[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def _probe_first_frame(stream_url: str) -> str:
    import cv2

    cap = cv2.VideoCapture(stream_url)
    try:
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open the resolved stream")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("OpenCV could not read the first frame")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"Unexpected first-frame shape: {frame.shape}")
        return "x".join(str(int(v)) for v in frame.shape)
    finally:
        cap.release()


def _reject_reason_from_exception(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    for marker in (
        "private",
        "age",
        "region",
        "unavailable",
        "live",
        "copyright",
        "sign in",
        "members-only",
    ):
        if marker in lowered:
            return marker.replace("-", " ")
    return text[:300] or exc.__class__.__name__


def verify_url(url: str) -> UrlVerificationResult:
    """Resolve and one-frame-probe a single URL."""
    try:
        resolved = resolve_youtube_stream(url)
        shape = _probe_first_frame(resolved.stream_url)
        return UrlVerificationResult(
            url=url,
            ok=True,
            title=resolved.title,
            duration_sec=resolved.duration_sec,
            width=resolved.width,
            height=resolved.height,
            stream_url=resolved.stream_url,
            first_frame_shape=shape,
        )
    except (YouTubeStreamError, Exception) as exc:
        return UrlVerificationResult(
            url=url,
            ok=False,
            reason=_reject_reason_from_exception(exc),
        )


def verify_reference_urls(
    urls_file: str | Path = DEFAULT_URLS_FILE,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    max_urls: Optional[int] = None,
) -> List[UrlVerificationResult]:
    """Verify every URL and write text/CSV/Markdown reports."""
    urls = read_reference_urls(urls_file)
    if max_urls is not None:
        urls = urls[: max(0, int(max_urls))]

    results = [verify_url(url) for url in urls]
    write_verification_outputs(results, out_dir)
    return results


def write_verification_outputs(
    results: Iterable[UrlVerificationResult],
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(results)

    verified = [r for r in rows if r.ok]
    rejected = [r for r in rows if not r.ok]

    (out / "verified_urls.txt").write_text(
        "\n".join(r.url for r in verified) + ("\n" if verified else ""),
        encoding="utf-8",
    )

    with (out / "rejected_urls.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "reason", "title"])
        writer.writeheader()
        for r in rejected:
            writer.writerow({"url": r.url, "reason": r.reason, "title": r.title})

    report_lines = [
        "# Reference URL Verification",
        "",
        f"- Checked URLs: {len(rows)}",
        f"- Verified: {len(verified)}",
        f"- Rejected: {len(rejected)}",
        "",
        "## Verified URLs",
        "",
    ]
    if verified:
        for r in verified:
            dims = f"{r.width or '?'}x{r.height or '?'}"
            report_lines.append(f"- {r.title or r.url} ({dims}, {r.duration_sec or '?'} sec)")
    else:
        report_lines.append("- None yet. Paste available gameplay URLs into the URL file.")
    report_lines.extend(["", "## Rejected URLs", ""])
    if rejected:
        for r in rejected:
            report_lines.append(f"- {r.url}: {r.reason}")
    else:
        report_lines.append("- None")

    (out / "verification_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    with (out / "verification_results.json").open("w", encoding="utf-8") as f:
        import json

        json.dump([asdict(r) for r in rows], f, indent=2)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify gameplay reference URLs.")
    parser.add_argument("--urls-file", default=str(DEFAULT_URLS_FILE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-urls", type=int, default=None)
    args = parser.parse_args()

    results = verify_reference_urls(args.urls_file, args.out_dir, args.max_urls)
    ok = sum(1 for r in results if r.ok)
    print(f"Verified {ok}/{len(results)} URLs. Report: {Path(args.out_dir) / 'verification_report.md'}")


if __name__ == "__main__":
    main()
