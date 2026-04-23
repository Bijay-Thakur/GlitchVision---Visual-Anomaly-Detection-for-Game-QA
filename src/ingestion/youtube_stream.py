"""Resolve a playable stream URL from a YouTube link using yt-dlp.

Design note
-----------
We intentionally do NOT download the full source video to disk. Instead we ask
yt-dlp to extract the direct media URL of the best progressive (muxed mp4)
stream and hand that URL to OpenCV's ``VideoCapture``. OpenCV (backed by
FFmpeg) can then read frames over HTTP without persisting the source file.

This is an MVP approach. Some format combinations (e.g. DASH-only videos on
certain clients) are not directly readable by OpenCV. In those cases we raise
``YouTubeStreamError`` with a friendly message so the UI can fall back to a
local upload.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class YouTubeStreamError(RuntimeError):
    """Raised when we cannot obtain a frame-readable stream URL."""


@dataclass(frozen=True)
class ResolvedStream:
    """Metadata for a resolved YouTube stream."""

    stream_url: str
    title: str
    duration_sec: Optional[float]
    format_id: Optional[str]
    ext: Optional[str]
    width: Optional[int]
    height: Optional[int]


def _pick_progressive_format(info: dict) -> Optional[dict]:
    """Choose the best progressive (video+audio muxed) MP4 format.

    OpenCV reads progressive MP4/H.264 most reliably. We prefer moderate
    resolutions (<=720p) to keep CPU decoding responsive on a laptop.
    """
    formats = info.get("formats") or []

    def is_progressive(fmt: dict) -> bool:
        vcodec = (fmt.get("vcodec") or "none").lower()
        acodec = (fmt.get("acodec") or "none").lower()
        ext = (fmt.get("ext") or "").lower()
        proto = (fmt.get("protocol") or "").lower()
        # muxed + mp4 + plain http(s) is what OpenCV handles best
        return (
            vcodec != "none"
            and acodec != "none"
            and ext == "mp4"
            and ("http" in proto or proto in {"", "https"})
        )

    candidates = [f for f in formats if is_progressive(f) and f.get("url")]
    if not candidates:
        # Second attempt: video-only progressive mp4 (no audio). OpenCV only
        # needs video frames, so this still works for our pipeline.
        def is_video_only_mp4(fmt: dict) -> bool:
            vcodec = (fmt.get("vcodec") or "none").lower()
            ext = (fmt.get("ext") or "").lower()
            proto = (fmt.get("protocol") or "").lower()
            return (
                vcodec != "none"
                and ext == "mp4"
                and ("http" in proto or proto in {"", "https"})
            )

        candidates = [f for f in formats if is_video_only_mp4(f) and f.get("url")]

    if not candidates:
        return None

    def score(fmt: dict) -> tuple:
        height = fmt.get("height") or 0
        # Prefer <=720p (closest to 720 without going way over)
        penalty = abs(min(height, 720) - 720) + max(0, height - 720) * 2
        tbr = fmt.get("tbr") or 0
        return (-penalty, -tbr)  # higher-quality first, capped at ~720p

    candidates.sort(key=score)
    return candidates[0]


def resolve_youtube_stream(url: str) -> ResolvedStream:
    """Return a streamable URL for OpenCV.

    Parameters
    ----------
    url:
        A YouTube video URL (youtube.com / youtu.be / shorts).

    Raises
    ------
    YouTubeStreamError
        If yt-dlp is missing, the URL cannot be resolved, or no
        OpenCV-friendly format is available.
    """
    if not url or not isinstance(url, str):
        raise YouTubeStreamError("Please provide a valid YouTube URL.")

    try:
        import yt_dlp  # imported lazily so the app starts even if missing
    except ImportError as exc:  # pragma: no cover - environment check
        raise YouTubeStreamError(
            "yt-dlp is not installed. Run: pip install -r requirements.txt"
        ) from exc

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        # Hint yt-dlp to prefer progressive mp4 first.
        "format": "best[ext=mp4][protocol^=http]/best[ext=mp4]/best",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt-dlp raises many subclasses; keep it simple
        raise YouTubeStreamError(
            f"Could not resolve that YouTube link: {exc}"
        ) from exc

    if info is None:
        raise YouTubeStreamError("yt-dlp returned no metadata for that URL.")

    chosen = _pick_progressive_format(info)
    # Fall back to the top-level URL if yt-dlp already selected one for us.
    stream_url = (chosen or {}).get("url") or info.get("url")
    if not stream_url:
        raise YouTubeStreamError(
            "No OpenCV-compatible progressive stream was found for this video. "
            "Please use the local upload fallback."
        )

    return ResolvedStream(
        stream_url=stream_url,
        title=str(info.get("title") or "YouTube video"),
        duration_sec=float(info["duration"]) if info.get("duration") else None,
        format_id=(chosen or {}).get("format_id"),
        ext=(chosen or {}).get("ext"),
        width=(chosen or {}).get("width"),
        height=(chosen or {}).get("height"),
    )
