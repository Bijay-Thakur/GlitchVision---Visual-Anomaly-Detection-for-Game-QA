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

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union


class YouTubeStreamError(RuntimeError):
    """Raised when we cannot obtain a frame-readable stream URL."""


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text or "")


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


def opera_gx_profile_dir() -> Optional[Path]:
    """Return Opera GX *user data* folder, or None if not found.

    yt-dlp only knows ``opera`` = Opera **Stable**; Opera GX lives under a
    different directory. Passing this path as the ``profile`` argument makes
    yt-dlp read the correct Cookies DB.
    """
    override = os.environ.get("GLITCHVISION_OPERA_GX_USER_DATA", "").strip()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        candidates.append(Path(appdata) / "Opera Software" / "Opera GX Stable")
    # Some installs use a slightly different folder name
    if appdata:
        candidates.append(Path(appdata) / "Opera Software" / "Opera GX")
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_dir():
            return p.resolve()
    return None


def _cookie_ydl_opts(
    *,
    cookies_file: Union[str, Path, None] = None,
    cookies_from_browser: Optional[str] = None,
) -> Dict[str, Any]:
    """Options for yt-dlp when YouTube requires login / anti-bot cookies."""
    out: Dict[str, Any] = {}
    cf = cookies_file
    if cf is None:
        for key in ("GLITCHVISION_YOUTUBE_COOKIES", "YT_DLP_COOKIE_FILE"):
            raw = os.environ.get(key, "").strip()
            if raw:
                cf = Path(raw)
                break
    if cf:
        p = Path(cf)
        if p.is_file():
            out["cookiefile"] = str(p.resolve())

    browser = (cookies_from_browser or "").strip().lower() or None
    if browser is None:
        raw = os.environ.get("GLITCHVISION_YOUTUBE_COOKIES_BROWSER", "").strip().lower()
        if raw:
            browser = raw
    if browser:
        if browser == "opera_gx":
            gx = opera_gx_profile_dir()
            if gx is not None:
                # yt-dlp ``opera`` = Opera Stable path; Opera GX needs explicit user-data dir.
                out["cookiesfrombrowser"] = ("opera", str(gx))
            # No GX folder: omit browser cookies (wrong to fall back to Opera Stable).
        else:
            out["cookiesfrombrowser"] = (browser,)

    # Explicit cookie file wins over browser extraction.
    if "cookiefile" in out:
        out.pop("cookiesfrombrowser", None)

    return out


def resolve_youtube_stream(
    url: str,
    *,
    cookies_file: Union[str, Path, None] = None,
    cookies_from_browser: Optional[str] = None,
) -> ResolvedStream:
    """Return a streamable URL for OpenCV.

    Parameters
    ----------
    url:
        A YouTube video URL (youtube.com / youtu.be / shorts).
    cookies_file:
        Netscape-format cookies file. Optional; can also set env
        ``GLITCHVISION_YOUTUBE_COOKIES`` or ``YT_DLP_COOKIE_FILE``.
    cookies_from_browser:
        Browser name for yt-dlp (e.g. ``\"chrome\"``, ``\"edge\"``, ``\"firefox\"``).
        Use ``\"opera_gx\"`` for **Opera GX** (see :func:`opera_gx_profile_dir`). Optional;
        env ``GLITCHVISION_YOUTUBE_COOKIES_BROWSER`` also works.

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

    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        # Hint yt-dlp to prefer progressive mp4 first.
        "format": "best[ext=mp4][protocol^=http]/best[ext=mp4]/best",
    }
    ydl_opts.update(
        _cookie_ydl_opts(
            cookies_file=cookies_file,
            cookies_from_browser=cookies_from_browser,
        )
    )

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt-dlp raises many subclasses; keep it simple
        msg = _strip_ansi(str(exc))
        raise YouTubeStreamError(
            f"Could not resolve that YouTube link: {msg}"
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
