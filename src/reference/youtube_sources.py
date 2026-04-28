"""Resolve known-good YouTube page URLs to OpenCV-readable stream URLs.

Used when building a :class:`~src.reference.ReferenceBank` from a text list of
links (no full-video download).
"""
from __future__ import annotations

from pathlib import Path

from typing import Callable, Sequence

from ..ingestion import YouTubeStreamError, resolve_youtube_stream

ProgressFn = Callable[[int, int], None]


def page_urls_to_stream_sources(
    page_urls: Sequence[str],
    *,
    on_progress: ProgressFn | None = None,
    cookies_file: str | Path | None = None,
    cookies_from_browser: str | None = None,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Return ``(stream_url, label)`` tuples plus human-readable error lines.

    One failure does not block other URLs.
    """
    sources: list[tuple[str, str]] = []
    errors: list[str] = []
    total = len(page_urls)
    for i, page_url in enumerate(page_urls):
        if on_progress is not None:
            on_progress(i + 1, total)
        try:
            resolved = resolve_youtube_stream(
                page_url,
                cookies_file=cookies_file,
                cookies_from_browser=cookies_from_browser,
            )
            label = (resolved.title or page_url)[:200]
            sources.append((resolved.stream_url, label))
        except YouTubeStreamError as exc:
            errors.append(f"{page_url}: {exc}")
        except Exception as exc:
            errors.append(f"{page_url}: {exc}")
    return sources, errors
