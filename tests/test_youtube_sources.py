from __future__ import annotations

from src.ingestion.youtube_stream import ResolvedStream, YouTubeStreamError
from src.reference import youtube_sources


def test_page_urls_to_stream_sources_partial_failures(monkeypatch):
    def fake_resolve(url: str, **_kwargs: object) -> ResolvedStream:
        if "bad" in url:
            raise YouTubeStreamError("unavailable")
        return ResolvedStream(
            stream_url="https://example.com/stream.mp4",
            title="t",
            duration_sec=1.0,
            format_id="18",
            ext="mp4",
            width=640,
            height=360,
        )

    monkeypatch.setattr(youtube_sources, "resolve_youtube_stream", fake_resolve)
    ok, bad = ["https://y/watch?v=1", "https://y/bad"], []
    sources, errors = youtube_sources.page_urls_to_stream_sources(ok)

    assert len(sources) == 1
    assert len(errors) == 1
    assert "bad" in errors[0]

    progress_hits: list[tuple[int, int]] = []

    def on_prog(cur: int, tot: int) -> None:
        progress_hits.append((cur, tot))

    youtube_sources.page_urls_to_stream_sources(ok, on_progress=on_prog)
    assert progress_hits == [(1, 2), (2, 2)]
