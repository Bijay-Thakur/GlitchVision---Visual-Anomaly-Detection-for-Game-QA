from __future__ import annotations

from pathlib import Path

from src.ingestion import url_reference_loader as loader
from src.ingestion.url_reference_loader import UrlVerificationResult


def test_url_file_parsing_ignores_blank_lines_and_comments(tmp_path: Path):
    path = tmp_path / "reference_videos.txt"
    path.write_text(
        "\n# comment\nhttps://example.com/a\n   \nhttps://example.com/b\n",
        encoding="utf-8",
    )
    assert loader.read_reference_urls(path) == ["https://example.com/a", "https://example.com/b"]


def test_rejected_url_handling_mocked(tmp_path: Path, monkeypatch):
    path = tmp_path / "reference_videos.txt"
    path.write_text("https://example.com/private\n", encoding="utf-8")

    def fake_verify(url: str) -> UrlVerificationResult:
        return UrlVerificationResult(url=url, ok=False, reason="private")

    monkeypatch.setattr(loader, "verify_url", fake_verify)
    results = loader.verify_reference_urls(path, tmp_path / "out")
    assert len(results) == 1
    assert not results[0].ok
    assert (tmp_path / "out" / "rejected_urls.csv").exists()
    assert "private" in (tmp_path / "out" / "verification_report.md").read_text(encoding="utf-8")
