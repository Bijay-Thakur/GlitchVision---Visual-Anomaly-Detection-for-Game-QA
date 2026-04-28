from __future__ import annotations

from pathlib import Path

from src.ingestion import youtube_stream as ys


def test_opera_gx_profile_dir_env_override(tmp_path: Path, monkeypatch):
    gx = tmp_path / "Opera GX Stable"
    gx.mkdir(parents=True)
    monkeypatch.setenv("GLITCHVISION_OPERA_GX_USER_DATA", str(gx))
    assert ys.opera_gx_profile_dir() == gx.resolve()


def test_cookie_opts_opera_gx_uses_tuple(tmp_path: Path, monkeypatch):
    gx = tmp_path / "Opera GX Stable"
    gx.mkdir(parents=True)
    monkeypatch.setenv("GLITCHVISION_OPERA_GX_USER_DATA", str(gx))
    opts = ys._cookie_ydl_opts(cookies_from_browser="opera_gx")
    assert opts["cookiesfrombrowser"] == ("opera", str(gx.resolve()))


def test_cookie_opts_plain_opera_tuple():
    opts = ys._cookie_ydl_opts(cookies_from_browser="edge")
    assert opts["cookiesfrombrowser"] == ("edge",)
