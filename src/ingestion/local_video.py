"""Local video file ingestion (fallback path for the demo).

The Streamlit UI's file uploader hands us a ``BytesIO``-like object. We write
it to a temp file so OpenCV can read from a real path. Temp files live in the
OS temp dir and are cleaned up by the caller via ``LocalVideoSource.cleanup``.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional


SUPPORTED_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@dataclass
class LocalVideoSource:
    """Handle to a local video file on disk."""

    path: Path
    display_name: str
    is_temp: bool = False

    def cleanup(self) -> None:
        if self.is_temp and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                # Non-fatal; the OS will eventually clean the temp dir.
                pass


def save_uploaded_file(
    fileobj: BinaryIO,
    original_name: Optional[str] = None,
) -> LocalVideoSource:
    """Persist a Streamlit upload to a temp file and return a source handle.

    We keep the original extension so OpenCV/FFmpeg can auto-detect the codec.
    """
    suffix = ""
    if original_name:
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_EXTS:
            suffix = ".mp4"  # best-effort default
    else:
        suffix = ".mp4"

    tmp = tempfile.NamedTemporaryFile(
        prefix="glitchvision_", suffix=suffix, delete=False
    )
    try:
        fileobj.seek(0)
    except Exception:
        pass
    try:
        tmp.write(fileobj.read())
    finally:
        tmp.close()

    path = Path(tmp.name)
    display = original_name or os.path.basename(tmp.name)
    return LocalVideoSource(path=path, display_name=display, is_temp=True)


def from_path(path: str | Path) -> LocalVideoSource:
    """Wrap an existing path on disk as a ``LocalVideoSource``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {p}")
    return LocalVideoSource(path=p, display_name=p.name, is_temp=False)
