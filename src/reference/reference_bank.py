"""Durable storage for a "known-good" reference embedding bank.

A reference bank is a compact, reusable artifact that captures what the
*expected* visual distribution of a build or a gameplay capture looks
like. It stores:

- the embedding matrix ``(N, D)`` as ``float32``
- per-frame metadata (source label, timestamp, frame index)
- configuration metadata (backbone name, image size, interval,
  creation time, number of source videos)

We use a simple ``.npz`` (vectors) + ``.json`` (metadata) split so the
bank is easy to inspect, diff, and reuse from a notebook or a CI script.

The file layout under a ``<bank_dir>/`` directory is:

    <bank_dir>/
        embeddings.npz     # 'embeddings' array
        metadata.json      # bank + per-frame metadata
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


EMBEDDINGS_FILENAME = "embeddings.npz"
METADATA_FILENAME = "metadata.json"


@dataclass
class FrameRef:
    """Per-frame metadata in a reference bank."""

    source_label: str
    frame_index: int
    timestamp_sec: float


@dataclass
class ReferenceBank:
    """In-memory representation of a reference embedding bank."""

    embeddings: np.ndarray
    frames: List[FrameRef]
    backbone: str = "resnet18"
    image_size: int = 224
    interval_sec: float = 1.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )
    source_videos: List[str] = field(default_factory=list)
    notes: str = ""

    # -------------------------------------------------------------
    # construction helpers
    # -------------------------------------------------------------
    def __post_init__(self) -> None:
        emb = np.asarray(self.embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(
                f"Reference embeddings must be 2-D; got shape {emb.shape}"
            )
        if emb.shape[0] != len(self.frames):
            raise ValueError(
                "Number of embeddings does not match number of frame metadata "
                f"entries ({emb.shape[0]} vs {len(self.frames)})."
            )
        self.embeddings = emb

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    # -------------------------------------------------------------
    # persistence
    # -------------------------------------------------------------
    def save(self, bank_dir: str | Path) -> Path:
        """Save the bank to a directory. Creates it if needed."""
        out = Path(bank_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(out / EMBEDDINGS_FILENAME, embeddings=self.embeddings)

        meta: Dict[str, Any] = {
            "backbone": self.backbone,
            "image_size": int(self.image_size),
            "interval_sec": float(self.interval_sec),
            "created_at": self.created_at,
            "source_videos": list(self.source_videos),
            "embedding_dim": self.embedding_dim,
            "size": self.size,
            "notes": self.notes,
            "frames": [asdict(f) for f in self.frames],
        }
        (out / METADATA_FILENAME).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, bank_dir: str | Path) -> "ReferenceBank":
        """Load a bank saved by :meth:`save`."""
        src = Path(bank_dir)
        emb_path = src / EMBEDDINGS_FILENAME
        meta_path = src / METADATA_FILENAME
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Reference bank is missing files under {src!s}. "
                f"Expected {EMBEDDINGS_FILENAME} and {METADATA_FILENAME}."
            )

        with np.load(emb_path) as data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        frames = [
            FrameRef(
                source_label=str(f.get("source_label", "")),
                frame_index=int(f.get("frame_index", -1)),
                timestamp_sec=float(f.get("timestamp_sec", 0.0)),
            )
            for f in meta.get("frames", [])
        ]

        return cls(
            embeddings=embeddings,
            frames=frames,
            backbone=str(meta.get("backbone", "resnet18")),
            image_size=int(meta.get("image_size", 224)),
            interval_sec=float(meta.get("interval_sec", 1.0)),
            created_at=str(meta.get("created_at", "")),
            source_videos=list(meta.get("source_videos", [])),
            notes=str(meta.get("notes", "")),
        )


def build_reference_bank(
    embeddings: np.ndarray,
    frames: List[FrameRef],
    *,
    backbone: str,
    image_size: int,
    interval_sec: float,
    source_videos: Optional[List[str]] = None,
    notes: str = "",
) -> ReferenceBank:
    """Convenience factory; validates inputs and stamps metadata."""
    return ReferenceBank(
        embeddings=np.asarray(embeddings, dtype=np.float32),
        frames=list(frames),
        backbone=backbone,
        image_size=int(image_size),
        interval_sec=float(interval_sec),
        source_videos=list(source_videos or []),
        notes=notes,
    )
