"""Central configuration for the GlitchVision demo.

Keeping knobs in one place makes it easy for reviewers to scan the system.
All values are MVP defaults tuned for an 8 GB / CPU-only laptop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
SAMPLES_DIR: Path = DATA_DIR / "samples"
OUTPUTS_DIR: Path = DATA_DIR / "outputs"
REFERENCE_BANKS_DIR: Path = DATA_DIR / "reference_banks"


@dataclass(frozen=True)
class AppDefaults:
    """User-facing defaults that populate the Streamlit controls."""

    frame_interval_sec: float = 1.0
    top_k: int = 12
    contamination: float = 0.05
    max_frames: int = 600  # hard cap to keep CPU demo responsive
    image_size: int = 224
    smoothing_window: int = 3  # temporal smoothing (1 = disabled)
    segment_window: int = 5
    segment_top_k: int = 5
    reference_k: int = 5
    hybrid_weight_within: float = 0.5
    hybrid_weight_reference: float = 0.5


@dataclass(frozen=True)
class ModelConfig:
    """Model-side knobs."""

    backbone: str = "resnet18"
    embedding_dim: int = 512  # ResNet-18 avgpool output
    isolation_forest_estimators: int = 200
    random_state: int = 42


DEFAULTS = AppDefaults()
MODEL = ModelConfig()
