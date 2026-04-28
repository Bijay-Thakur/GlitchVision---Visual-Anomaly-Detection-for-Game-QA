"""Gameplay-reference benchmark utilities for GlitchVision."""

from .evaluate import compute_metrics, save_metrics  # noqa: F401
from .glitch_injection import (  # noqa: F401
    GameGlitchInterval,
    GlitchDataset,
    inject_gameplay_glitches,
    plan_gameplay_glitches,
)
