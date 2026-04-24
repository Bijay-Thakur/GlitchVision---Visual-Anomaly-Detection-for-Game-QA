"""Synthetic glitch injection and honest proxy benchmark metrics."""

from .evaluate import BenchmarkMetrics, evaluate_run  # noqa: F401
from .glitch_injection import (  # noqa: F401
    GlitchInterval,
    available_glitch_kinds,
    inject_glitches,
    plan_glitch_schedule,
)
