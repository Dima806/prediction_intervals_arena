"""Coverage-first evaluation: the arena is scored here, not in the methods."""

from .comparison import coverage_report, run_arena
from .conditional import conditional_coverage
from .coverage import coverage
from .width import mean_width

__all__ = [
    "conditional_coverage",
    "coverage",
    "coverage_report",
    "mean_width",
    "run_arena",
]
