"""The evaluation primitives must be correct against known truth."""

from __future__ import annotations

import numpy as np

from src.evaluation.conditional import conditional_coverage
from src.evaluation.coverage import coverage
from src.evaluation.width import mean_width


def test_coverage_counts_inclusively() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lower = np.array([0.0, 0.0, 5.0, -1.0])  # point 2 (y=2) excluded; others inside
    upper = np.array([0.5, 2.0, 6.0, 3.0])
    # inside: y0=0 in [0,0.5] yes; y1=1 in [0,2] yes; y2=2 in [5,6] no; y3=3 in [-1,3] yes
    assert coverage(y, lower, upper) == 0.75


def test_coverage_all_or_nothing() -> None:
    y = np.zeros(10)
    assert coverage(y, np.full(10, -1.0), np.full(10, 1.0)) == 1.0
    assert coverage(y, np.full(10, 2.0), np.full(10, 3.0)) == 0.0


def test_mean_width() -> None:
    lower = np.array([0.0, 1.0, 2.0])
    upper = np.array([2.0, 2.0, 5.0])  # widths 2, 1, 3
    assert mean_width(lower, upper) == 2.0


def test_conditional_coverage_detects_region_failure() -> None:
    """A constant-width band over a widening target should undercover the high region."""
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0.0, 10.0, size=n)
    y = x + rng.normal(0.0, 0.2 + 0.6 * x, size=n)  # noise grows with x
    half = 2.0  # one constant half-width
    lower, upper = x - half, x + half

    table = conditional_coverage(y, lower, upper, by=x, n_bins=5)
    assert len(table) == 5
    assert (table["count"] > 0).all()
    # Calm low-x region should cover more than the volatile high-x region.
    assert table.iloc[0]["coverage"] > table.iloc[-1]["coverage"]
