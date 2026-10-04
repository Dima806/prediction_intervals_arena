"""Empirical coverage: the one metric that judges any interval.

An interval that claims nominal coverage ``c`` should contain the true value in a
fraction ``c`` of held-out cases. This measures that fraction.
"""

from __future__ import annotations

import numpy as np


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of ``y`` that falls within ``[lower, upper]`` (inclusive).

    Args:
        y: observed truths, shape ``(n,)``.
        lower: interval lower bounds, shape ``(n,)``.
        upper: interval upper bounds, shape ``(n,)``.

    Returns:
        Empirical coverage in ``[0, 1]``.
    """
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y >= lower) & (y <= upper)
    return float(np.mean(inside))
