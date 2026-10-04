"""Interval width (efficiency). At EQUAL coverage, tighter intervals are better.

Width is only meaningful once coverage is matched: a narrow interval that does not
cover is useless, so always read :func:`mean_width` next to coverage, never alone.
"""

from __future__ import annotations

import numpy as np


def mean_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean interval width ``upper - lower``.

    Args:
        lower: interval lower bounds, shape ``(n,)``.
        upper: interval upper bounds, shape ``(n,)``.

    Returns:
        The average width.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean(upper - lower))
