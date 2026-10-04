"""Small shared helpers for the interval methods."""

from __future__ import annotations

import numpy as np


def as_2d(x: np.ndarray) -> np.ndarray:
    """Return ``x`` as a 2-D float array of shape ``(n, d)``."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def rearrange(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enforce ``lower <= upper`` element-wise (monotonic rearrangement).

    Separately-fitted quantiles can cross; swapping restores a valid interval and is
    the standard fix. The arena reports when it happens rather than hiding it.
    """
    return np.minimum(lower, upper), np.maximum(lower, upper)


def conformal_level(n: int, coverage: float) -> float:
    """The finite-sample conformal quantile level ``ceil((n+1) * coverage) / n``.

    Taking this (slightly inflated) empirical quantile of the calibration scores is
    what gives split conformal its coverage guarantee under exchangeability.
    """
    return float(min(1.0, np.ceil((n + 1) * coverage) / n))
