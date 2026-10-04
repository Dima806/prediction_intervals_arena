"""Conditional coverage: coverage WITHIN regions of the prediction range.

Overall coverage can look fine while the interval is badly miscalibrated locally --
too wide where the model is confident, too narrow where it is not. Binning by a
conditioning variable (``x`` or the prediction) exposes exactly that failure, which
is the showpiece of notebook 02.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def conditional_coverage(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    by: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Coverage and width within equal-count bins of ``by``.

    Args:
        y: observed truths, shape ``(n,)``.
        lower: interval lower bounds, shape ``(n,)``.
        upper: interval upper bounds, shape ``(n,)``.
        by: conditioning variable (e.g. the feature or the prediction), shape ``(n,)``.
        n_bins: number of quantile bins.

    Returns:
        One row per non-empty bin with columns ``bin``, ``center``, ``coverage``,
        ``mean_width`` and ``count``.
    """
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    by = np.asarray(by, dtype=float).ravel()

    edges = np.quantile(by, np.linspace(0.0, 1.0, n_bins + 1))
    # Assign each point to a bin from the interior edges; endpoints fall in [0, n_bins-1].
    idx = np.clip(np.digitize(by, edges[1:-1]), 0, n_bins - 1)
    inside = (y >= lower) & (y <= upper)

    rows: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": float(b),
                "center": float(np.mean(by[mask])),
                "coverage": float(np.mean(inside[mask])),
                "mean_width": float(np.mean(upper[mask] - lower[mask])),
                "count": float(mask.sum()),
            }
        )
    return pd.DataFrame(rows)
