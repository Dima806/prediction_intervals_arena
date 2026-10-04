"""The arena harness: run methods across error structures, score on coverage and width.

Nothing here assumes a particular method class; any object exposing
``fit(x, y)`` and ``predict_interval(x) -> (lower, upper)`` can enter the arena.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..datasets.synthetic import make_synthetic
from .coverage import coverage as coverage_fn
from .width import mean_width


class IntervalMethod(Protocol):
    """Structural type every interval method satisfies."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> Any: ...

    def predict_interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def coverage_report(
    methods: Mapping[str, IntervalMethod],
    x_test: np.ndarray,
    y_test: np.ndarray,
    nominal: float = 0.90,
) -> pd.DataFrame:
    """Score already-fitted methods on held-out data: actual vs nominal coverage.

    This is the function notebook 06 ships so the reader measures rather than trusts.
    """
    rows: list[dict[str, Any]] = []
    for name, method in methods.items():
        lower, upper = method.predict_interval(x_test)
        rows.append(
            {
                "method": name,
                "coverage": coverage_fn(y_test, lower, upper),
                "nominal": nominal,
                "mean_width": mean_width(lower, upper),
            }
        )
    return pd.DataFrame(rows)


def run_arena(
    method_factories: Mapping[str, Callable[[], IntervalMethod]],
    structures: Sequence[str],
    n_samples: int,
    seed: int,
    nominal: float = 0.90,
    test_fraction: float = 0.30,
) -> pd.DataFrame:
    """Fit every method on every error structure and score each on held-out data.

    Args:
        method_factories: name -> zero-arg callable returning a FRESH method instance.
        structures: error structures to sweep (see ``datasets.synthetic.STRUCTURES``).
        n_samples: rows per synthetic dataset.
        seed: reproducibility seed (also seeds the train/test split).
        nominal: the coverage every interval promises.
        test_fraction: held-out fraction used to measure coverage.

    Returns:
        A tidy frame with one row per (structure, method): ``coverage`` and ``mean_width``.
    """
    rows: list[dict[str, Any]] = []
    for structure in structures:
        data = make_synthetic(structure, n_samples=n_samples, seed=seed)
        x_train, x_test, y_train, y_test = train_test_split(
            data.x, data.y, test_size=test_fraction, random_state=seed
        )
        for name, factory in method_factories.items():
            method = factory()
            method.fit(x_train, y_train)
            lower, upper = method.predict_interval(x_test)
            rows.append(
                {
                    "structure": structure,
                    "method": name,
                    "coverage": coverage_fn(y_test, lower, upper),
                    "nominal": nominal,
                    "mean_width": mean_width(lower, upper),
                }
            )
    return pd.DataFrame(rows)
