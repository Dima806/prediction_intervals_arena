"""Normal-theory interval: the textbook default and the arena's cautionary protagonist.

``prediction +/- z * residual_std`` -- a single constant width for every prediction.
It is correct only when the errors are constant-variance, symmetric and normal. The
arena breaks each of those assumptions and measures what happens to coverage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor

from .base import as_2d


class NormalInterval:
    """Constant-width interval from the residual standard deviation and a normal quantile."""

    def __init__(self, base_estimator: Any = None, coverage: float = 0.90) -> None:
        self.base_estimator = base_estimator
        self.coverage = coverage
        self._model: Any = None
        self._sigma: float = 0.0
        self._z: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> NormalInterval:
        """Fit the mean model and estimate one residual standard deviation."""
        y = np.asarray(y, dtype=float)
        self._model = (
            clone(self.base_estimator)
            if self.base_estimator is not None
            else GradientBoostingRegressor(random_state=0)
        )
        x2 = as_2d(x)
        self._model.fit(x2, y)
        residuals = y - self._model.predict(x2)
        self._sigma = float(np.std(residuals, ddof=1))
        self._z = float(norm.ppf(0.5 + self.coverage / 2.0))
        return self

    def predict_interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the constant-width interval ``prediction +/- z * sigma``."""
        prediction = self._model.predict(as_2d(x))
        half_width = self._z * self._sigma
        return prediction - half_width, prediction + half_width
