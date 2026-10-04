"""Bootstrap intervals: resample the training data, refit, read off the spread.

Honest caveat built into the method: the plain bootstrap captures uncertainty in the
ESTIMATE (where the fitted curve sits) far more than the full PREDICTIVE uncertainty
(where the next observation lands), so it tends to be too narrow and to undercover.
The arena measures that shortfall rather than papering over it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor

from .base import as_2d


class BootstrapInterval:
    """Percentile interval over predictions from models refit on bootstrap resamples."""

    def __init__(
        self,
        base_estimator: Any = None,
        n_bootstrap: int = 100,
        coverage: float = 0.90,
        random_state: int = 0,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_bootstrap = n_bootstrap
        self.coverage = coverage
        self.random_state = random_state
        self._models: list[Any] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> BootstrapInterval:
        """Refit the base estimator on ``n_bootstrap`` resamples of the training data."""
        x2 = as_2d(x)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n = x2.shape[0]
        self._models = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            model = (
                clone(self.base_estimator)
                if self.base_estimator is not None
                else GradientBoostingRegressor(random_state=0)
            )
            model.fit(x2[idx], y[idx])
            self._models.append(model)
        return self

    def predict_interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``alpha/2`` and ``1 - alpha/2`` percentiles across the refits."""
        x2 = as_2d(x)
        preds = np.vstack([model.predict(x2) for model in self._models])
        alpha = 1.0 - self.coverage
        lower = np.percentile(preds, 100.0 * (alpha / 2.0), axis=0)
        upper = np.percentile(preds, 100.0 * (1.0 - alpha / 2.0), axis=0)
        return lower, upper
