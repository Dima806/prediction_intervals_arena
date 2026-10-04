"""Native gradient-boosting quantiles: the practical version of the same idea.

Gradient boosting has a built-in quantile (pinball) objective. Fit one booster for the
lower bound and one for the upper bound and read the band off directly. This is the
powerful, library-provided method most practitioners have but rarely use for intervals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .base import as_2d, rearrange


class GBMQuantileInterval:
    """Two gradient boosters trained on the quantile loss for the lower and upper bounds."""

    def __init__(
        self,
        quantiles: tuple[float, float] = (0.05, 0.95),
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        random_state: int = 0,
    ) -> None:
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._lower_model: Any = None
        self._upper_model: Any = None

    def _make_model(self, alpha: float) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> GBMQuantileInterval:
        """Fit the two quantile boosters."""
        x2 = as_2d(x)
        y = np.asarray(y, dtype=float)
        tau_low, tau_high = self.quantiles
        self._lower_model = self._make_model(tau_low).fit(x2, y)
        self._upper_model = self._make_model(tau_high).fit(x2, y)
        return self

    def predict_interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)``, rearranged where the boosters cross."""
        x2 = as_2d(x)
        lower = self._lower_model.predict(x2)
        upper = self._upper_model.predict(x2)
        return rearrange(lower, upper)
