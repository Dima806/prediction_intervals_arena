"""Quantile regression from scratch via the pinball (quantile) loss.

Instead of predicting the mean and bolting on a symmetric band, predict the lower and
upper quantiles directly. Because each bound is its own fit, the interval WIDTH can
vary with the input -- the thing the normal interval cannot do.

The fit minimises a smoothed pinball loss with Adam on polynomial features (a compact,
dependency-free "from scratch" implementation). Honest limitation: this retrains the
model to target quantiles and still carries no coverage guarantee -- which is what
motivates conformal prediction.
"""

from __future__ import annotations

import numpy as np

from .base import as_2d, rearrange


def _pinball_grad_weights(residual: np.ndarray, tau: float, smooth: float) -> np.ndarray:
    """d(loss)/d(prediction) for the pinball loss, with the kink smoothed by a sigmoid."""
    # 1{residual < 0} smoothed; as smooth -> 0 this recovers the exact subgradient.
    below = 1.0 / (1.0 + np.exp(residual / smooth))
    return -(tau - below)


class QuantileRegressor:
    """Linear-in-polynomial-features quantile regression trained with the pinball loss."""

    def __init__(
        self,
        quantiles: tuple[float, float] = (0.05, 0.95),
        degree: int = 2,
        learning_rate: float = 0.05,
        n_iter: int = 2000,
        smooth: float = 0.1,
    ) -> None:
        self.quantiles = quantiles
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.smooth = smooth
        self._x_mean: np.ndarray = np.zeros(1)
        self._x_std: np.ndarray = np.ones(1)
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._coef: dict[float, np.ndarray] = {}

    def _features(self, x: np.ndarray) -> np.ndarray:
        """Standardised polynomial features with a leading intercept column."""
        x2 = as_2d(x)
        cols = [np.ones((x2.shape[0], 1))]
        for d in range(1, self.degree + 1):
            standardised = (x2**d - self._x_mean[d - 1]) / self._x_std[d - 1]
            cols.append(standardised)
        return np.hstack(cols)

    def _fit_quantile(self, features: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
        """Adam on the smoothed pinball loss for a single quantile."""
        n, p = features.shape
        beta = np.zeros(p)
        m = np.zeros(p)
        v = np.zeros(p)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.n_iter + 1):
            residual = y - features @ beta
            grad = features.T @ _pinball_grad_weights(residual, tau, self.smooth) / n
            m = b1 * m + (1.0 - b1) * grad
            v = b2 * v + (1.0 - b2) * grad**2
            m_hat = m / (1.0 - b1**t)
            v_hat = v / (1.0 - b2**t)
            beta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        return beta

    def fit(self, x: np.ndarray, y: np.ndarray) -> QuantileRegressor:
        """Fit both quantile bounds on standardised features and targets."""
        x2 = as_2d(x)
        y = np.asarray(y, dtype=float)
        powers = np.hstack([x2**d for d in range(1, self.degree + 1)])
        self._x_mean = powers.mean(axis=0)
        self._x_std = powers.std(axis=0) + 1e-12
        self._y_mean = float(y.mean())
        self._y_std = float(y.std() + 1e-12)

        features = self._features(x2)
        y_std = (y - self._y_mean) / self._y_std
        self._coef = {tau: self._fit_quantile(features, y_std, tau) for tau in self.quantiles}
        return self

    def predict_quantile(self, x: np.ndarray, tau: float) -> np.ndarray:
        """Predict a single quantile, mapped back to the original target scale."""
        return self._features(x) @ self._coef[tau] * self._y_std + self._y_mean

    def predict_interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` from the two fitted quantiles, rearranged if crossed."""
        tau_low, tau_high = self.quantiles
        lower = self.predict_quantile(x, tau_low)
        upper = self.predict_quantile(x, tau_high)
        return rearrange(lower, upper)
