"""Conformal prediction: guaranteed coverage around any model, built from scratch.

Split conformal: hold out a calibration set, measure the model's absolute errors on it,
and take the appropriate empirical quantile as the interval half-width. Under a mild
exchangeability assumption this covers at the nominal rate -- whatever the model,
whatever the data -- and needs no retraining of the underlying model.

Adaptive (Mondrian) conformal: compute that quantile SEPARATELY within regions of the
prediction, so the width varies by region. This fixes plain conformal's one weakness
(a single constant width) while keeping the guarantee.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from .base import as_2d, conformal_level


class SplitConformal:
    """Split-conformal interval: a constant half-width from calibration-set errors."""

    def __init__(
        self,
        base_estimator: Any = None,
        coverage: float = 0.90,
        calibration_fraction: float = 0.25,
        random_state: int = 0,
    ) -> None:
        self.base_estimator = base_estimator
        self.coverage = coverage
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state
        self._model: Any = None
        self._q: float = 0.0

    def _new_model(self) -> Any:
        if self.base_estimator is not None:
            return clone(self.base_estimator)
        return GradientBoostingRegressor(random_state=0)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        prefit: bool = False,
    ) -> SplitConformal:
        """Fit (unless ``prefit``) and calibrate.

        Args:
            x, y: training data. When ``prefit`` is False the data is split into a
                proper-training part (to fit the model) and a calibration part.
            prefit: if True, ``base_estimator`` is already fitted and ALL of ``(x, y)``
                is used only as the calibration set -- the "wrap my model, no retraining"
                path. The guarantee is unchanged.
        """
        x2 = as_2d(x)
        y = np.asarray(y, dtype=float)
        if prefit:
            if self.base_estimator is None:
                msg = "prefit=True requires an already-fitted base_estimator"
                raise ValueError(msg)
            self._model = self.base_estimator
            x_cal, y_cal = x2, y
        else:
            x_proper, x_cal, y_proper, y_cal = train_test_split(
                x2, y, test_size=self.calibration_fraction, random_state=self.random_state
            )
            self._model = self._new_model()
            self._model.fit(x_proper, y_proper)

        scores = np.abs(y_cal - self._model.predict(x_cal))
        level = conformal_level(len(scores), self.coverage)
        self._q = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``prediction +/- q``, the calibrated constant-width interval."""
        prediction = self._model.predict(as_2d(x))
        return prediction - self._q, prediction + self._q


class AdaptiveConformal:
    """Mondrian conformal: a separate conformal half-width per prediction region."""

    def __init__(
        self,
        base_estimator: Any = None,
        coverage: float = 0.90,
        calibration_fraction: float = 0.25,
        n_bins: int = 5,
        min_bin: int = 50,
        random_state: int = 0,
    ) -> None:
        self.base_estimator = base_estimator
        self.coverage = coverage
        self.calibration_fraction = calibration_fraction
        self.n_bins = n_bins
        self.min_bin = min_bin
        self.random_state = random_state
        self._model: Any = None
        self._edges: np.ndarray = np.zeros(0)
        self._bin_q: np.ndarray = np.zeros(0)
        self._global_q: float = 0.0

    def _new_model(self) -> Any:
        if self.base_estimator is not None:
            return clone(self.base_estimator)
        return GradientBoostingRegressor(random_state=0)

    def _bin_index(self, prediction: np.ndarray) -> np.ndarray:
        return np.clip(np.digitize(prediction, self._edges[1:-1]), 0, self.n_bins - 1)

    def fit(self, x: np.ndarray, y: np.ndarray, prefit: bool = False) -> AdaptiveConformal:
        """Fit (unless ``prefit``) and calibrate one conformal quantile per region."""
        x2 = as_2d(x)
        y = np.asarray(y, dtype=float)
        if prefit:
            if self.base_estimator is None:
                msg = "prefit=True requires an already-fitted base_estimator"
                raise ValueError(msg)
            self._model = self.base_estimator
            x_cal, y_cal = x2, y
        else:
            x_proper, x_cal, y_proper, y_cal = train_test_split(
                x2, y, test_size=self.calibration_fraction, random_state=self.random_state
            )
            self._model = self._new_model()
            self._model.fit(x_proper, y_proper)

        pred_cal = self._model.predict(x_cal)
        scores = np.abs(y_cal - pred_cal)
        self._global_q = float(
            np.quantile(scores, conformal_level(len(scores), self.coverage), method="higher")
        )

        # Region boundaries are quantiles of the calibration predictions.
        self._edges = np.quantile(pred_cal, np.linspace(0.0, 1.0, self.n_bins + 1))
        idx = self._bin_index(pred_cal)
        bin_q = np.empty(self.n_bins)
        for b in range(self.n_bins):
            mask = idx == b
            if mask.sum() >= self.min_bin:
                level = conformal_level(int(mask.sum()), self.coverage)
                bin_q[b] = np.quantile(scores[mask], level, method="higher")
            else:
                # Too few points to calibrate the region: fall back to the global width.
                bin_q[b] = self._global_q
        self._bin_q = bin_q
        return self

    def predict_interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return a region-varying interval: ``prediction +/- q(region)``."""
        prediction = self._model.predict(as_2d(x))
        half_width = self._bin_q[self._bin_index(prediction)]
        return prediction - half_width, prediction + half_width
