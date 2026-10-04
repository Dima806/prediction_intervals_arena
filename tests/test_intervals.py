"""Every method produces a valid interval: right shape, finite, lower <= upper, width > 0."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.datasets.synthetic import make_synthetic
from src.intervals.bootstrap import BootstrapInterval
from src.intervals.conformal import AdaptiveConformal, SplitConformal
from src.intervals.gbm_quantile import GBMQuantileInterval
from src.intervals.normal import NormalInterval
from src.intervals.quantile_regression import QuantileRegressor


def _method(name: str) -> object:
    """Fresh, deliberately small/fast instances keyed by name."""
    fast_base = LinearRegression()
    builders = {
        "normal": lambda: NormalInterval(coverage=0.9),
        "quantile_regression": lambda: QuantileRegressor(n_iter=400),
        "gbm_quantile": lambda: GBMQuantileInterval(n_estimators=40),
        "bootstrap": lambda: BootstrapInterval(base_estimator=fast_base, n_bootstrap=20),
        "split_conformal": lambda: SplitConformal(coverage=0.9),
        "adaptive_conformal": lambda: AdaptiveConformal(coverage=0.9, n_bins=4, min_bin=30),
    }
    return builders[name]()


ALL_METHODS = [
    "normal",
    "quantile_regression",
    "gbm_quantile",
    "bootstrap",
    "split_conformal",
    "adaptive_conformal",
]


@pytest.mark.parametrize("name", ALL_METHODS)
@pytest.mark.parametrize("structure", ["homoscedastic", "heteroscedastic"])
def test_produces_valid_interval(name: str, structure: str) -> None:
    data = make_synthetic(structure, n_samples=1200, seed=0)
    x_train, x_test = data.x[:900], data.x[900:]
    y_train = data.y[:900]

    method = _method(name)
    method.fit(x_train, y_train)  # type: ignore[attr-defined]
    lower, upper = method.predict_interval(x_test)  # type: ignore[attr-defined]

    assert lower.shape == (x_test.shape[0],)
    assert upper.shape == (x_test.shape[0],)
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))
    assert np.all(upper >= lower), f"{name} produced lower > upper"
    assert np.mean(upper - lower) > 0.0


def test_quantile_regression_width_grows_with_input() -> None:
    """Quantile regression should widen in the volatile region -- the point of it."""
    data = make_synthetic("heteroscedastic", n_samples=4000, seed=1)
    method = QuantileRegressor(quantiles=(0.05, 0.95), n_iter=2000)
    method.fit(data.x, data.y)
    lower, upper = method.predict_interval(data.x)
    width = upper - lower
    x = data.x.ravel()
    low_region = width[x < 2.0].mean()
    high_region = width[x > 8.0].mean()
    assert high_region > low_region
