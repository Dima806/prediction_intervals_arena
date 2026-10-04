"""The payoff, asserted: conformal hits nominal coverage on any structure, any model.

Split conformal should land near the nominal rate regardless of the error structure,
and should do so when simply WRAPPED around an already-fitted model. Adaptive
(Mondrian) conformal should keep that coverage while letting the width vary by region.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from src.datasets.synthetic import STRUCTURES, make_synthetic
from src.evaluation.conditional import conditional_coverage
from src.evaluation.coverage import coverage
from src.intervals.conformal import AdaptiveConformal, SplitConformal

NOMINAL = 0.90
TOL = 0.05  # finite-sample slack; conformal may over-cover but must not undercover much


def _split(structure: str, seed: int = 42) -> tuple[np.ndarray, ...]:
    data = make_synthetic(structure, n_samples=6000, seed=seed)
    return data.x[:4000], data.y[:4000], data.x[4000:], data.y[4000:]


@pytest.mark.parametrize("structure", STRUCTURES)
def test_split_conformal_hits_nominal(structure: str) -> None:
    x_train, y_train, x_test, y_test = _split(structure)
    method = SplitConformal(coverage=NOMINAL).fit(x_train, y_train)
    lower, upper = method.predict_interval(x_test)
    cov = coverage(y_test, lower, upper)
    assert NOMINAL - TOL <= cov <= NOMINAL + TOL, f"{structure}: split conformal cov={cov:.3f}"


def test_split_conformal_wraps_a_prefit_model_without_retraining() -> None:
    data = make_synthetic("heteroscedastic", n_samples=6000, seed=7)
    x_train, y_train = data.x[:3000], data.y[:3000]
    x_cal, y_cal = data.x[3000:4500], data.y[3000:4500]
    x_test, y_test = data.x[4500:], data.y[4500:]

    model = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
    wrapped = SplitConformal(base_estimator=model, coverage=NOMINAL)
    wrapped.fit(x_cal, y_cal, prefit=True)

    # The exact same fitted object is reused -- no retraining happened.
    assert wrapped._model is model
    lower, upper = wrapped.predict_interval(x_test)
    cov = coverage(y_test, lower, upper)
    assert NOMINAL - TOL <= cov <= NOMINAL + TOL, f"wrapped cov={cov:.3f}"


@pytest.mark.parametrize("structure", STRUCTURES)
def test_adaptive_conformal_keeps_coverage(structure: str) -> None:
    x_train, y_train, x_test, y_test = _split(structure)
    method = AdaptiveConformal(coverage=NOMINAL, n_bins=5, min_bin=50).fit(x_train, y_train)
    lower, upper = method.predict_interval(x_test)
    cov = coverage(y_test, lower, upper)
    # Over-coverage is acceptable (still has the guarantee); undercoverage is not.
    assert NOMINAL - TOL <= cov <= NOMINAL + 0.06, f"{structure}: adaptive cov={cov:.3f}"


def test_adaptive_conformal_varies_width_and_fixes_conditional_coverage() -> None:
    """On heteroscedastic data the width must widen with the volatile region."""
    x_train, y_train, x_test, y_test = _split("heteroscedastic")
    method = AdaptiveConformal(coverage=NOMINAL, n_bins=5, min_bin=50).fit(x_train, y_train)
    lower, upper = method.predict_interval(x_test)

    width = upper - lower
    assert np.unique(np.round(width, 6)).size >= 2, "adaptive conformal width did not vary"

    x = x_test.ravel()
    assert width[x > 8.0].mean() > width[x < 2.0].mean(), (
        "width should grow in the volatile region"
    )

    # Unlike the normal interval (which collapses to ~0.66 here), conditional coverage holds.
    table = conditional_coverage(y_test, lower, upper, by=x, n_bins=5)
    assert table["coverage"].min() > 0.80, f"worst-region coverage={table['coverage'].min():.3f}"
