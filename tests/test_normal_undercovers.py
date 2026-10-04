"""The cautionary result, asserted: the textbook interval fails where real data lives.

On homoscedastic normal data the normal-theory interval is fine -- that is its home.
On heteroscedastic data its single constant width undercovers overall AND, more
tellingly, is far too narrow in the volatile region while far too wide in the calm one.
"""

from __future__ import annotations

import numpy as np

from src.datasets.synthetic import make_synthetic
from src.evaluation.conditional import conditional_coverage
from src.evaluation.coverage import coverage
from src.intervals.conformal import SplitConformal
from src.intervals.normal import NormalInterval

NOMINAL = 0.90


def _split(structure: str) -> tuple[np.ndarray, ...]:
    data = make_synthetic(structure, n_samples=6000, seed=42)
    return data.x[:4000], data.y[:4000], data.x[4000:], data.y[4000:]


def test_normal_interval_is_fine_on_homoscedastic_data() -> None:
    """Its home: near-nominal coverage and flat conditional coverage."""
    x_train, y_train, x_test, y_test = _split("homoscedastic")
    method = NormalInterval(coverage=NOMINAL).fit(x_train, y_train)
    lower, upper = method.predict_interval(x_test)

    cov = coverage(y_test, lower, upper)
    assert NOMINAL - 0.04 <= cov <= NOMINAL + 0.04, f"homoscedastic cov={cov:.3f}"

    table = conditional_coverage(y_test, lower, upper, by=x_test.ravel(), n_bins=5)
    spread = table["coverage"].max() - table["coverage"].min()
    assert spread < 0.08, f"conditional coverage should be flat, spread={spread:.3f}"


def test_normal_interval_undercovers_under_heteroscedasticity() -> None:
    """The showpiece failure, measured against known truth."""
    x_train, y_train, x_test, y_test = _split("heteroscedastic")
    x = x_test.ravel()

    normal = NormalInterval(coverage=NOMINAL).fit(x_train, y_train)
    n_lower, n_upper = normal.predict_interval(x_test)
    normal_cov = coverage(y_test, n_lower, n_upper)

    # 1. Overall coverage sits below nominal.
    assert normal_cov < NOMINAL - 0.01, f"expected undercoverage, got {normal_cov:.3f}"

    # 2. The conditional-coverage plot is the real tell: too narrow in the volatile
    #    region (badly undercovers) and too wide in the calm region (overcovers).
    table = conditional_coverage(y_test, n_lower, n_upper, by=x, n_bins=5)
    assert table["coverage"].min() < 0.75, f"volatile-region cov={table['coverage'].min():.3f}"
    assert table["coverage"].max() > 0.95, f"calm-region cov={table['coverage'].max():.3f}"

    # 3. Split conformal, same nominal level, does not undercover -- the contrast.
    conformal = SplitConformal(coverage=NOMINAL).fit(x_train, y_train)
    c_lower, c_upper = conformal.predict_interval(x_test)
    conformal_cov = coverage(y_test, c_lower, c_upper)
    assert normal_cov < conformal_cov
    assert conformal_cov >= NOMINAL - 0.05
