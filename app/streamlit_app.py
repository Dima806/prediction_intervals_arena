"""Prediction Intervals Arena -- interactive companion to the notebooks.

Four panels:
  1. Interval viewer   -- pick an error structure and method, see the band.
  2. Coverage meter    -- actual vs the promised 90% on held-out data.
  3. Assumption breaker-- crank heteroscedasticity, watch the normal interval collapse
                          while conformal holds.
  4. Wrap-my-model     -- split conformal wrapped around a pre-fit model, no retraining.

Run with ``make run`` (``uv run streamlit run app/streamlit_app.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from src.datasets.synthetic import STRUCTURES, make_synthetic
from src.evaluation.coverage import coverage
from src.evaluation.width import mean_width
from src.intervals.bootstrap import BootstrapInterval
from src.intervals.conformal import AdaptiveConformal, SplitConformal
from src.intervals.gbm_quantile import GBMQuantileInterval
from src.intervals.normal import NormalInterval
from src.intervals.quantile_regression import QuantileRegressor
from src.visualisation import plot_coverage_bar, plot_intervals

NOMINAL = 0.90
SEED = 42


def _build_method(name: str) -> object:
    """Return a fresh interval method by display name."""
    factories = {
        "Normal (textbook)": lambda: NormalInterval(coverage=NOMINAL),
        "Quantile regression": lambda: QuantileRegressor(quantiles=(0.05, 0.95)),
        "GBM quantiles": lambda: GBMQuantileInterval(quantiles=(0.05, 0.95)),
        "Bootstrap": lambda: BootstrapInterval(n_bootstrap=60, coverage=NOMINAL),
        "Split conformal": lambda: SplitConformal(coverage=NOMINAL),
        "Adaptive conformal": lambda: AdaptiveConformal(coverage=NOMINAL),
    }
    return factories[name]()


@st.cache_data
def _dataset(structure: str, n_samples: int, base_scale: float) -> tuple[np.ndarray, ...]:
    data = make_synthetic(structure, n_samples=n_samples, seed=SEED, base_scale=base_scale)
    x_train, x_test, y_train, y_test = train_test_split(
        data.x, data.y, test_size=0.3, random_state=SEED
    )
    return x_train, x_test, y_train, y_test


def _fit_predict(name: str, structure: str, base_scale: float) -> tuple[np.ndarray, ...]:
    x_train, x_test, y_train, y_test = _dataset(structure, 4000, base_scale)
    method = _build_method(name)
    method.fit(x_train, y_train)  # type: ignore[attr-defined]
    lower, upper = method.predict_interval(x_test)  # type: ignore[attr-defined]
    return x_test, y_test, lower, upper


def main() -> None:
    st.set_page_config(page_title="Prediction Intervals Arena", layout="wide")
    st.title("Your model gives you a number. It should give you a range.")
    st.caption("Coverage is the metric: does the 90% interval contain the truth 90% of the time?")

    tab_view, tab_meter, tab_break, tab_wrap = st.tabs(
        ["Interval viewer", "Coverage meter", "Assumption breaker", "Wrap my model"]
    )

    method_names = [
        "Normal (textbook)",
        "Quantile regression",
        "GBM quantiles",
        "Bootstrap",
        "Split conformal",
        "Adaptive conformal",
    ]

    with tab_view:
        col_a, col_b = st.columns(2)
        structure = col_a.selectbox("Error structure", STRUCTURES, key="view_structure")
        name = col_b.selectbox("Method", method_names, key="view_method")
        x_test, y_test, lower, upper = _fit_predict(name, structure, base_scale=2.0)
        st.pyplot(plot_intervals(x_test, y_test, lower, upper, title=f"{name} on {structure}"))
        st.metric("Coverage", f"{coverage(y_test, lower, upper):.1%}", f"nominal {NOMINAL:.0%}")

    with tab_meter:
        structure = st.selectbox("Error structure", STRUCTURES, key="meter_structure")
        rows = []
        for name in method_names:
            x_test, y_test, lower, upper = _fit_predict(name, structure, base_scale=2.0)
            rows.append(
                {
                    "method": name,
                    "coverage": coverage(y_test, lower, upper),
                    "mean_width": mean_width(lower, upper),
                }
            )
        report = pd.DataFrame(rows)
        st.pyplot(plot_coverage_bar(report, nominal=NOMINAL))
        st.dataframe(report, use_container_width=True)

    with tab_break:
        st.write(
            "Crank the noise scale on heteroscedastic data and compare the textbook "
            "normal interval with split conformal."
        )
        scale = st.slider("Base noise scale", 0.5, 6.0, 2.0, 0.5)
        col_normal, col_conformal = st.columns(2)
        for label, column in (
            ("Normal (textbook)", col_normal),
            ("Split conformal", col_conformal),
        ):
            x_test, y_test, lower, upper = _fit_predict(label, "heteroscedastic", base_scale=scale)
            column.metric(label, f"{coverage(y_test, lower, upper):.1%}", f"nominal {NOMINAL:.0%}")
            column.pyplot(plot_intervals(x_test, y_test, lower, upper, title=label))

    with tab_wrap:
        st.write(
            "Train a plain gradient booster, then wrap split conformal around it with no "
            "retraining -- the calibration set alone delivers the guarantee."
        )
        x_train, x_test, y_train, y_test = _dataset("heteroscedastic", 4000, 2.0)
        model = GradientBoostingRegressor(random_state=SEED).fit(x_train, y_train)
        wrapped = SplitConformal(base_estimator=model, coverage=NOMINAL)
        wrapped.fit(x_test[: len(x_test) // 2], y_test[: len(y_test) // 2], prefit=True)
        x_eval, y_eval = x_test[len(x_test) // 2 :], y_test[len(y_test) // 2 :]
        lower, upper = wrapped.predict_interval(x_eval)
        st.metric(
            "Wrapped coverage", f"{coverage(y_eval, lower, upper):.1%}", f"nominal {NOMINAL:.0%}"
        )
        st.pyplot(plot_intervals(x_eval, y_eval, lower, upper, title="Split conformal wrapped"))


if __name__ == "__main__":
    main()
