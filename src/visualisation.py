"""Matplotlib helpers for the notebooks and app. Figures are returned, never shown.

These are presentation-only; all scoring lives in :mod:`src.evaluation`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_intervals(
    x: np.ndarray,
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str = "Prediction interval",
) -> Figure:
    """Scatter the data with the prediction band drawn over it."""
    x = np.asarray(x, dtype=float).ravel()
    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, s=8, alpha=0.3, color="#5B6B7B", label="observed")
    ax.fill_between(
        x[order],
        np.asarray(lower, dtype=float)[order],
        np.asarray(upper, dtype=float)[order],
        alpha=0.3,
        color="#2E7DB8",
        label="interval",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_coverage_bar(report: pd.DataFrame, nominal: float = 0.90) -> Figure:
    """Bar chart of actual coverage per method against the nominal line."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(report["method"], report["coverage"], color="#2E7DB8")
    ax.axhline(nominal, color="#C0392B", linestyle="--", label=f"nominal = {nominal:.0%}")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Does the interval cover as promised?")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_conditional_coverage(conditional: pd.DataFrame, nominal: float = 0.90) -> Figure:
    """Coverage by region: the plot that exposes too-wide/too-narrow miscalibration."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(conditional["center"], conditional["coverage"], marker="o", color="#2E7DB8")
    ax.axhline(nominal, color="#C0392B", linestyle="--", label=f"nominal = {nominal:.0%}")
    ax.set_xlabel("prediction region (bin center)")
    ax.set_ylabel("coverage in region")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Conditional coverage across the prediction range")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
