"""Synthetic regression data with a KNOWN error structure.

Coverage can only be validated when we control the truth, so these generators let
each assumption behind the textbook interval be switched on and off:

- ``homoscedastic``  : constant-variance Gaussian noise (the normal interval's home).
- ``heteroscedastic``: noise grows with the input, the realistic case that breaks a
  single constant-width interval.
- ``skewed``         : asymmetric (shifted-exponential) noise, where symmetric intervals misfit.
- ``heavy_tailed``   : Student-t noise, where normality fails.

The mean signal ``f(x)`` is identical across structures; only the noise changes, so a
difference in coverage is attributable to the error structure alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STRUCTURES: tuple[str, ...] = (
    "homoscedastic",
    "heteroscedastic",
    "skewed",
    "heavy_tailed",
)

_X_HIGH = 10.0


@dataclass
class Dataset:
    """A synthetic dataset with the true mean retained for diagnostics.

    Attributes:
        x: feature matrix, shape ``(n_samples, 1)``.
        y: observed response, shape ``(n_samples,)``.
        signal: the true conditional mean ``f(x)``, shape ``(n_samples,)``.
        structure: the error structure used to generate ``y``.
    """

    x: np.ndarray
    y: np.ndarray
    signal: np.ndarray
    structure: str


def _signal(x: np.ndarray) -> np.ndarray:
    """The true conditional mean: a gently curved trend."""
    return 2.0 * x + 10.0 + 2.0 * np.sin(x)


def _noise(
    structure: str,
    x: np.ndarray,
    rng: np.random.Generator,
    base_scale: float,
) -> np.ndarray:
    """Draw zero-mean noise with the requested structure (unit-ish scale * base_scale)."""
    n = x.shape[0]
    if structure == "homoscedastic":
        return rng.normal(0.0, base_scale, size=n)
    if structure == "heteroscedastic":
        # Scale climbs steeply with x: calm at the low end, volatile at the high end.
        scale = base_scale * (0.1 + 4.0 * (x / _X_HIGH))
        return rng.normal(0.0, scale)
    if structure == "skewed":
        # Shifted exponential: mean 0, variance base_scale**2, heavy right tail.
        return base_scale * (rng.standard_exponential(n) - 1.0)
    if structure == "heavy_tailed":
        # Student-t with 3 d.o.f., rescaled to variance base_scale**2.
        return base_scale * rng.standard_t(df=3, size=n) / np.sqrt(3.0)
    msg = f"unknown error structure: {structure!r}; choose one of {STRUCTURES}"
    raise ValueError(msg)


def make_synthetic(
    structure: str,
    n_samples: int,
    seed: int,
    base_scale: float = 2.0,
) -> Dataset:
    """Generate a synthetic regression dataset with a known error structure.

    Args:
        structure: one of :data:`STRUCTURES`.
        n_samples: number of rows.
        seed: seed for reproducibility.
        base_scale: base noise standard deviation.

    Returns:
        A :class:`Dataset` with ``x``, observed ``y``, and the true ``signal``.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, _X_HIGH, size=n_samples)
    signal = _signal(x1)
    y = signal + _noise(structure, x1, rng, base_scale)
    return Dataset(x=x1.reshape(-1, 1), y=y, signal=signal, structure=structure)
