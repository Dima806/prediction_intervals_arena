"""Persist notebook outputs: figures as PNG, text/numbers as JSON.

Every result a notebook produces that a reader might cite is
written under ``outputs/`` so it is reproducible and quotable without re-running:
figures to ``outputs/figures/*.png`` and structured results to ``outputs/data/*.json``.
Paths resolve relative to the project root, so saving works regardless of the kernel's
working directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DATA_DIR = PROJECT_ROOT / "outputs" / "data"


def _json_default(value: Any) -> Any:
    """Make numpy scalars and arrays JSON-serialisable."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    msg = f"cannot serialise {type(value)!r} to JSON"
    raise TypeError(msg)


def save_fig(fig: Figure, name: str, dpi: int = 150) -> Path:
    """Write ``fig`` to ``outputs/figures/<name>.png`` and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_json(name: str, payload: Any) -> Path:
    """Write ``payload`` to ``outputs/data/<name>.json`` and return the path.

    A pandas DataFrame is stored as a list of row records; everything else is dumped
    as-is with numpy types coerced to native Python.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "columns") and hasattr(payload, "to_dict"):
        payload = payload.to_dict(orient="records")
    path = DATA_DIR / (name if name.endswith(".json") else f"{name}.json")
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return path
