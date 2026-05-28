"""
core/encoders.py — feature transforms callable directly inside statsmodels formulas.

These are plain functions, not sklearn transformers, because the fitting layer
fits through the statsmodels formula interface (smf.ols / smf.logit), not
sklearn pipelines. They are injected into the formula evaluation namespace by
the fit layers, so you can write transforms inline:

    "OnTime ~ Distance + cyc_sin(DayOfWeek, 7) + cyc_cos(DayOfWeek, 7)"

Cyclical encoding maps a periodic integer (day of week, month, hour) onto a
circle via sine and cosine, so the model sees that period 7 is adjacent to
period 1 — which one-hot and ordinal encodings both miss.
"""

import numpy as np


def cyc_sin(x, period):
    """Sine component of a cyclical encoding for a periodic feature.

    Parameters
    ----------
    x : array-like
        Periodic values (e.g. DayOfWeek in 1..7, Month in 1..12).
    period : float
        The cycle length (7 for weekdays, 12 for months, 24 for hours).
    """
    return np.sin(2.0 * np.pi * np.asarray(x, dtype=float) / period)


def cyc_cos(x, period):
    """Cosine component of a cyclical encoding for a periodic feature."""
    return np.cos(2.0 * np.pi * np.asarray(x, dtype=float) / period)


# Mapping merged into the fit layers' formula evaluation namespace.
FORMULA_NAMESPACE = {
    "cyc_sin": cyc_sin,
    "cyc_cos": cyc_cos,
    "np": np,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "exp": np.exp,
}