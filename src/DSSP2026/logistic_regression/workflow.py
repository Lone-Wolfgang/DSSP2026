"""
logistic/workflow.py — internal-CV decision-threshold tuning for logistic regression.

The model *formula is fixed*; what we tune is the decision threshold. This runs
stratified k-fold cross-validation, and within each fold sweeps the probability
cutoff over a **shared fixed grid** (so fold curves are aligned and can be
averaged), then aggregates to mean ± std across folds.

Why a shared grid: ``tune.tune_threshold`` defaults to each fold's own ROC
thresholds, which differ fold to fold — you can't average "metric at threshold
t" across folds if every fold has a different set of t's. Evaluating all folds
on one grid fixes that. The per-fold *best* threshold is still recorded and
averaged separately, since that's a genuine fold-level quantity.

What it produces (primary output: mean ± std across folds)
----------------------------------------------------------
- ``summary_df``     : per-threshold mean ± std of each metric across folds
- ``roc``            : per-fold and mean ROC curve points (for the ROC plot)
- ``fold_best``      : each fold's best threshold + the metric it optimized
- a one-row headline (mean best threshold, mean metric, mean AUC)

Everything routes through ``core``: metrics from ``core.metrics``, plots on the
seaborn-Objects engine (``core.viz``), figures saved via ``core.figure``, and
the summary table via ``core.tables`` so it prints in a notebook *and* saves to
csv / xlsx / html / png from one call.

Fitting itself lives in ``logistic.fit``; threshold sweeping in ``logistic.tune``.
"""

from typing import Optional

import numpy as np
import pandas as pd

from DSSP2026.core.threshold import (
    CVThresholdResult,
)
from DSSP2026.core.threshold_viz import (
    _DEFAULT_PLOT_METRICS,
    att_palette_for,
    make_cv_summary_df,
    plot_cv_roc,
    plot_cv_threshold_metrics,
    save_cv_roc,
    save_cv_summary,
    save_cv_threshold_metrics,
    style_cv_summary,
)

from DSSP2026.logistic_regression.binary import fit_logit, predict_proba, get_endog