"""
logistic/tune.py — compatibility exports for threshold tuning.
"""

from DSSP2026.core.threshold import (
    ThresholdSweepResult,
    tune_threshold,
    tune_roc_threshold,
    _ALLOWED_METRICS,
    _MINIMIZE_METRICS,
)