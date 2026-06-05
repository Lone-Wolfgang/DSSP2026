"""
workflows/data.py — data loading and the train / evaluation split.

Handles CLI decision #3 (which split):

- ``use_test_file=False`` (default): load TRAIN only, carve a stratified
  validation slice from it (ratio + seed from config). The "evaluation" frame is
  that validation slice — the real test file is never touched.
- ``use_test_file=True``: load TRAIN and TEST; the evaluation frame is TEST.

Either way the rest of the pipeline receives a uniform ``(train, evaluation)``
pair, so every model family is fit on ``train`` and scored once on
``evaluation``. Numeric NaNs are left as NaN at load time; each family applies
its own numeric handling (``impute_numeric`` for tree/forest/MLP-refit,
``standardise_numeric`` for logistic) on copies.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from DSSP2026.workflows import config as C

logger = logging.getLogger(__name__)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Flags -> int, numeric -> float, non-finite numeric -> NaN. In place-ish."""
    for c in C.FLAG_FEATURES:
        df[c] = df[c].astype(int)
    for c in C.NUMERIC_FEATURES:
        df[c] = df[c].astype(float)
    df[C.NUMERIC_FEATURES] = df[C.NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    return df


def load_train_eval(
    *,
    use_test_file: bool = False,
    train_file: Path = None,
    test_file: Path = None,
    validation_ratio: float = None,
    random_state: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load data and return ``(train, evaluation, eval_kind)``.

    Parameters
    ----------
    use_test_file : bool
        False (default) -> carve a validation slice from TRAIN. True -> use the
        separate TEST file as the evaluation set.
    train_file, test_file : Path, optional
        Override the config paths.
    validation_ratio : float, optional
        Fraction of TRAIN held out as validation in the default mode. Defaults
        to ``config.VALIDATION_RATIO``.
    random_state : int, optional
        Seed for the validation split. Defaults to ``config.RANDOM_STATE``.

    Returns
    -------
    train, evaluation : DataFrame
    eval_kind : str
        "validation" or "test" — used in titles/labels so output is unambiguous.
    """
    train_file = Path(train_file or C.TRAIN_FILE)
    test_file = Path(test_file or C.TEST_FILE)
    validation_ratio = (validation_ratio if validation_ratio is not None
                        else C.VALIDATION_RATIO)
    random_state = random_state if random_state is not None else C.RANDOM_STATE

    if use_test_file:
        train = _coerce_dtypes(pd.read_parquet(train_file))
        evaluation = _coerce_dtypes(pd.read_parquet(test_file))
        logger.info("Loaded train %d / test %d traces (evaluating on TEST file)",
                    len(train), len(evaluation))
        return train, evaluation, "test"

    # Default: carve a stratified validation slice from TRAIN only.
    from sklearn.model_selection import train_test_split

    full = _coerce_dtypes(pd.read_parquet(train_file))
    train, evaluation = train_test_split(
        full, test_size=validation_ratio, random_state=random_state,
        stratify=full[C.TARGET])
    logger.info("Loaded train %d -> split into fit %d / validation %d "
                "(ratio=%.2f, seed=%d; TEST file untouched)",
                len(full), len(train), len(evaluation),
                validation_ratio, random_state)
    return train, evaluation, "validation"


def impute_numeric(train: pd.DataFrame, evaluation: pd.DataFrame):
    """Median-impute numeric features, fit on train only. Returns copies.

    For tree / forest / MLP-refit paths that can't take NaN. Scaling is not
    applied (trees are scale-invariant; the MLP pipeline scales internally).
    """
    med = train[C.NUMERIC_FEATURES].median()
    train = train.copy()
    evaluation = evaluation.copy()
    train[C.NUMERIC_FEATURES] = train[C.NUMERIC_FEATURES].fillna(med)
    evaluation[C.NUMERIC_FEATURES] = evaluation[C.NUMERIC_FEATURES].fillna(med)
    return train, evaluation


def standardise_numeric(train: pd.DataFrame, evaluation: pd.DataFrame):
    """Median-impute then z-score numeric features, fit on train only. Copies.

    mnlogit is unregularised MLE and diverges to NaN coefficients on raw,
    near-separated predictors; standardising is the reliable fix. Flags (0/1)
    are left untouched.
    """
    train, evaluation = impute_numeric(train, evaluation)
    mu = train[C.NUMERIC_FEATURES].mean()
    sd = train[C.NUMERIC_FEATURES].std().replace(0, 1.0)   # guard constant cols
    train[C.NUMERIC_FEATURES] = (train[C.NUMERIC_FEATURES] - mu) / sd
    evaluation[C.NUMERIC_FEATURES] = (evaluation[C.NUMERIC_FEATURES] - mu) / sd
    return train, evaluation
