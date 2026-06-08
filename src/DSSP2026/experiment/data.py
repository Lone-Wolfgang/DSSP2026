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
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _coerce_dtypes(
    df: pd.DataFrame, *, numeric_features: Sequence[str], flag_features: Sequence[str]
) -> pd.DataFrame:
    """Flags -> int, numeric -> float, non-finite numeric -> NaN. In place-ish."""
    for c in flag_features:
        df[c] = df[c].astype(int)
    for c in numeric_features:
        df[c] = df[c].astype(float)
    df[list(numeric_features)] = df[list(numeric_features)].replace([np.inf, -np.inf], np.nan)
    return df


def load_train_eval(
    *,
    use_test_file: bool = False,
    train_file: Path,
    test_file: Path,
    validation_ratio: float,
    random_state: int,
    target: str,
    numeric_features: Sequence[str],
    flag_features: Sequence[str],
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
    train_file = Path(train_file)
    test_file = Path(test_file)

    if use_test_file:
        train = _coerce_dtypes(
            pd.read_parquet(train_file),
            numeric_features=numeric_features,
            flag_features=flag_features)
        evaluation = _coerce_dtypes(
            pd.read_parquet(test_file),
            numeric_features=numeric_features,
            flag_features=flag_features)
        logger.info("Loaded train %d / test %d traces (evaluating on TEST file)",
                    len(train), len(evaluation))
        return train, evaluation, "test"

    # Default: carve a stratified validation slice from TRAIN only.
    from sklearn.model_selection import train_test_split

    full = _coerce_dtypes(
        pd.read_parquet(train_file),
        numeric_features=numeric_features,
        flag_features=flag_features)
    train, evaluation = train_test_split(
        full, test_size=validation_ratio, random_state=random_state,
        stratify=full[target])
    logger.info("Loaded train %d -> split into fit %d / validation %d "
                "(ratio=%.2f, seed=%d; TEST file untouched)",
                len(full), len(train), len(evaluation),
                validation_ratio, random_state)
    return train, evaluation, "validation"


def load_train_val_test(
    *,
    train_file: Path,
    test_file: Path,
    validation_ratio: float,
    seed: int,
    target: str,
    numeric_features: Sequence[str],
    flag_features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, test) — three disjoint preprocessed frames.

    Carves a stratified validation slice from the train file (ratio+seed from
    config) and loads the test file as the test frame. Applies the SAME
    preprocessing the existing loader applies (reuse load_train_eval internals;
    do not duplicate preprocessing logic).
    """
    train, validation, _ = load_train_eval(
        use_test_file=False,
        train_file=train_file,
        test_file=test_file,
        validation_ratio=validation_ratio,
        random_state=seed,
        target=target,
        numeric_features=numeric_features,
        flag_features=flag_features,
    )
    test_file = Path(test_file)
    test = _coerce_dtypes(
        pd.read_parquet(test_file),
        numeric_features=numeric_features,
        flag_features=flag_features)
    train, validation, test = impute_numeric(
        train, validation, test, numeric_features=numeric_features)
    logger.info("Loaded held-out test %d traces for final scoring", len(test))
    return train, validation, test


def impute_numeric(
    train: pd.DataFrame, *others: pd.DataFrame, numeric_features: Sequence[str]
):
    """Median-impute numeric features, fit on train only. Returns copies.

    For tree / forest / MLP-refit paths that can't take NaN. Scaling is not
    applied (trees are scale-invariant; the MLP pipeline scales internally).

    The median is fit on ``train`` and applied to ``train`` and every frame in
    ``others`` (validation, test, or any number of additional splits). Returns
    ``train`` followed by each transformed frame, in the order given — so the
    two-frame call ``train, ev = impute_numeric(train, ev)`` still unpacks
    cleanly, and ``train, val, test = impute_numeric(train, val, test)`` works
    the same way.
    """
    numeric_features = list(numeric_features)
    med = train[numeric_features].median()
    train = train.copy()
    train[numeric_features] = train[numeric_features].fillna(med)
    out = [train]
    for frame in others:
        frame = frame.copy()
        frame[numeric_features] = frame[numeric_features].fillna(med)
        out.append(frame)
    return tuple(out)


def standardise_numeric(
    train: pd.DataFrame, *others: pd.DataFrame, numeric_features: Sequence[str]
):
    """Median-impute then z-score numeric features, fit on train only. Copies.

    mnlogit is unregularised MLE and diverges to NaN coefficients on raw,
    near-separated predictors; standardising is the reliable fix. Flags (0/1)
    are left untouched.

    Both the imputation median and the z-score mean/std are fit on ``train``
    only, then applied to ``train`` and every frame in ``others`` (validation,
    test, or any number of additional splits). Returns ``train`` followed by
    each transformed frame, in the order given — so ``train, ev = ...`` and
    ``train, val, test = ...`` both unpack correctly.
    """
    numeric_features = list(numeric_features)
    train, *others = impute_numeric(train, *others, numeric_features=numeric_features)
    mu = train[numeric_features].mean()
    sd = train[numeric_features].std().replace(0, 1.0)   # guard constant cols
    train[numeric_features] = (train[numeric_features] - mu) / sd
    out = [train]
    for frame in others:
        frame[numeric_features] = (frame[numeric_features] - mu) / sd
        out.append(frame)
    return tuple(out)
