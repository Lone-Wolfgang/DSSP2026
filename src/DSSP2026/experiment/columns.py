"""
experiment/columns.py — data-agnostic column-type resolution.

The framework core describes *data + task* (a dataframe, a target, optional named
feature sets). The one model that needs more — the MLP, whose preprocessor must
know which columns to scale vs pass through — gets that from a column-type map,
which is either supplied by the user or **inferred from dtype** so the common
case is zero-config.

Types:
  "numeric"      -> median-impute + standardize (continuous)
  "categorical"  -> passthrough today (already 0/1 flags or encoded); a one-hot
                    step can hook in here later without touching the core
  "passthrough"  -> use as-is

Only the MLP study consumes this; tree / RF / logistic / XGBoost ignore it.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import pandas as pd
from pandas.api import types as pdt

VALID_TYPES = ("numeric", "categorical", "passthrough")


def infer_column_types(df: pd.DataFrame, columns: Sequence[str],
                       *, max_int_cardinality: int = 12) -> dict:
    """Infer a column-type map from dtypes.

    - float dtype                       -> numeric
    - bool dtype                        -> categorical (a flag)
    - integer dtype with few distinct   -> categorical (likely an encoded label)
    - integer dtype with many distinct  -> numeric (a count/continuous measure)
    - object/category dtype             -> categorical
    Anything unrecognised defaults to numeric (the safe scale-it choice).

    ``max_int_cardinality`` is the cutoff below which an integer column is
    treated as categorical rather than numeric — the one heuristic worth an
    override, which the user-supplied map provides.
    """
    out = {}
    for c in columns:
        s = df[c]
        if pdt.is_bool_dtype(s):
            out[c] = "categorical"
        elif pdt.is_float_dtype(s):
            out[c] = "numeric"
        elif pdt.is_integer_dtype(s):
            out[c] = ("categorical" if s.nunique(dropna=True) <= max_int_cardinality
                      else "numeric")
        elif pdt.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            out[c] = "categorical"
        else:
            out[c] = "numeric"
    return out


def resolve_column_types(df: pd.DataFrame, columns: Sequence[str],
                         column_types: Optional[Mapping[str, str]] = None) -> dict:
    """Return a complete {col: type} map for ``columns``.

    Starts from inference, then overlays any user-supplied entries (so a user
    only needs to specify the columns inference would get wrong). Validates the
    supplied types, and that every referenced column actually exists in ``df``.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"feature_sets reference {len(missing)} column(s) not present in the "
            f"data: {missing}. Check that the dataframe matches the configured "
            f"feature sets (a column may have been renamed or dropped).")
    resolved = infer_column_types(df, columns)
    if column_types:
        for c, t in column_types.items():
            if c not in columns:
                continue
            if t not in VALID_TYPES:
                raise ValueError(
                    f"column_types[{c!r}]={t!r} invalid; use one of {VALID_TYPES}.")
            resolved[c] = t
    return resolved


def numeric_and_flag(column_types: Mapping[str, str]):
    """Split a column-type map into the (numeric, flag) lists the MLP wants.

    The MLP preprocessor scales ``numeric`` columns and passes the rest through,
    so "categorical" and "passthrough" both map to the flag (pass-through) list.
    This adapter is the only place the framework speaks the MLP's old
    numeric/flag vocabulary.
    """
    numeric = [c for c, t in column_types.items() if t == "numeric"]
    flag = [c for c, t in column_types.items() if t != "numeric"]
    return numeric, flag