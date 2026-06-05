"""
experiment/spaces.py — spec-driven hyperparameter search spaces.

Each model's search space is a JSON-serializable **spec** (a list of parameter
descriptors), not hardcoded ``trial.suggest_*`` calls. One interpreter,
``suggest_from_spec``, turns a spec into the sampled params dict. The same spec
object both drives the Optuna objective AND serializes into an experiment's
config.json — so "the search space" is a record you can read, edit, and re-feed.

Param descriptor types
----------------------
- ``{"name", "type": "int",   "low", "high", "step"?, "log"?}``
- ``{"name", "type": "float", "low", "high", "log"?}``
- ``{"name", "type": "categorical", "choices": [...]}``
- ``{"type": "group", "count", "item", "collect"}`` — a CONDITIONAL block: sample
  the ``count`` param (an int descriptor), then sample ``item`` that many times,
  naming them ``f"{item['name']}{i}"`` (i=1..n), and collect the values into a
  tuple under the key ``collect``. This expresses MLP's n_layers -> per-layer
  widths without flattening.

``default_space(model)`` returns the built-in spec; callers may override and
whatever is actually used is recorded. ``suggest_params(trial, model, spec=None)``
samples a full params dict for a model (feature_set is sampled separately by the
objective).
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence


def suggest_feature_set(trial, feature_set_names: Sequence[str]) -> str:
    """Sample which named feature set to use (shared first step for all models)."""
    return trial.suggest_categorical("feature_set", list(feature_set_names))


# ---------------------------------------------------------------------------
# Default specs (JSON-serializable). Edit a copy and pass it back to override.
# ---------------------------------------------------------------------------

_DEFAULT_SPACES = {
    "Logistic regression": [],   # nothing to tune; one config per feature set
    "Decision tree": [
        {"name": "max_depth", "type": "int", "low": 1, "high": 19},
    ],
    "Random forest": [
        {"name": "n_estimators", "type": "categorical",
         "choices": [50, 100, 200, 300]},
        {"name": "max_features", "type": "categorical",
         "choices": [2, 3, 5, "sqrt", "log2"]},
    ],
    "MLP": [
        {"type": "group", "collect": "hidden",
         "count": {"name": "n_layers", "type": "int", "low": 1, "high": 4},
         "item": {"name": "width", "type": "int", "low": 16, "high": 512,
                  "log": True}},
        {"name": "activation", "type": "categorical",
         "choices": ["relu", "tanh"]},
        {"name": "alpha", "type": "float", "low": 1e-6, "high": 1e1, "log": True},
        {"name": "lr_init", "type": "float", "low": 1e-4, "high": 1e-1,
         "log": True},
    ],
    "XGBoost": [
        {"name": "n_estimators", "type": "int", "low": 100, "high": 600,
         "step": 50},
        {"name": "max_depth", "type": "int", "low": 2, "high": 10},
        {"name": "learning_rate", "type": "float", "low": 1e-3, "high": 3e-1,
         "log": True},
        {"name": "subsample", "type": "float", "low": 0.5, "high": 1.0},
        {"name": "colsample_bytree", "type": "float", "low": 0.5, "high": 1.0},
        {"name": "reg_lambda", "type": "float", "low": 1e-3, "high": 1e1,
         "log": True},
        {"name": "reg_alpha", "type": "float", "low": 1e-3, "high": 1e1,
         "log": True},
        {"name": "min_child_weight", "type": "float", "low": 1.0, "high": 10.0},
    ],
}


def default_space(model: str) -> list:
    """Return a deep copy of the built-in spec for ``model`` (safe to edit)."""
    if model not in _DEFAULT_SPACES:
        raise ValueError(f"no default space for model {model!r}.")
    return copy.deepcopy(_DEFAULT_SPACES[model])


def default_spaces() -> dict:
    """All built-in specs as {model: spec} (deep-copied)."""
    return {m: copy.deepcopy(s) for m, s in _DEFAULT_SPACES.items()}


# ---------------------------------------------------------------------------
# Interpreter: spec -> sampled params (the single place specs become suggests)
# ---------------------------------------------------------------------------

def _suggest_one(trial, d: dict):
    """Sample a single flat descriptor; return (name, value)."""
    t = d["type"]
    name = d["name"]
    if t == "int":
        return name, trial.suggest_int(
            name, d["low"], d["high"], step=d.get("step", 1),
            log=bool(d.get("log", False)))
    if t == "float":
        return name, trial.suggest_float(
            name, d["low"], d["high"], log=bool(d.get("log", False)))
    if t == "categorical":
        return name, trial.suggest_categorical(name, list(d["choices"]))
    raise ValueError(f"unknown param type {t!r} in spec descriptor {d!r}.")


def suggest_from_spec(trial, spec: list) -> dict:
    """Interpret a spec into a sampled params dict.

    Flat descriptors map to a single suggested value under their ``name``. A
    ``group`` descriptor samples its ``count`` int, then its ``item`` that many
    times (named ``item_name`` + index), collecting the values into a tuple under
    the group's ``collect`` key — this is MLP's conditional layer block.
    """
    params = {}
    for d in spec:
        if d.get("type") == "group":
            cnt_d = d["count"]
            n = trial.suggest_int(
                cnt_d["name"], cnt_d["low"], cnt_d["high"],
                step=cnt_d.get("step", 1), log=bool(cnt_d.get("log", False)))
            item = d["item"]
            values = []
            for i in range(1, n + 1):
                sub = dict(item, name=f"{item['name']}{i}")
                _, v = _suggest_one(trial, sub)
                values.append(v)
            params[d["collect"]] = tuple(values)
        else:
            name, value = _suggest_one(trial, d)
            params[name] = value
    return params


def suggest_params(trial, model: str, spec: Optional[list] = None) -> dict:
    """Sample a model's hyperparameters from ``spec`` (or its default)."""
    if spec is None:
        spec = default_space(model)
    return suggest_from_spec(trial, spec)


# Which models use Optuna's GridSampler (exhaustive over a discrete grid) vs the
# default TPE sampler. RF is a grid; the rest are continuous/TPE.
GRID_MODELS = {"Random forest"}
