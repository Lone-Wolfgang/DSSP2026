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


VALID_MODELS = (
    "Logistic regression",
    "Decision tree",
    "Random forest",
    "MLP",
    "XGBoost",
)
GRID_MODELS = {"Random forest"}
DEFAULT_SAMPLERS = {m: ("grid" if m in GRID_MODELS else "tpe") for m in VALID_MODELS}


def default_space(model: str) -> list:
    """Return a deep copy of the built-in spec for ``model`` (safe to edit)."""
    if model not in _DEFAULT_SPACES:
        raise ValueError(f"no default space for model {model!r}.")
    return copy.deepcopy(_DEFAULT_SPACES[model])


def default_spaces() -> dict:
    """All built-in specs as {model: spec} (deep-copied)."""
    return {m: copy.deepcopy(s) for m, s in _DEFAULT_SPACES.items()}


def _validate_model(model: str):
    if model not in VALID_MODELS:
        raise ValueError(
            f"unknown model {model!r}; valid models are {list(VALID_MODELS)}.")


def _descriptor_names(spec: list) -> set:
    names = set()
    for d in spec:
        if d.get("type") == "group":
            names.add(d["count"]["name"])
            names.add(d["item"]["name"])
            names.add(d["collect"])
        elif "name" in d:
            names.add(d["name"])
    return names


def tunable_params(model: str) -> list[dict]:
    """The tunable hyperparameters for `model`: name, type, and bounds/choices,
    from the built-in default space. Raises for unknown model."""
    _validate_model(model)
    params = []
    for d in _DEFAULT_SPACES[model]:
        if d.get("type") == "group":
            count = copy.deepcopy(d["count"])
            count["group"] = d["collect"]
            params.append(count)
            item = copy.deepcopy(d["item"])
            item["name"] = item["name"]
            item["group"] = d["collect"]
            params.append(item)
        else:
            params.append(copy.deepcopy(d))
    return params


def _grid_bad_descriptor(spec: list):
    for d in spec:
        if d.get("type") == "group":
            return d["collect"], "group"
        t = d.get("type")
        if t == "categorical":
            continue
        if t == "int" and "low" in d and "high" in d and not d.get("log", False):
            continue
        return d.get("name", "<unnamed>"), t
    return None


def grid_compatible(model: str, spec: list | None = None) -> bool:
    """True iff the (given or default) spec is fully discrete — every descriptor
    is categorical, or int with finite low/high (no float, no log-float).
    Grid search requires this."""
    _validate_model(model)
    spec = default_space(model) if spec is None else spec
    return _grid_bad_descriptor(spec) is None


def _format_descriptor(d: dict) -> str:
    t = d.get("type")
    name = d.get("name", d.get("collect", "<unnamed>"))
    if t == "categorical":
        return f"{name}: categorical choices={list(d['choices'])}"
    if t in ("int", "float"):
        bits = [f"{name}: {t}", f"{d['low']}..{d['high']}"]
        if d.get("step") is not None:
            bits.append(f"step={d['step']}")
        if d.get("log"):
            bits.append("log")
        return " ".join(bits)
    if t == "group":
        return f"{d['collect']}: group count={d['count']['name']} item={d['item']['name']}"
    return f"{name}: {t}"


def describe(model: str | None = None) -> str:
    """Human-readable summary: for each model (or one), list tunable params with
    type + range/choices, the default sampler, and whether grid is available.
    Returns a string."""
    models = VALID_MODELS if model is None else (model,)
    lines = []
    for m in models:
        _validate_model(m)
        lines.append(f"{m}")
        lines.append(f"  default sampler: {DEFAULT_SAMPLERS[m]}")
        lines.append(f"  grid available: {'yes' if grid_compatible(m) else 'no'}")
        spec = _DEFAULT_SPACES[m]
        if not spec:
            lines.append("  no tunable hyperparameters (one config per feature set)")
        else:
            lines.append("  tunable parameters:")
            for d in spec:
                lines.append(f"    - {_format_descriptor(d)}")
    return "\n".join(lines)


def _normalize_entry(model: str, entry):
    if entry is None:
        params = default_space(model)
        sampler = DEFAULT_SAMPLERS[model]
    elif isinstance(entry, list):
        params = copy.deepcopy(entry)
        sampler = DEFAULT_SAMPLERS[model]
    elif isinstance(entry, dict):
        params = copy.deepcopy(entry.get("params", default_space(model)))
        sampler = entry.get("sampler", DEFAULT_SAMPLERS[model])
    else:
        raise ValueError(
            f"search_space entry for {model!r} must be a list or dict; got {type(entry).__name__}.")
    if sampler not in {"grid", "tpe"}:
        raise ValueError(f"sampler for {model!r} must be 'grid' or 'tpe'; got {sampler!r}.")
    valid_names = _descriptor_names(_DEFAULT_SPACES[model])
    for d in params:
        names = _descriptor_names([d])
        unknown = sorted(n for n in names if n not in valid_names)
        if unknown:
            raise ValueError(
                f"unknown tunable parameter(s) for {model!r}: {unknown}; "
                f"valid names are {sorted(valid_names)}.")
    if sampler == "grid":
        bad = _grid_bad_descriptor(params)
        if bad is not None:
            name, typ = bad
            raise ValueError(
                f"{model} grid requested but param {name} is a continuous {typ}; "
                "use tpe or replace with categorical choices.")
    return {"params": params, "sampler": sampler}


def normalize_search_space(user_space: dict | None) -> dict:
    """Return {model: {"params": list, "sampler": "grid"|"tpe"}} merged over the
    built-in defaults."""
    if user_space:
        for model in user_space:
            _validate_model(model)
    return {
        model: _normalize_entry(model, user_space.get(model) if user_space else None)
        for model in VALID_MODELS
    }


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


