"""
experiment/trials.py — normalize an Optuna study's trials to flat rows.

One place that turns ``study.trials`` into the dict rows the report builder
persists. A trial row captures the candidate's params, CV value, state, rank
(by value among completed), whether it's the best, and duration.
"""

from __future__ import annotations


def trials_from_study(study, *, model=None) -> list:
    """Return a list of trial dicts for an Optuna study.

    Each dict: {trial_number, params, cv_value, state, rank, is_best,
    duration_sec, user_attrs}. Rank is 1-based by descending value among
    completed trials.
    """
    try:
        best_number = study.best_trial.number
    except Exception:
        best_number = None

    completed = [t for t in study.trials
                 if t.value is not None and str(t.state).endswith("COMPLETE")]
    order = sorted(completed, key=lambda t: t.value, reverse=True)
    rank_of = {t.number: i + 1 for i, t in enumerate(order)}

    rows = []
    for t in study.trials:
        duration = None
        if getattr(t, "datetime_start", None) and getattr(t, "datetime_complete", None):
            duration = (t.datetime_complete - t.datetime_start).total_seconds()
        rows.append({
            "model": model,
            "trial_number": t.number,
            "params": dict(t.params),
            "cv_value": float(t.value) if t.value is not None else None,
            "state": str(t.state).split(".")[-1],
            "rank": rank_of.get(t.number),
            "is_best": (t.number == best_number),
            "duration_sec": duration,
            "user_attrs": dict(t.user_attrs),
        })
    return rows
