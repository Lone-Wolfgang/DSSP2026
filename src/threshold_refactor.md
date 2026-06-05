# Reorganization plan: consolidate classification threshold tuning

## Goal

Classification decision-threshold tuning is currently implemented across three
files with substantial duplication. This plan extracts the shared machinery into
a new `core/threshold.py` module, rewrites the per-family entry points as thin
adapters over a single CV engine, removes dead code, and fixes one latent bug.

The public API (function names, signatures, return types) must remain
**backward compatible** unless explicitly noted below. Existing callers and the
plot/table layer should keep working with no changes beyond import paths.

---

## Current state

### File 1 — `DSSP2026/logistic_regression/tune.py`

Contains the core primitive and some dead code.

- `tune_threshold(y_true, y_proba, *, metric="f1", thresholds=None, pos_label=1, min_recall=None, max_false_negative_rate=None) -> ThresholdSweepResult`
  — the real engine. Sweeps a probability grid, computes the full per-threshold
  metric frame (accuracy, precision, recall, f1, youden, FNR, FPR, specificity,
  tp/fp/tn/fn, roc_auc), applies optional recall / FNR constraints, returns the
  best row. **Model-agnostic** — takes labels and probabilities only.
- `tune_roc_threshold(...)` — thin pass-through to `tune_threshold(thresholds=None)`.
- `ThresholdSweepResult` (dataclass) — return type of the two functions above.
- `_ALLOWED_METRICS` (tuple), `_MINIMIZE_METRICS` (set) — module constants used
  for validation and to decide max-vs-min selection.
- `_confusion_counts(y_true, y_pred, *, pos_label=1)` — **DEAD CODE**. No caller
  anywhere in the repo. `tune_threshold` computes tp/fp/tn/fn inline.
- `_youden_j(y_true, y_pred, *, pos_label=1)` — **DEAD CODE**. No caller. Youden's
  J is computed inline in `tune_threshold` (`recall + specificity - 1`).

### File 2 — `DSSP2026/logistic_regression/workflow.py`

CV wrapper plus the plot/table layer.

- `cross_validate_threshold(df, formula, *, metric="f1", n_splits=5, thresholds=None, pos_label=1, min_recall=None, max_false_negative_rate=None, random_state=0, maxiter=100) -> CVThresholdResult`
  — stratified k-fold CV. Per fold: fit a logit (`fit_logit` / `predict_proba` /
  `get_endog` from `logistic_regression.binary`), call `tune_threshold` on a
  **shared fixed grid** (default `np.linspace(0.01, 0.99, 99)`), then aggregate
  mean ± std by threshold and interpolate per-fold ROC onto a common FPR grid
  (`np.linspace(0.0, 1.0, 101)`). Builds and returns `CVThresholdResult`.
- `CVThresholdResult` (dataclass) — pure-data container (summary_df, per_fold_df,
  roc_per_fold, roc_mean, fold_best, metric, formula, n_splits, headline means).
- Plot / table functions, **all model-agnostic — they read only a
  `CVThresholdResult` and need no changes beyond their import of `CVThresholdResult`**:
  `plot_cv_threshold_metrics`, `plot_cv_roc`, `save_cv_threshold_metrics`,
  `save_cv_roc`, `make_cv_summary_df`, `style_cv_summary`, `save_cv_summary`,
  `_summary_caption`, `_save_cv_summary_png`, `att_palette_for`,
  `_DEFAULT_PLOT_METRICS`.

### File 3 — `DSSP2026/tree/classification/tune.py`

Contains the duplication.

- `cross_validate_tree_threshold(df, features, target, *, metric="f1", max_depth=None, class_weight="balanced", n_splits=5, thresholds=None, pos_label=1, min_recall=None, max_false_negative_rate=None, random_state=0, estimator_factory=None) -> CVThresholdResult`
  — the redundant function. The CV fold-loop and the entire aggregation body
  (groupby-threshold mean/std, ROC interpolation, `CVThresholdResult`
  construction) are **copy-pasted nearly verbatim** from
  `cross_validate_threshold`. The only genuine differences:
  1. Per fold the estimator is a `DecisionTreeClassifier` (or an
     `estimator_factory`) instead of a logit.
  2. Probabilities come from `clf.predict_proba(...)[:, pos_col]`, where
     `pos_col` is found by mapping `pos_label` through `clf.classes_` (with a
     guard that raises if `pos_label` is not a fitted class).
  3. `formula` is set to a readable `"target ~ f1 + f2 + ..."` label.
  It imports `tune_threshold`, `CVThresholdResult`, and `_ALLOWED_METRICS` from
  the `logistic_regression` package — a `tree` module reaching into
  `logistic_regression` for shared machinery (architectural smell).
- `cross_validate_rf_threshold(df, features, target, *, metric="f1_macro", n_estimators=300, max_features="sqrt", max_depth=None, class_weight="balanced", n_splits=5, thresholds=None, pos_label=1, min_recall=None, max_false_negative_rate=None, random_state=0) -> CVThresholdResult`
  — builds a `RandomForestClassifier` factory and delegates to
  `cross_validate_tree_threshold`. **Has a latent bug:** the default
  `metric="f1_macro"` is not in `_ALLOWED_METRICS` (which contains `f1`, not
  `f1_macro`), so calling it with defaults raises `ValueError` from
  `tune_threshold`. See fix below.
- Also in this file but **unrelated to threshold tuning — leave untouched:**
  `DepthTuneResult`, `tune_dt_depth_cv`, `RFGridResult`, `tune_rf_grid_classify`.

### Other consumers (no duplication — handle as noted)

- `DSSP2026/evaluation/classification.py::plot_roc_threshold_tuning` /
  `save_roc_threshold_tuning_png` consume a `ThresholdSweepResult` but **only by
  duck typing** (`tuning.roc_df`, `tuning.best_row`, `tuning.best_threshold`,
  `tuning.best_metric`, `tuning.best_value`). They do **not** import
  `ThresholdSweepResult`. No changes required.
- `DSSP2026/workflows/runners.py` uses `tree.classification.tune` as `TT` but
  only calls `tune_dt_depth_cv`, `tune_rf_grid_classify`, and the grid heatmap —
  **not** the threshold CV functions. No changes required, but verify after the
  move that `from DSSP2026.tree.classification import tune as TT` still resolves
  (it will, since those functions stay in that module).

---

## Target state

### New file — `DSSP2026/core/threshold.py`

This becomes the single home for all model-agnostic threshold machinery,
consistent with the existing pattern where `core/` holds shared building blocks
(`metrics`, `style`, `tables`, `figure`, `color_scales`).

Move into it, unchanged in behavior:

- `ThresholdSweepResult` (dataclass)
- `CVThresholdResult` (dataclass)
- `_ALLOWED_METRICS`, `_MINIMIZE_METRICS`
- `tune_threshold(...)`
- `tune_roc_threshold(...)`

Add a new **generic CV engine** that both logistic and tree adapters call:

```python
def cross_validate_threshold_generic(
    df,
    *,
    fold_proba_fn,        # callable(train_df, val_df) -> (y_val: np.ndarray, proba: np.ndarray)
    strat_labels,         # array-like used for StratifiedKFold split + as per-fold truth source
    formula,              # label string stored on CVThresholdResult.formula
    metric="f1",
    n_splits=5,
    thresholds=None,
    pos_label=1,
    min_recall=None,
    max_false_negative_rate=None,
    random_state=0,
) -> CVThresholdResult:
    ...
```

This function holds the **single copy** of the shared body that is currently
duplicated:

- validate `metric in _ALLOWED_METRICS`, `n_splits >= 2`
- default `thresholds = np.linspace(0.01, 0.99, 99)`
- `StratifiedKFold(n_splits, shuffle=True, random_state=random_state)` over
  `strat_labels`
- per fold: call `fold_proba_fn(train, val)` to get `(y_val, proba)`, then
  `tune_threshold(y_val, proba, metric=..., thresholds=..., pos_label=...,
  min_recall=..., max_false_negative_rate=...)`, wrapping its `ValueError` with
  the existing `"Fold {i}: ... constraint ... could not be met"` message
- collect per-fold sweep frames, fold-best rows, and per-fold ROC interpolated
  onto `fpr_grid = np.linspace(0.0, 1.0, 101)`
- aggregate mean/std by threshold, build long `per_fold_df`, `roc_mean`,
  `fold_best`, and construct `CVThresholdResult` with the headline means

Keep the aggregation arithmetic **byte-for-byte equivalent** to the current
logistic implementation (same `ddof=0`, same `_mean`/`_std` suffixes, same
`interp_tprs[:, 0] = 0.0` origin fix) so existing outputs/plots are unchanged.

### `DSSP2026/logistic_regression/tune.py`

- Remove `ThresholdSweepResult`, `_ALLOWED_METRICS`, `_MINIMIZE_METRICS`,
  `tune_threshold`, `tune_roc_threshold` (now in `core.threshold`).
- **Delete the dead `_confusion_counts` and `_youden_j` entirely.**
- For backward compatibility, re-export the moved names so existing imports
  (`from DSSP2026.logistic_regression.tune import tune_threshold, _ALLOWED_METRICS`)
  keep working:
  ```python
  from DSSP2026.core.threshold import (
      ThresholdSweepResult, tune_threshold, tune_roc_threshold,
      _ALLOWED_METRICS, _MINIMIZE_METRICS,
  )
  ```
  (If the team prefers a hard cutover instead of shims, update the two importing
  files directly and drop the re-export — see the import sites listed below.)

### `DSSP2026/logistic_regression/workflow.py`

- Import `CVThresholdResult`, `tune_threshold`, `_ALLOWED_METRICS`,
  `cross_validate_threshold_generic` from `DSSP2026.core.threshold`.
- Rewrite `cross_validate_threshold` as a thin adapter: define a local
  `fold_proba_fn(train, val)` that runs `fit_logit(train, formula, maxiter=...)`,
  `predict_proba`, `get_endog`, returning `(y_val, proba)`; pull `strat_labels`
  from `df[lhs]` (keep the existing LHS-column validation and error message);
  pass `formula=formula`. Delegate everything else to
  `cross_validate_threshold_generic`. Signature and return value unchanged.
- The plot/table layer stays in this file; only its `CVThresholdResult` import
  changes to `core.threshold`. (Optional, larger refactor — NOT required here:
  these are fully generic and could later move to `core` too. Out of scope.)

### `DSSP2026/tree/classification/tune.py`

- Replace the duplicated body of `cross_validate_tree_threshold` with a thin
  adapter over `cross_validate_threshold_generic`: define `fold_proba_fn(train,
  val)` that builds the estimator (default `DecisionTreeClassifier(max_depth=...,
  class_weight=..., random_state=...)` unless `estimator_factory` is supplied),
  fits on `train[features]/train[target]`, maps `pos_label` through
  `clf.classes_` to get `pos_col` (keep the guard that raises if `pos_label`
  isn't a fitted class), and returns `(val[target].to_numpy(),
  clf.predict_proba(val[features])[:, pos_col])`. Pass `strat_labels =
  df[target].to_numpy()`, keep the `target in df.columns` check, and set
  `formula = f"{target} ~ " + " + ".join(features)`. Signature and return value
  unchanged.
- Update imports to pull `cross_validate_threshold_generic` (and any needed
  helpers) from `DSSP2026.core.threshold` instead of from
  `logistic_regression.tune` / `logistic_regression.workflow`. This removes the
  `tree -> logistic_regression` dependency.
- **Fix the `f1_macro` bug** in `cross_validate_rf_threshold`: change the default
  from `metric="f1_macro"` to `metric="f1"` (the per-threshold sweep is binary
  and `tune_threshold` only accepts `_ALLOWED_METRICS`, which has `f1`). Update
  the docstring line that currently says the default is `"f1_macro"` to match.
  Note: `f1_macro` remains valid where it belongs — `tune_dt_depth_cv` and
  `tune_rf_grid_classify` use sklearn scorers and are unaffected.

---

## Concrete import sites to update

- `DSSP2026/logistic_regression/workflow.py:45`
  `from DSSP2026.logistic_regression.tune import tune_threshold, _ALLOWED_METRICS`
  → import from `DSSP2026.core.threshold` (plus `CVThresholdResult`,
  `cross_validate_threshold_generic`).
- `DSSP2026/tree/classification/tune.py:266-268`
  `from DSSP2026.logistic_regression.tune import tune_threshold`
  `from DSSP2026.logistic_regression.workflow import CVThresholdResult`
  `from DSSP2026.core.metrics import roc_curve_points`
  → `tune_threshold` / `CVThresholdResult` / `cross_validate_threshold_generic`
  from `DSSP2026.core.threshold`; `roc_curve_points` import is only needed if the
  adapter still computes ROC itself, which it should NOT after delegation —
  remove it from this function.
- `DSSP2026/tree/classification/tune.py:340`
  `from DSSP2026.logistic_regression.tune import _ALLOWED_METRICS`
  → no longer needed in the adapter (the generic engine validates the metric);
  remove.

If keeping the re-export shim in `logistic_regression/tune.py`, no other edits
to import statements are required.

---

## Acceptance criteria

1. `DSSP2026/core/threshold.py` exists and contains `ThresholdSweepResult`,
   `CVThresholdResult`, `tune_threshold`, `tune_roc_threshold`,
   `cross_validate_threshold_generic`, `_ALLOWED_METRICS`, `_MINIMIZE_METRICS`.
2. The fold-loop + aggregation body exists in **exactly one place**
   (`cross_validate_threshold_generic`). `cross_validate_threshold` (logistic)
   and `cross_validate_tree_threshold` (tree) are thin adapters with no
   duplicated aggregation logic.
3. `_confusion_counts` and `_youden_j` are deleted; a repo-wide grep finds no
   references to them.
4. No module under `DSSP2026/tree/` imports from
   `DSSP2026.logistic_regression.*` for threshold machinery.
5. `cross_validate_rf_threshold` runs with its default `metric` without raising
   `ValueError`.
6. `plot_roc_threshold_tuning` / `save_roc_threshold_tuning_png` in
   `evaluation/classification.py` and all `plot_cv_*` / `save_cv_*` /
   `make_cv_summary_df` / `style_cv_summary` functions work unchanged on the
   objects returned by the refactored functions.
7. `DSSP2026/workflows/runners.py` still imports and runs (`tune_dt_depth_cv`,
   `tune_rf_grid_classify`, grid heatmap) without modification.
8. Existing public signatures and return types are preserved (modulo the
   `f1_macro` -> `f1` default fix in item 5).

## Suggested verification

- `grep -rn "_confusion_counts\|_youden_j" DSSP2026/` → no matches.
- `grep -rn "logistic_regression" DSSP2026/tree/` → no threshold-related imports.
- Smoke test on a small synthetic imbalanced binary frame: call
  `cross_validate_threshold` (logistic), `cross_validate_tree_threshold`, and
  `cross_validate_rf_threshold` (with default metric) and assert each returns a
  `CVThresholdResult` whose `summary_df` has `*_mean`/`*_std` columns and whose
  `roc_mean` starts at the origin.
- If the team has snapshot fixtures, confirm the refactored logistic
  `cross_validate_threshold` reproduces the pre-refactor `summary_df`,
  `roc_mean`, and headline means bit-for-bit.

## Out of scope (do not do as part of this change)

- Moving the plot/table layer out of `logistic_regression/workflow.py` into
  `core` (possible future cleanup; not required here).
- Touching `tune_dt_depth_cv`, `tune_rf_grid_classify`, `DepthTuneResult`,
  `RFGridResult`, or any non-threshold tuning.
- The default-0.5 `predict_labels` thresholding in
  `logistic_regression/binary.py` (that's prediction, not tuning).
  