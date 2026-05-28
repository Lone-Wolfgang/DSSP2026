# DSSP2026

Personal toolkit for the DSSP 2026 coursework. A small modelling stack built
around two ideas: every model family follows the same four-file shape, and
anything that isn't family-specific lives in `core`.

The goal is to keep notebooks short — write the data prep and the formula,
let the library produce the figures and tables.


## Layout

```
modules/
  core/
    style.py        AT&T brand palette, fonts, matplotlib rcParams, HTML
                    table styles. Apply with apply_att_style().
    metrics.py      Family-agnostic scoring. regression_metrics(),
                    classification_metrics(), make_confusion_matrix(),
                    roc_curve_points(), make_classification_report_df().
    plot.py         save_figure, the plot_tile_grid primitive used by the
                    grid-search heatmap and confusion matrix, plus
                    plot_confusion_matrix / plot_roc_curve and the cross-
                    family MSPE/R² comparison chart.
    tables.py      File-format dispatch (csv/xlsx/html/png), the generic
                    matplotlib PNG table renderer, classification report
                    styler, and classifier-comparison styler.
    encoders.py     Formula-callable transforms (cyc_sin, cyc_cos) plus
                    numpy helpers, exported as FORMULA_NAMESPACE. Both fit
                    layers inject this into the formula evaluation env, so
                    you can write transforms inline in smf formulas.

  regression/
    fit.py          fit_ols, predict_ols, make_ols_coef_df,
                    make_ols_diagnostics_df, rank_ols_models.
    tune.py         (stub — info-criterion ranking still lives in fit.py)
    plots.py        Fit line, actual-vs-fitted, partial regression,
                    residual diagnostics quad, model-comparison curves,
                    static + interactive (plotly) prediction plots.
    tables.py       Coefficient table, model-comparison table, and the
                    wide-format prediction summary with extrapolation flags.
    workflow.py     fit_and_plot_ols, predict_and_plot — one-call
                    orchestrators for notebook use.

  logistic/
    fit.py          fit_logit (statsmodels GLM via the formula interface),
                    predict_logit, make_logit_coef_df with odds ratios.
                    Includes a guard that catches non-0/1 targets early.
    tune.py         tune_threshold — sweep decision cutoff, optimize F1
                    or Youden's J.
    plots.py        plot_odds_ratios — the logistic-specific forest plot.
                    Classification eval plots live in core (shared).
    tables.py       Odds-ratio coefficient styler and PNG saver.

  tree/
    fit.py          fit_decision_tree, fit_random_forest,
                    feature_importance_df.
    tune.py         tune_dt_depth (elbow), tune_rf_grid (CV grid search).
    plots.py        Depth-elbow curve, feature importance bars, grid
                    search heatmap (uses the core tile primitive).
    tables.py       Re-exports the tree metrics styler from core.
```

The pattern across families: `fit` is pure stats (returns models and
DataFrames), `tune` handles hyperparameter selection, `plots` and `tables`
produce presentation-quality figures and styled tables. Anything that takes
`(y_true, y_pred)` or `(y_true, y_score)` lives in `core`, since every
classifier family reuses it.


## Install

```bash
pip install -r requirements.txt
```


## Conventions

**Brand styling.** Call `apply_att_style()` once at the top of a script or
notebook. Matplotlib rcParams, seaborn theme, fonts, and two colormaps
(`att_sequential`, `att_diverging`) are set globally.

**Formulas with inline transforms.** Both fit layers inject
`FORMULA_NAMESPACE` into the patsy evaluation environment, so `log`,
`sqrt`, `exp`, `cyc_sin`, and `cyc_cos` work directly in the formula:

```python
fit_logit(df, "OnTime ~ Distance + cyc_sin(DayOfWeek, 7) + cyc_cos(DayOfWeek, 7)")
fit_ols(df, "y ~ log(carat) + C(cut)")
```

**Saving outputs.** Every plot has a `save_*_png` companion in the same
module. Tables route through `core.tables.save_table_by_extension`, which
dispatches on `.csv` / `.xlsx` / `.html` / `.png`.

**Confusion matrix readability.** `plot_confusion_matrix(..., normalize=True)`
colors cells by row-normalized recall while printing both the raw count
and the percentage in each cell, so the diagonal stays interpretable on
imbalanced classes. Text contrast is chosen per-cell by luminance.


## Quick example — logistic regression

```python
import pandas as pd
from sklearn.model_selection import train_test_split

from modules.core.style import apply_att_style
from modules.core import metrics as CM, plot as CP, tables as CT
from modules.logistic import fit as LF, plots as LP, tables as LTB

apply_att_style()
train, test = train_test_split(df, test_size=0.3, random_state=42,
                               stratify=df["OnTime"])

# Fit + odds-ratio coefficient table
res = LF.fit_logit(train, "OnTime ~ Distance + C(DayOfWeek)")
LTB.save_logit_coefficients_png(res.coef_df, res.model, "outputs/coefs.png")
LP.save_odds_ratios_png(res.coef_df, "outputs/odds_ratios.png")

# Predict on test set, evaluate through core
pred = LF.predict_logit(res.model, test, threshold=0.5)
y_true = LF.get_endog(res.model, test)

cm = CM.make_confusion_matrix(y_true, pred.labels, labels=[0, 1])
CP.save_confusion_matrix_png(cm, "outputs/cm.png",
                             class_labels=["Late", "OnTime"], normalize=True)

fpr, tpr, _, auc = CM.roc_curve_points(y_true, pred.proba)
CP.save_roc_curve_png((fpr, tpr, auc), "outputs/roc.png")
```


## Design notes

- **Fit layers never import presentation.** `regression/fit.py` and
  `logistic/fit.py` return models and DataFrames; figures and stylers come
  from the family's `plots.py` / `tables.py`. This keeps the dependency
  direction one-way (`workflow → plots/tables → fit/core`).
- **Result containers are data-only.** `OLSResult`, `LogitResult`,
  `PredictionResult` don't hold figures or Stylers — those are produced on
  demand by the presentation layer.
- **Shared primitives.** `plot_tile_grid` in `core/plot.py` is the single
  heatmap renderer used by both the RF grid-search heatmap and the
  confusion matrix. Text contrast is luminance-based.
- **Formula targets.** statsmodels' logit needs a numeric 0/1 target;
  `fit_logit` checks this up front and raises an actionable error rather
  than letting users hit the cryptic "multiple columns" message.


## License

MIT.