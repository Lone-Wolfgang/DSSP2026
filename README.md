# DSSP2026

A modelling toolkit for the DSSP 2026 coursework. It takes the same dataset
through five classifier families — logistic regression, decision trees, random
forests, an MLP, and XGBoost — tunes each one with Optuna, persists everything
to a single SQLite file, and then reads that file back through a `Report` object
that produces the comparison tables, confusion matrices, ROC curves, and
cost-sensitive decision analysis you actually present.

Two ideas hold the package together:

- **`Experiment` writes, `Report` reads.** You hand `Experiment` a dataframe, a
  schema, and a target; it runs the studies and emits a `report.db`. Everything
  downstream — every table and figure — comes from opening that file with
  `Report`. The database is the contract between the two halves, which means a
  run is fully reproducible from the artifact alone.
- **Anything that isn't family-specific lives in `core`.** Metrics, threshold
  tuning, the heatmap/confusion primitive, the AT&T brand styling, and the
  shared result containers are written once and reused by every family.


## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Core dependencies: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `patsy`,
`scipy`, `optuna`, `matplotlib`, `seaborn`, `plotnine`. XGBoost support needs
`xgboost`; the interactive plots use `plotly`; the dashboards add `streamlit`
and `altair`; the YAML-config CLI path needs `pyyaml`. These last four are only
required for the features that use them.


## The main workflow

You are responsible for loading and cleaning the data and for the train /
validation / test split — the package does not guess at preprocessing. A typical
run looks like this:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("...")
TARGET = "readmit"

# Three-way split: tune on train, select on validation, report on test.
train_val, test = train_test_split(
    df, test_size=0.20, stratify=df[TARGET], random_state=42)
train, validation = train_test_split(
    train_val, test_size=0.25, stratify=train_val[TARGET], random_state=42)

# Declare each column's role. "categorical" columns are passed through; numeric
# columns are imputed and standardised as each family requires.
schema = {
    "age":                  "numeric",
    "length_of_stay":       "numeric",
    "num_prior_admissions": "numeric",
    "has_diabetes":         "categorical",
    "has_hypertension":     "categorical",
}

from DSSP2026.experiment.experiment import Experiment

exp = Experiment(
    train=train,
    evaluation=validation,
    test=test,
    target=TARGET,
    schema=schema,
    feature_sets={"all": list(schema)},
    experiment_dir="./readmission_run",
    n_trials=20,      # Optuna trials per tuned family
    n_splits=5,       # internal CV folds
    random_state=42,
)

exp.run()             # runs all five families with default search spaces
print(exp.report_db)  # -> ./readmission_run/.artifacts/report.db
```

Then open the result and ask it what it holds:

```python
from DSSP2026.report import Report

r = Report(exp.report_db)
print(r.describe())                                # what models / views are available
r.compare_models(allow_ensemble=True).df           # master comparison table
r.roc_compare(r.models(include_ensemble=True))      # overlaid ROC curves
r.confusion_matrix(r._best_model_name(), policy="F1")
```

`Report` is a façade composed from mixins — comparison, diagnostics, cost,
fit/selection, logistic-specific views, evaluation curves, tuning plots, and the
`describe()` capability probe — so the methods you call all live on the one
object regardless of which subsystem implements them.


## Decision policies and thresholds

Classification metrics depend on where you put the decision cutoff, so the
threshold is a first-class choice rather than a hard-coded 0.5. `core/threshold.py`
sweeps the cutoff and can optimise for **F1** or **Youden's J**, with
cross-validated variants in `experiment/cv.py`. The same policy name threads
through `compare_models`, `confusion_matrix`, and `best_fit`:

```python
r.compare_models(policy="Youden's J", allow_ensemble=True).df
best = r.best_fit(allow_ensemble=True, target="F1")   # select on validation, report on test
```


## Cost-sensitive optimization

When false positives and false negatives have different dollar consequences, you
can rank models by expected net benefit instead of by a statistical score. You
supply a payoff table — the value of each outcome per class — and `cost_optimize`
sweeps every model and threshold to maximise expected value, returning a
`CostDecision` with the winning model, its policy, the realised net benefit on
test, and ready-to-render tables:

```python
payoff = pd.DataFrame(
    {"TP/FP": [-500.0, -500.0],
     "TP":    [0.0,    2000.0],
     "FN":    [0.0,   -8000.0]},
    index=[0, 1],   # class labels
)

decision = r.cost_optimize(payoff, allow_ensemble=True)
print(decision.model_type, decision.policy, round(decision.net_benefit))
decision.policy_table.show()
decision.class_breakdown.show()
r.cost_threshold_plot(payoff, model=decision.model_type, target_class="1")
```

The cost math, the API sweep, and the deployable `CostDecisionModel` (which
attaches the chosen decision layer to a refit estimator) live in `report/cost/`.


## Command-line runner

The same comparison is available without a notebook. `DSSP2026.cli` exposes the
three workflow choices — which families to train, which feature sets to use, and
whether to evaluate against a held-out test file or a validation slice — as
arguments:

```bash
python -m DSSP2026.cli                              # all families, default sets
python -m DSSP2026.cli --models tree rf
python -m DSSP2026.cli --feature-sets both --use-test-file
python -m DSSP2026.cli --models mlp --n-trials 50
```

Every family is fit on train and scored once on the identical evaluation set, so
the rows of the master comparison table are directly comparable.


## Package layout

```
DSSP2026/
  cli.py                  Single CLI entry point for the model comparison.

  core/                   Family-agnostic foundation.
    style.py              AT&T matplotlib/seaborn theme; apply_att_style().
    color_scales.py       AT&T seaborn Objects API palettes & colormaps.
    gg.py                 AT&T plotnine (ggplot2-grammar) themes and scales.
    metrics.py            regression/classification metrics, confusion matrix,
                          ROC points, classification-report frames.
    threshold.py          Decision-cutoff tuning (F1 / Youden's J), per-class
                          and CV variants.
    heatmap.py            The shared tile-grid primitive behind the confusion
                          matrix, z-score grid, and grid-search heatmap.
    figure.py             The one save_figure helper.
    tables.py             File-format dispatch (csv/xlsx/html/png) and the
                          styled-table renderers.
    results.py            ClassificationResult container.
    encoders.py           Formula-callable transforms (cyc_sin, cyc_cos).

  experiment/             The "writes" half — runs studies, emits report.db.
    experiment.py         The Experiment orchestrator.
    data.py               Loading and the train / validation / test split.
    columns.py            Schema-driven column-type resolution.
    spaces.py             Spec-driven Optuna search spaces (one per family).
    objectives.py         Unified Optuna objective factory.
    cv.py                 The authoritative home for all cross-validation.
    study.py              Run one family's study and evaluate its winner.
    sidecar.py            The eval sidecar DB, kept separate from the study DB.
    refit.py              Rebuild a fitted predictor from stored config — the
                          single source of truth across all five families.
    trials.py             Flatten an Optuna study's trials to rows.
    logistic_adapter.py   Binary/multiclass routing for logistic studies.
    fit.py                Metric-policy selection and deployable fitting.
    tuning/               Optuna search runner plus training-curve and
                          parallel-coordinates report plots.

  report/                 The "reads" half — everything you present.
    report.py             The Report façade (composed from the mixins below).
    report_builder.py     Merge study + eval DBs into the reporting report.db.
    base.py               ReportBase: DB access and shared helpers.
    compare.py            compare_models, roc_compare.
    evaluation.py         Feature-importance and training-curve views.
    diagnostics.py        Confusion matrices and related diagnostics.
    fit.py                best_fit / PolicyModel selection.
    policy.py             Metric-driven decision policies.
    logistic.py           Logistic coefficients table & odds-ratio plot.
    logistic_tables.py    Logistic table stylers.
    tuning.py             report.db-backed tuning views (elbow, heatmap, …).
    describe.py           Report.describe() capability probe.
    residual_artifact.py  Persist/load the OLS residual-diagnostics frame.
    plots.py              ConfusionPlot, ROCPlot, ThresholdSweepPlot, etc.
    tables.py             ReportTable.
    cost/                 Cost-sensitive subsystem.
      math.py             Shared cost / net-benefit math.
      api.py              The cost_optimize sweep (CostMixin).
      fit.py              CostDecisionModel — attach a cost-optimal layer.
      tables.py           CostDecision, PolicyTable, ClassBreakdownTable,
                          CostConfusion.

  linear_regression/      OLS — fit, diagnostics, ranking, interactive plots.
  logistic_regression/    Binary (binary.py) and multinomial (multiclass.py).
  tree/                   Shared tree helpers, split into classification/ and
                          regression/ fit+tune.
  mlp/                    MLP fit, sklearn pipeline construction, Optuna tune.
  xgboost/                XGBoost fit and Optuna tune.

  dashboards/             Streamlit/Altair apps.
    cost_analysis.py      Interactive cost / net-benefit comparison.
    residuals.py          Standalone OLS residual-diagnostics dashboard.
    databricks.py         Launch the dashboard from a Databricks notebook.
    style.py              AT&T-aligned CSS for Streamlit.
```


## Conventions

**Brand styling.** Call `apply_att_style()` once at the top of a script or
notebook (the `report` package does this on import). Matplotlib rcParams,
seaborn theme, fonts, and the `att_sequential` / `att_diverging` colormaps are
set globally. The same palette is available for the seaborn Objects API
(`core/color_scales.py`) and for plotnine (`core/gg.py`).

**Schema, not guesswork.** Columns are typed explicitly as `"numeric"` or
`"categorical"`. Numeric columns are imputed and standardised per family;
categorical 0/1 flags are passed through untouched.

**One-way dependencies.** Fit layers return models and DataFrames and never
import the presentation layer; figures and stylers come from each family's
`plots.py` / `tables.py`. Result containers (`OLSResult`, `LogitResult`,
`ClassificationResult`, `CostDecision`, …) are data-only and produce figures on
demand.

**Reproducible artifacts.** A run is reconstructable from `report.db` alone.
`experiment/refit.py` is the single place that rebuilds a fitted estimator from
stored configuration, so every family refits the same way whether you're scoring,
plotting, or attaching a cost layer.

**Confusion-matrix readability.** `plot_confusion_matrix(..., normalize=True)`
shades cells by row-normalised recall while printing both the raw count and the
percentage, so the diagonal stays interpretable on imbalanced classes; per-cell
text contrast is chosen by luminance.

**Formulas with inline transforms.** The statsmodels-based fit layers inject a
formula namespace into the patsy environment, so `log`, `sqrt`, `exp`, `cyc_sin`,
and `cyc_cos` work directly:

```python
fit_logit(df, "OnTime ~ Distance + cyc_sin(DayOfWeek, 7) + cyc_cos(DayOfWeek, 7)")
fit_ols(df, "y ~ log(carat) + C(cut)")
```


## License

MIT.