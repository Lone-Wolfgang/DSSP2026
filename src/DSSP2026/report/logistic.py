"""
report/logistic.py — logistic-regression-specific report views.

The two views here are the logistic family's own diagnostics: the odds-ratio
coefficient table and the odds-ratio forest plot (the "log odds plot"). Both
are built from a *fitted statsmodels model*, not from stored predictions — so
unlike the confusion / ROC / threshold views (which reconstruct from the
probability matrix in report.db), these cannot be served from the DB alone.

Why a refit
-----------
report.db persists predictions, per-class metrics, confusion counts,
hyperparams, and the winning ``feature_list`` — but never the fitted
coefficients (params / std errors / p-values / CIs). Logistic regression here
is unregularised MLE, so the coefficients can't be reconstructed from the
stored metrics; they only exist on a fitted model object. To recover them we
refit the winning feature set on the training frame the caller supplies, then
hand the fitted model to the existing presentation code in
``logistic_regression`` — the same refit the experiment's ``_refit_winner`` and
the ``workflows`` runner already perform. The DB supplies the *which* (the
winning feature list and whether the study was binary or multiclass); the
caller supplies the *data* to refit on.

This keeps the heavy lifting in ``logistic_regression`` (one source of truth
for styling and the forest plot) and adds only the report-shaped result
wrappers (``.to_png`` / ``.to_csv`` / ``.to_html`` + inline render) so these
views chain like every other report result.
"""

from __future__ import annotations

import json
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DSSP2026.core.color_scales import so, att_nominal
from DSSP2026.core.figure import save_figure
from DSSP2026.core.style import ATT_COLORS


# ---------------------------------------------------------------------------
# Result wrappers — delegate to logistic_regression's styling / plotting
# ---------------------------------------------------------------------------

def plot_odds_ratios(
    coef_df: pd.DataFrame,
    *,
    drop_intercept: bool = True,
    log_scale: bool = True,
    title: str = "Odds ratios with 95% CI",
    figsize: Optional[tuple] = None,
    ax=None,
):
    """Forest plot of odds ratios and their confidence intervals.

    Expects the columns produced by logistic.fit.make_logit_coef_df:
    'Term', 'Odds Ratio', and the two 'OR CI ...' bounds. Predictors whose
    CI excludes 1 (significant) are drawn in the brand accent color; the rest
    are muted. A reference line marks OR = 1 (no effect).

    Parameters
    ----------
    log_scale : bool
        Plot the OR axis on a log scale (default). Odds ratios are
        multiplicative, so a log axis renders 0.5 and 2.0 symmetrically about 1.
    """
    or_col = "Odds Ratio"
    ci_cols = [c for c in coef_df.columns if c.startswith("OR CI ")]
    if or_col not in coef_df.columns or len(ci_cols) != 2:
        raise ValueError(
            "coef_df must have an 'Odds Ratio' column and two 'OR CI ...' "
            "bound columns (use logistic.fit.make_logit_coef_df)."
        )
    low_col, high_col = ci_cols[0], ci_cols[1]

    df = coef_df.copy()
    if drop_intercept:
        df = df[df["Term"] != "Intercept"]
    df = df.iloc[::-1].reset_index(drop=True)  # first term at the top

    # Tidy frame for the Objects API. Significance becomes a real column so the
    # color split goes through the brand scale instead of a hand-set loop.
    plot_df = pd.DataFrame({
        "Term": df["Term"].astype(str),
        "or": df[or_col].to_numpy(),
        "lo": df[low_col].to_numpy(),
        "hi": df[high_col].to_numpy(),
    })
    plot_df["Significance"] = np.where(
        (plot_df["lo"] > 1) | (plot_df["hi"] < 1), "Significant", "Not significant")

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or (9, max(3, 0.6 * len(plot_df) + 1.8)))
    else:
        fig = ax.figure

    # Order so the first term sits at the top; categorical y preserves order.
    order = plot_df["Term"].tolist()
    # Significant -> deep_blue (first palette slot), Not -> gray.
    sig_scale = att_nominal(order=["Significant", "Not significant"])

    p = (
        so.Plot(plot_df, y="Term", color="Significance")
          .add(so.Range(linewidth=2.4), xmin="lo", xmax="hi")
          .add(so.Dot(pointsize=9, edgecolor="white", edgewidth=1.0), x="or")
          .scale(y=so.Nominal(order=order),
                 color=so.Nominal(values=[ATT_COLORS["deep_blue"],
                                          ATT_COLORS["gray_500"]],
                                  order=["Significant", "Not significant"]))
          .label(x="Odds ratio", y="", title=title)
    )
    p.on(ax).plot()

    # Reference line + log scale are not data layers — add them directly.
    ax.axvline(1.0, color=ATT_COLORS["orange"], linestyle="--", linewidth=1.6,
               zorder=2, label="No effect (OR = 1)")
    if log_scale:
        ax.set_xscale("log")
        ax.set_xlabel("Odds ratio (log scale)")
    ax.margins(y=0.12)
    fig.tight_layout()
    return fig


def save_odds_ratios(coef_df, path, *, drop_intercept=True, log_scale=True,
                     dpi=220, **kwargs):
    """Render the forest plot and save it (matplotlib formats: png/pdf/svg…)."""
    fig = plot_odds_ratios(coef_df, drop_intercept=drop_intercept,
                           log_scale=log_scale, **kwargs)
    return save_figure(fig, path, dpi=dpi)


save_odds_ratios_png = save_odds_ratios

class CoefficientsTable:
    """Odds-ratio coefficient table for a fitted logistic model.

    Wraps the fitted model and its tidy ``coef_df`` and routes every render and
    export through ``logistic_regression``'s savers, so the notebook view and
    every saved format (.html / .csv / .xlsx / .png) stay identical to what the
    standalone logistic workflow produces. ``binary`` selects the binary vs
    multinomial presentation path.
    """

    def __init__(self, fit_result, *, binary: bool, context: str = "report"):
        self._res = fit_result          # LogitResult or MNLogitResult
        self.binary = binary
        self.context = context
        self.df = fit_result.coef_df

    def styler(self):
        if self.binary:
            from DSSP2026.report.logistic_tables import (
                style_logit_coefficients)
            return style_logit_coefficients(
                self._res.coef_df, self._res.model, context=self.context)
        from DSSP2026.logistic_regression.multiclass import (
            style_mnlogit_coefficients)
        return style_mnlogit_coefficients(self._res, context=self.context)

    def _repr_html_(self):
        try:
            return self.styler().to_html()
        except Exception:
            return self.df.to_html(index=False)

    def __repr__(self):
        return self.df.to_string(index=False)

    def to_csv(self, path, **kwargs):
        self.df.to_csv(path, index=False, **kwargs)
        return str(path)

    def to_png(self, path, *, dpi: int = 220):
        if self.binary:
            from DSSP2026.report.logistic_tables import (
                save_logit_coefficients)
            return save_logit_coefficients(
                self._res.coef_df, self._res.model, path,
                context=self.context, dpi=dpi)
        from DSSP2026.logistic_regression.multiclass import (
            save_mnlogit_coefficients)
        return save_mnlogit_coefficients(
            self._res, path, context=self.context, dpi=dpi)

    # .html / .xlsx route through the same extension-dispatching saver.
    to_html = to_png
    to_xlsx = to_png

    def _repr_png_(self):
        import os
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        self.to_png(tmp, dpi=150)
        with open(tmp, "rb") as f:
            data = f.read()
        os.unlink(tmp)
        return data


class OddsRatioPlot:
    """Odds-ratio forest plot (the log-odds plot) for a fitted logistic model.

    Each predictor's odds ratio is drawn with its CI against the OR = 1
    no-effect line, on a log OR axis by default. Delegates to the forest-plot
    builder in ``logistic_regression`` (binary: one panel; multinomial: one
    panel per non-baseline class).
    """

    def __init__(self, fit_result, *, binary: bool, drop_intercept: bool = True,
                 log_scale: bool = True, title: Optional[str] = None):
        self._res = fit_result
        self.binary = binary
        self.drop_intercept = drop_intercept
        self.log_scale = log_scale
        self.title = title

    def _figure(self):
        if self.binary:
            kwargs = dict(drop_intercept=self.drop_intercept,
                          log_scale=self.log_scale)
            if self.title is not None:
                kwargs["title"] = self.title
            return plot_odds_ratios(self._res.coef_df, **kwargs)
        from DSSP2026.logistic_regression.multiclass import (
            plot_mnlogit_odds_ratios)
        kwargs = dict(drop_intercept=self.drop_intercept,
                      log_scale=self.log_scale)
        if self.title is not None:
            kwargs["title"] = self.title
        return plot_mnlogit_odds_ratios(self._res, **kwargs)

    def figure(self):
        return self._figure()

    def to_png(self, path, *, dpi: int = 220):
        from DSSP2026.core.figure import save_figure
        return save_figure(self._figure(), path, dpi=dpi)

    def _repr_png_(self):
        import io
        import matplotlib.pyplot as plt
        fig = self._figure()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class LogisticMixin:
    """Coefficient-table and odds-ratio (log-odds) views for a logistic study.

    Both refit the winning feature set on the supplied training frame; see the
    module docstring for why a refit is required (coefficients aren't persisted
    in report.db).
    """

    LOGISTIC_MODEL = "Logistic regression"

    def coefficients_table(self, train: pd.DataFrame, target: str, *,
                           model: Optional[str] = None,
                           context: str = "report") -> CoefficientsTable:
        """Stylized odds-ratio coefficient table for the winning logistic model.

        Refits the model's stored winning feature set on ``train`` (logistic
        coefficients aren't persisted, so a refit on the actual data is
        required), then returns a ``CoefficientsTable`` that renders inline and
        exports via ``.to_png`` / ``.to_csv`` / ``.to_html`` / ``.to_xlsx``.

        Parameters
        ----------
        train : DataFrame
            The training frame the study was fit on — must contain ``target``
            and every column in the model's stored feature list. Apply the same
            preprocessing the study used (e.g. ``standardise_numeric``) so the
            coefficients match the reported model.
        target : str
            Name of the outcome column.
        model : str, optional
            Model name in report.db. Defaults to the logistic-regression study
            ("Logistic regression"); pass a name to disambiguate if needed.
        """
        res, binary = self._refit_logistic(train, target, model)
        return CoefficientsTable(res, binary=binary, context=context)

    def odds_ratios_plot(self, train: pd.DataFrame, target: str, *,
                         model: Optional[str] = None,
                         drop_intercept: bool = True,
                         log_scale: bool = True,
                         title: Optional[str] = None) -> OddsRatioPlot:
        """Odds-ratio forest plot (log-odds plot) for the winning logistic model.

        Refits as in :meth:`coefficients_table` and returns an ``OddsRatioPlot``
        (inline render + ``.to_png``). Each predictor's odds ratio is shown with
        its 95% CI against the OR = 1 reference line; the OR axis is log-scaled
        by default (``log_scale=True``) so 0.5 and 2.0 sit symmetrically about 1.

        ``drop_intercept`` hides the intercept term (the usual choice for a
        predictor-effects plot). For a multiclass study the plot has one panel
        per non-baseline class.
        """
        res, binary = self._refit_logistic(train, target, model)
        return OddsRatioPlot(res, binary=binary, drop_intercept=drop_intercept,
                             log_scale=log_scale, title=title)

    # -- internal: locate the winning logistic config and refit it --

    def _refit_logistic(self, train, target, model):
        """Refit the winning logistic config on ``train``; return (result, binary).

        Reads the model's stored ``feature_list`` and ``detail`` from report.db,
        builds the patsy formula exactly as ``study._refit_winner`` does, and
        fits via the family's own ``fit_logit`` / ``fit_mnlogit``. The
        binary/multiclass branch follows the actual class count in ``train``
        (the stored ``detail`` is a cross-check, not the source of truth, in
        case the caller passes a differently-scoped frame).
        """
        from DSSP2026.experiment.logistic_adapter import is_binary

        model = model or self.LOGISTIC_MODEL
        features, detail = self._logistic_meta(model)

        if target not in train.columns:
            raise ValueError(
                f"target {target!r} not found in the supplied train frame.")
        missing = [c for c in features if c not in train.columns]
        if missing:
            raise ValueError(
                f"train frame is missing {len(missing)} feature column(s) the "
                f"stored model {model!r} was fit on: {missing}. Pass the same "
                "frame (and preprocessing) the study used.")

        formula = f"{target} ~ " + " + ".join(features)
        binary = is_binary(train, target)
        if detail and detail.startswith("multiclass") and binary:
            import warnings
            warnings.warn(
                f"stored model {model!r} was multiclass but the supplied train "
                "frame has only two classes for the target; refitting as "
                "binary.", stacklevel=3)

        if binary:
            from DSSP2026.logistic_regression.binary import fit_logit
            from DSSP2026.experiment.logistic_adapter import _encode_df
            enc_train, _ = _encode_df(train, target)
            res = fit_logit(enc_train, formula)
        else:
            from DSSP2026.logistic_regression.multiclass import fit_mnlogit
            res = fit_mnlogit(train, formula)
        return res, binary

    def _logistic_meta(self, model):
        """Return (feature_list, detail) for ``model`` in the selected experiment."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT feature_list, detail FROM models "
                "WHERE experiment_id=? AND model=?",
                (self.experiment_id, model)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise ValueError(
                f"model {model!r} not in experiment {self.experiment_id}. "
                f"Available models: {self.models()}.")
        if not row[0]:
            raise ValueError(
                f"model {model!r} has no stored feature_list; cannot rebuild "
                "the coefficient table.")
        features = [str(c) for c in json.loads(row[0])]
        detail = row[1]
        return features, detail
