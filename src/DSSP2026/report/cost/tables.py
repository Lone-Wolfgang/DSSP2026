"""
report/cost.py — cost-optimal model/policy selection.

``Report.cost_optimize`` sweeps every (model × policy) combination, scores each
by net benefit under the user's schedule, and returns a :class:`CostDecision`
describing the single best combination. The four policies:

- **ArgMax**     — stored argmax predictions (threshold 0.5).
- **F1**         — per-class thresholds tuned to maximise F1.
- **Youden's J** — per-class thresholds tuned to maximise Youden's J.
- **Bayes**      — expected-value decision under the schedule (no threshold).

The decision carries two display tables (each with ``show()`` / ``save()`` and
the dashboard's red/yellow/green shading) and a ``fit()`` method that refits the
winning model from the parquet sidecar and wraps it with the winning decision
layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from DSSP2026.report.cost.math import (
    fmt_currency, style_cost_table,
)

POLICIES = ("ArgMax", "F1", "Youden's J", "Bayes")
_POLICY_METRIC = {"F1": "f1", "Youden's J": "youden"}


def _text_color(rgb):
    """White on dark fills, near-black on light, by perceived luminance.

    ``rgb`` is a 3- or 4-tuple of 0..1 floats (matplotlib cmap output). Uses
    the Rec. 601 luma; threshold 0.5 splits the RdYlGn scale so the saturated
    red/green ends get white text and the pale midrange keeps dark text.
    """
    r, g, b = rgb[0], rgb[1], rgb[2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luma < 0.5 else "#1A1A1A"


# ---------------------------------------------------------------------------
# Display tables
# ---------------------------------------------------------------------------

class PolicyTable:
    """Net-benefit comparison across (model, policy) combinations.

    Columns: ``Model, Policy, Investment, Loss, Gains, Gross, Net Benefit``;
    currency columns shaded on the dashboard's symmetric RdYlGn scale.

    Default view: three rows — No Action, winner under ArgMax, winner under its
    best policy. Passing ``models`` / ``policies`` to ``show``/``save`` shows
    those permutations ranked by net benefit, No Action pinned first.
    """

    _CURRENCY = ["Investment", "Loss", "Gains", "Gross", "Net Benefit"]
    _DISPLAY_COLS = ["Model", "Policy",
                     "Investment", "Loss", "Gains", "Gross", "Net Benefit"]

    def __init__(self, rows_df, *, no_action_row, best_model, best_policy,
                 abbreviate=True):
        self._rows = rows_df.reset_index(drop=True)
        self._no_action = no_action_row
        self.best_model = best_model
        self.best_policy = best_policy
        self._abbrev = abbreviate

    def _no_action_display_row(self):
        na = self._no_action
        return {
            "Model": "No Action", "Policy": "—",
            "Investment": na["Investment"], "Loss": na["Loss"],
            "Gains": na["Gains"], "Gross": na["Gross"],
            "Net Benefit": na["Net benefit"],
        }

    def _build(self, models=None, policies=None, columns=None):
        if models is None and policies is None:
            best_argmax = self._rows[
                (self._rows["Model"] == self.best_model)
                & (self._rows["Policy"] == "ArgMax")]
            best_best = self._rows[
                (self._rows["Model"] == self.best_model)
                & (self._rows["Policy"] == self.best_policy)]
            body = pd.concat([best_argmax, best_best], ignore_index=True)
            body = body.drop_duplicates(subset=["Model", "Policy"])
        else:
            sub = self._rows
            if models is not None:
                want_m = {models} if isinstance(models, str) else set(models)
                sub = sub[sub["Model"].isin(want_m)]
            if policies is not None:
                want_p = {policies} if isinstance(policies, str) else set(policies)
                sub = sub[sub["Policy"].isin(want_p)]
            body = sub.sort_values(
                "Net Benefit", ascending=False).reset_index(drop=True)

        out = pd.concat(
            [pd.DataFrame([self._no_action_display_row()]), body],
            ignore_index=True)
        cols = self._resolve_columns(columns)
        return out[cols]

    def _resolve_columns(self, columns):
        """Validate/normalise a requested column subset (keeps given order).

        ``None`` -> the full default column list. Identifying columns (Model,
        Policy) are always retained even if omitted, pinned at the front.
        """
        if columns is None:
            return list(self._DISPLAY_COLS)
        want = [columns] if isinstance(columns, str) else list(columns)
        unknown = [c for c in want if c not in self._DISPLAY_COLS]
        if unknown:
            raise KeyError(
                f"unknown column(s) {unknown}; "
                f"choose from {self._DISPLAY_COLS}")
        keep_id = [c for c in ("Model", "Policy")
                   if c not in want]
        return keep_id + want

    @property
    def df(self):
        return self._build()

    def styler(self, *, models=None, policies=None, columns=None):
        frame = self._build(models=models, policies=policies, columns=columns)
        cur = [c for c in self._CURRENCY if c in frame.columns]
        return style_cost_table(
            frame, currency_cols=cur, abbreviate=self._abbrev)

    def show(self, *, models=None, policies=None, columns=None):
        sty = self.styler(models=models, policies=policies, columns=columns)
        try:
            from IPython.display import display
            display(sty)
        except ImportError:
            print(self._build(models=models, policies=policies,
                              columns=columns).to_string(index=False))

    def save(self, path, *, models=None, policies=None, columns=None, dpi=220):
        frame = self._build(models=models, policies=policies, columns=columns)
        cur = [c for c in self._CURRENCY if c in frame.columns]
        return _save_table_png(
            frame, path, currency_cols=cur, abbreviate=self._abbrev, dpi=dpi,
            title="Policy comparison — net benefit")


class ClassBreakdownTable:
    """Per-class economic breakdown for the winning model+policy.

    Columns: ``Class, TP, FP, FN, TP Value, FP Value, FN Value, Net`` —
    identical to the dashboard per-class breakdown. Currency columns shaded.
    """

    _CURRENCY = ["TP Value", "FP Value", "FN Value", "Net"]
    _DISPLAY_COLS = ["Class", "TP", "FP", "FN",
                     "TP Value", "FP Value", "FN Value", "Net"]

    def __init__(self, frame, *, abbreviate=True):
        self._frame = frame.reset_index(drop=True)
        self._abbrev = abbreviate

    @property
    def df(self):
        return self._frame

    def _build(self, rows=None, columns=None):
        out = self._frame
        if rows is not None:
            want = [rows] if isinstance(rows, str) else list(rows)
            want = [str(r) for r in want]
            mask = out["Class"].astype(str).isin(want)
            missing = set(want) - set(out["Class"].astype(str))
            if missing:
                raise KeyError(f"unknown class label(s) {sorted(missing)}")
            out = out[mask].reset_index(drop=True)
        cols = self._resolve_columns(columns)
        return out[cols]

    def _resolve_columns(self, columns):
        if columns is None:
            return [c for c in self._DISPLAY_COLS if c in self._frame.columns]
        want = [columns] if isinstance(columns, str) else list(columns)
        unknown = [c for c in want if c not in self._frame.columns]
        if unknown:
            raise KeyError(
                f"unknown column(s) {unknown}; "
                f"choose from {list(self._frame.columns)}")
        keep_id = ["Class"] if "Class" not in want else []
        return keep_id + want

    def styler(self, *, rows=None, columns=None):
        frame = self._build(rows=rows, columns=columns)
        cur = [c for c in self._CURRENCY if c in frame.columns]
        return style_cost_table(frame, currency_cols=cur,
                                abbreviate=self._abbrev)

    def show(self, *, rows=None, columns=None):
        try:
            from IPython.display import display
            display(self.styler(rows=rows, columns=columns))
        except ImportError:
            print(self._build(rows=rows, columns=columns)
                  .to_string(index=False))

    def save(self, path, *, rows=None, columns=None, dpi=220):
        frame = self._build(rows=rows, columns=columns)
        cur = [c for c in self._CURRENCY if c in frame.columns]
        return _save_table_png(
            frame, path, currency_cols=cur,
            abbreviate=self._abbrev, dpi=dpi, title="Per-class breakdown")


def _save_table_png(frame, path, *, currency_cols, abbreviate, dpi, title):
    """Render a styled cost table to PNG via matplotlib (no browser needed).

    Sized to fit the data rather than a fixed canvas: each column gets a width
    proportional to its widest cell, the figure is the sum of those widths, and
    ``bbox_inches="tight"`` trims the margins. The result drops cleanly onto a
    13.33×7.5in widescreen PowerPoint slide without manual rescaling.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from DSSP2026.core.style import ATT_COLORS

    disp = frame.copy()
    for c in currency_cols:
        if c in disp.columns:
            disp[c] = disp[c].map(
                lambda v: fmt_currency(v, abbreviate=abbreviate))

    cmap = mpl.colormaps["RdYlGn"]
    cur = [c for c in currency_cols if c in frame.columns]
    vals = frame[cur].to_numpy(float) if cur else np.zeros((1, 1))
    max_abs = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
    if not np.isfinite(max_abs) or max_abs == 0.0:
        max_abs = 1.0
    norm = mpl.colors.Normalize(vmin=-max_abs, vmax=max_abs)

    n_rows, n_cols = disp.shape

    # --- content-proportional column widths -------------------------------
    # Width of each column = widest of its header / cells, measured in chars,
    # converted to inches. Clamped so a stray long cell can't blow out the
    # layout and a 2-char column ("TP") still reads.
    CHAR_IN = 0.082         # ~inches per character at fontsize 10
    PAD_IN = 0.22           # left+right cell padding
    MIN_COL, MAX_COL = 0.55, 2.6
    str_cells = disp.astype(str)
    col_w = []
    for j, name in enumerate(disp.columns):
        widest = max([len(str(name))] +
                     [len(s) for s in str_cells.iloc[:, j]])
        w = widest * CHAR_IN + PAD_IN
        col_w.append(min(max(w, MIN_COL), MAX_COL))
    total_w = sum(col_w)
    # Keep within a slide's usable width (~10in leaves generous margins).
    if total_w > 10.0:
        scale = 10.0 / total_w
        col_w = [w * scale for w in col_w]
        total_w = 10.0
    col_frac = [w / total_w for w in col_w]

    row_h_in = 0.42
    fig_w = total_w
    fig_h = row_h_in * (n_rows + 1) + 0.6   # +1 header, +title room
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.axis("off")

    tbl = ax.table(cellText=disp.values, colLabels=list(disp.columns),
                   colWidths=col_frac, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    navy = ATT_COLORS.get("navy", "#002A5C")
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(navy)
            cell.set_text_props(color="white", weight="bold")
        else:
            col_name = disp.columns[c]
            if col_name in cur:
                raw = frame.iloc[r - 1][col_name]
                try:
                    rgba = cmap(norm(float(raw)))
                    cell.set_facecolor(rgba)
                    cell.set_text_props(color=_text_color(rgba))
                except (TypeError, ValueError):
                    pass
    ax.set_title(title, fontsize=12, color=navy, weight="bold", pad=12)

    from DSSP2026.core.figure import save_figure
    fig.set_layout_engine("tight")
    try:
        out = save_figure(fig, path, dpi=dpi, bbox_inches="tight",
                          transparent=True)
    except TypeError:
        # save_figure doesn't forward **kwargs to savefig; the per-patch alpha
        # set above still yields a transparent background.
        out = save_figure(fig, path, dpi=dpi)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CostConfusion — cost-aware confusion matrix
# ---------------------------------------------------------------------------

class CostConfusion:
    """A confusion matrix shaded by per-cell cost.

    Cells are coloured on a symmetric ``-max_abs … +max_abs`` RdYlGn scale by the
    per-cell dollar value under the chosen ``viewpoint``:

    - ``viewpoint="net_benefit"`` (default) — diagonal credits the avoided miss
      (positive spin).
    - ``viewpoint="gross"`` — diagonal shows the honest intervention cost (can be
      negative even on correct predictions).

    Cell text is controlled by ``text``: ``"counts"`` (default, abbreviated),
    ``"currency"`` (the per-cell dollar total = count × unit value), or
    ``"both"`` (count over currency).

    ``styler()`` / ``show()`` render the shaded grid (no colorbar); ``save()``
    writes a matplotlib PNG with a currency colorbar on the right.
    """

    def __init__(self, counts_df, unit_matrix, *, class_labels, viewpoint,
                 text="counts", model=None, policy=None):
        self.counts = counts_df                 # (K,K) int DataFrame, true x pred
        self.unit = np.asarray(unit_matrix, float)
        self.class_labels = [str(c) for c in class_labels]
        self.viewpoint = viewpoint
        self.text = text
        self.model = model
        self.policy = policy
        # Per-cell dollar total = count * unit value.
        self.value = self.counts.to_numpy(float) * self.unit

    def _cell_text(self, i, j):
        from DSSP2026.report.cost.math import fmt_count, fmt_currency
        cnt = int(self.counts.iat[i, j])
        val = self.value[i, j]
        if self.text == "counts":
            return fmt_count(cnt)
        if self.text == "currency":
            return fmt_currency(val)
        if self.text == "both":
            return f"{fmt_count(cnt)}\n{fmt_currency(val)}"
        raise ValueError(
            f"text must be 'counts', 'currency', or 'both'; got {self.text!r}.")

    def _select_idx(self, rows):
        """Resolve a row/col label selection to positional indices.

        Square submatrix: the same labels index both axes. ``None`` -> all.
        Order follows the request; unknown labels raise.
        """
        if rows is None:
            return list(range(len(self.class_labels)))
        want = [rows] if isinstance(rows, str) else list(rows)
        want = [str(r) for r in want]
        pos = {lab: i for i, lab in enumerate(self.class_labels)}
        missing = [w for w in want if w not in pos]
        if missing:
            raise KeyError(
                f"unknown class label(s) {missing}; "
                f"choose from {self.class_labels}")
        return [pos[w] for w in want]

    def df(self, *, rows=None):
        """Cell-text DataFrame (true rows × predicted cols).

        ``rows`` selects a square submatrix by class label (same labels on both
        axes). Note this is a method (call ``.df()``); selection needs an arg.
        """
        idx = self._select_idx(rows)
        labels = [self.class_labels[i] for i in idx]
        data = [[self._cell_text(i, j) for j in idx] for i in idx]
        out = pd.DataFrame(data, index=labels, columns=labels)
        out.index.name = "True"
        out.columns.name = "Predicted"
        return out

    def _max_abs(self):
        m = float(np.nanmax(np.abs(self.value))) if self.value.size else 0.0
        return m if (np.isfinite(m) and m > 0) else 1.0

    def styler(self, *, rows=None):
        """pandas Styler: shaded grid, no colorbar (for notebook display).

        ``rows`` selects a square submatrix by label; the colour scale stays
        fixed to the full matrix so submatrices remain comparable.
        """
        import matplotlib as mpl
        idx = self._select_idx(rows)
        text_df = self.df(rows=rows)
        max_abs = self._max_abs()          # full-matrix scale, fixed
        cmap = mpl.colormaps["RdYlGn"]
        norm = mpl.colors.Normalize(vmin=-max_abs, vmax=max_abs)

        def _style(_):
            css = pd.DataFrame("", index=text_df.index, columns=text_df.columns)
            for a, i in enumerate(idx):
                for b, j in enumerate(idx):
                    rgba = cmap(norm(self.value[i, j]))
                    r, g, bch, _ = rgba
                    # Text colour by fill luminance: white on saturated
                    # red/green ends, dark on the pale midrange. white-space:
                    # pre-line makes the "both" newline render in HTML.
                    css.iat[a, b] = (
                        f"background-color: rgb({int(r*255)},{int(g*255)},"
                        f"{int(bch*255)}); color: {_text_color(rgba)}; "
                        f"font-weight: 600; text-align: center; "
                        f"white-space: pre-line;")
            return css

        return text_df.style.apply(_style, axis=None)

    def show(self, *, rows=None):
        try:
            from IPython.display import display
            display(self.styler(rows=rows))
        except ImportError:
            print(self.df(rows=rows).to_string())

    def save(self, path, *, rows=None, dpi=220):
        """Matplotlib PNG: shaded grid + currency colorbar on the right.

        ``rows`` selects a square submatrix by label; the colour scale (and
        colorbar) stays fixed to the full matrix.
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from DSSP2026.core.style import ATT_COLORS
        from DSSP2026.core.figure import save_figure

        idx = self._select_idx(rows)
        labels = [self.class_labels[i] for i in idx]
        K = len(labels)
        sub_val = self.value[np.ix_(idx, idx)]
        max_abs = self._max_abs()          # full-matrix scale, fixed
        cmap = mpl.colormaps["RdYlGn"]
        norm = mpl.colors.Normalize(vmin=-max_abs, vmax=max_abs)
        navy = ATT_COLORS.get("navy", "#002A5C")

        fig, ax = plt.subplots(figsize=(1.2 + 1.1 * K, 1.0 + 1.0 * K))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.imshow(sub_val, cmap=cmap, norm=norm, aspect="equal")

        for a, i in enumerate(idx):
            for b, j in enumerate(idx):
                ax.text(b, a, self._cell_text(i, j), ha="center", va="center",
                        fontsize=10, color=_text_color(cmap(norm(self.value[i, j]))))

        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11, color=navy)
        ax.set_ylabel("True", fontsize=11, color=navy)
        ttl = "Cost-aware confusion"
        if self.model:
            ttl += f" — {self.model}"
            if self.policy:
                ttl += f" ({self.policy})"
        ttl += f"\nshaded by {self.viewpoint.replace('_', ' ')}"
        ax.set_title(ttl, fontsize=12, color=navy, weight="bold")

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        from DSSP2026.report.cost.math import fmt_currency
        cbar.ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _pos: fmt_currency(v)))
        cbar.ax.tick_params(labelsize=9)

        fig.tight_layout()
        try:
            out = save_figure(fig, path, dpi=dpi, transparent=True)
        except TypeError:
            out = save_figure(fig, path, dpi=dpi)
        plt.close(fig)
        return out


# ---------------------------------------------------------------------------
# CostDecision
# ---------------------------------------------------------------------------

@dataclass
class CostDecision:
    """The winning (model, policy) combination from a cost sweep.

    Attributes
    ----------
    model_type : str
        Winning model name (e.g. "XGBoost", "Ensemble").
    policy : str
        Winning policy ("ArgMax", "F1", "Youden's J", or "Bayes").
    threshold : float | dict | None
        ArgMax → 0.5; F1/Youden → ``{class: cutoff}``; Bayes → None.
    policy_table : PolicyTable
        Net-benefit comparison across combinations (show()/save()).
    class_breakdown : ClassBreakdownTable
        Per-class breakdown for the winner (show()/save()).
    net_benefit : float
        The winner's net benefit (relative to No Action).

    ``fit()`` refits the winning model from the training parquet and wraps it
    with the winning decision layer.
    """
    model_type: str
    policy: str
    threshold: object
    policy_table: "PolicyTable"
    class_breakdown: "ClassBreakdownTable"
    net_benefit: float
    _report: object = field(default=None, repr=False)
    _schedule: object = field(default=None, repr=False)
    _class_order: object = field(default=None, repr=False)
    _allow_ensemble: bool = field(default=False, repr=False)
    _decisions: object = field(default=None, repr=False)
    _y_true: object = field(default=None, repr=False)

    def fit(self):
        """Refit the winning model from the parquet sidecar and attach the
        winning decision layer. Returns a deployable predictor.

        Requires the training parquet (``ReportBase.load_train_data``). For an
        Ensemble winner, refits every member and wraps them in an averaging
        predictor.
        """
        from DSSP2026.report.cost.fit import fit_cost_model
        return fit_cost_model(
            self._report, self.model_type, self.policy, self.threshold,
            self._schedule, self._class_order,
            allow_ensemble=self._allow_ensemble)

    def cost_confusion(self, *, viewpoint="net_benefit", text="counts"):
        """Cost-aware confusion matrix for the winning model+policy.

        Parameters
        ----------
        viewpoint : {"net_benefit", "gross"}
            Controls the per-cell value used for shading. ``net_benefit``
            credits the avoided miss on the diagonal (positive spin); ``gross``
            shows the honest intervention cost (diagonal can be negative even on
            correct predictions). Off-diagonal cells are identical either way.
        text : {"counts", "currency", "both"}
            Cell contents: abbreviated counts (default), the per-cell dollar
            total (count × unit value), or both stacked.

        Returns
        -------
        CostConfusion
            ``.styler()`` / ``.show()`` render the shaded grid; ``.save(path)``
            writes a PNG with a currency colorbar.
        """
        from DSSP2026.report.cost.math import (
            confusion_from_decisions, cost_unit_matrix)
        co = self._class_order
        cm = confusion_from_decisions(self._y_true, self._decisions, co)
        unit = cost_unit_matrix(co, self._schedule, viewpoint=viewpoint)
        # Render with friendly display names (id2label) while keeping the count
        # and value matrices aligned to the raw class order.
        relabel = getattr(self._report, "_relabel", None)
        display_labels = ([relabel(c) for c in co] if relabel else list(co))
        return CostConfusion(
            cm, unit, class_labels=display_labels, viewpoint=viewpoint,
            text=text, model=self.model_type, policy=self.policy)