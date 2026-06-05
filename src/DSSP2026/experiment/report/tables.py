from __future__ import annotations

import numpy as np
import pandas as pd

from DSSP2026.experiment.report.plots import ConfusionPlot

class ReportTable:
    """A DataFrame result that renders readably and exports to CSV / PNG.

    Wrapping the frame (rather than returning a bare DataFrame) lets results
    chain — ``report.compare_models().to_png(...)`` — while still behaving like a
    DataFrame in a notebook (``_repr_html_`` and ``__repr__`` delegate to it, and
    ``.df`` exposes the underlying frame).
    """

    def __init__(self, df: pd.DataFrame, *, best_by: str = "F1",
                 title: str = "", context: str = "report"):
        self.df = df.reset_index(drop=True)
        self.best_by = best_by if best_by in df.columns else None
        self.title = title
        self.context = context

    # -- notebook display: a clean ASCII/HTML table --
    def __repr__(self):
        return self.df.to_string(index=False)

    def _repr_html_(self):
        try:
            return self.styler().to_html()
        except Exception:
            return self.df.to_html(index=False)

    def styler(self):
        """AT&T-styled pandas Styler (best row highlighted by ``best_by``).

        Uses the shared comparison styler when the table is a model comparison
        (has a "Model" column and a highlightable best row). For other tables
        (e.g. a per-class report keyed by "Class", or a cross-model per-class
        view), styles directly: AT&T table chrome, numeric-only float format, no
        row highlight.
        """
        from DSSP2026.core.style import att_table_styles
        if "Model" in self.df.columns and self.best_by:
            from DSSP2026.core.tables import style_classification_comparison
            return style_classification_comparison(
                self.df, best_by=self.best_by, context=self.context)
        # Generic path: format only genuinely-numeric, non-integer columns.
        float_cols = [c for c in self.df.columns
                      if pd.api.types.is_float_dtype(self.df[c])]
        fmt = {c: (lambda v: "" if pd.isna(v) else f"{v:,.4f}") for c in float_cols}
        return (self.df.style
                .set_table_styles(att_table_styles(context=self.context))
                .hide(axis="index")
                .format(fmt))

    def to_csv(self, path, **kwargs):
        """Write the table to CSV. Returns the path."""
        self.df.to_csv(path, index=False, **kwargs)
        return str(path)

    def to_png(self, path, *, dpi: int = 220, **kwargs):
        """Render the table to a PNG via the package's generic table renderer."""
        from DSSP2026.core.tables import save_generic_table_png
        # Format genuine floats to 4dp; leave integer-like columns (e.g. Support)
        # as-is so counts don't render as 180.0000.
        float_cols = [c for c in self.df.columns
                      if pd.api.types.is_float_dtype(self.df[c])]
        col_fmts = {c: (lambda v: "" if pd.isna(v) else f"{v:,.4f}")
                    for c in float_cols}
        highlight = None
        if self.best_by and self.best_by in self.df.columns:
            highlight = int(self.df[self.best_by].idxmax())
        return save_generic_table_png(
            self.df, path, title=self.title, col_fmts=col_fmts,
            highlight_row=highlight, context=self.context, dpi=dpi, **kwargs)

class PolicyCostTable:
    """Policy-comparison cost table rendered as a STYLED TABLE (not a heatmap).

    Built by :meth:`CostDecision.policy_table`. One row per decision policy
    (Do Nothing, Always Intervene, Baseline Model, Cost Optimized); columns are
    count/money line items. All cost figures are NEGATIVE (money out). Renders
    through the project's standard table machinery — AT&T deep-blue header band,
    gray gridlines, and money cells COLOR-SCALED on the ``att_diverging`` ramp.
    Each money cell is shaded by its value on a symmetric ``-max_abs..max_abs``
    domain (costs negative, net benefits positive, so the ramp midpoint sits at
    break-even). Count columns keep the plain zebra background.

    ``df`` is the underlying frame (already labeled with the caller's column
    names); ``money_cols`` lists the dollar-valued columns; ``count_cols`` the
    integer ones.
    """

    def __init__(self, df, *, money_cols, count_cols, policy_col="Policy",
                 title="", context="report"):
        self.df = df.reset_index(drop=True)
        self.money_cols = [c for c in money_cols if c in self.df.columns]
        self.count_cols = [c for c in count_cols if c in self.df.columns]
        self.policy_col = policy_col
        self.title = title
        self.context = context

    # -- color scale over money cells --
    def _color_norm(self):
        """Symmetric Normalize from -max_abs..max_abs over all money cells.

        Costs are negative, net benefits positive, so a symmetric domain centers
        the att_diverging ramp's midpoint at break-even (zero).
        """
        from matplotlib.colors import Normalize
        vals = [float(self.df.iloc[r][c])
                for c in self.money_cols for r in range(len(self.df))]
        max_abs = max((abs(v) for v in vals), default=1.0) or 1.0
        return Normalize(vmin=-max_abs, vmax=max_abs)

    def _cell_rgba(self, v, norm):
        from matplotlib import colormaps
        return colormaps["att_diverging"](norm(float(v)))

    @property
    def table(self) -> pd.DataFrame:
        """The underlying policy x line-item frame (for inline / CSV use)."""
        cols = [self.policy_col] + self.count_cols + self.money_cols
        return self.df[[c for c in cols if c in self.df.columns]]

    @staticmethod
    def _rgba_to_css(rgba):
        r, g, b, _ = rgba
        return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

    @staticmethod
    def _text_color(rgba):
        lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
        return "white" if lum < 0.5 else "black"

    # -- pandas Styler (HTML / notebook) --
    def styler(self):
        from DSSP2026.core.style import att_table_styles
        disp = self.table
        norm = self._color_norm()

        def _shade(col):
            out = []
            for v in col:
                rgba = self._cell_rgba(v, norm)
                out.append(f"background-color: {self._rgba_to_css(rgba)}; "
                           f"color: {self._text_color(rgba)};")
            return out

        money = ConfusionPlot._fmt_money
        fmt = {c: (lambda v, _m=money: _m(v)) for c in self.money_cols}
        fmt.update({c: (lambda v: f"{int(round(float(v))):,}")
                    for c in self.count_cols})
        sty = (disp.style
               .set_table_styles(att_table_styles(context=self.context))
               .hide(axis="index")
               .format(fmt, na_rep="—"))
        if self.money_cols:
            sty = sty.apply(_shade, subset=self.money_cols)
        if self.title:
            sty = sty.set_caption(self.title)
        return sty

    def _repr_html_(self):
        return self.styler().to_html()

    # -- matplotlib PNG (real ax.table grid, money cells color-scaled) --
    def _figure(self):
        import matplotlib.pyplot as plt
        from DSSP2026.core.style import ATT_COLORS, _SIZE_PRESETS
        sizes = _SIZE_PRESETS[self.context]
        font_size = sizes["table"]
        row_height = 0.52 if self.context == "report" else 0.65

        disp = self.table.copy()
        money_fmt = ConfusionPlot._fmt_money
        for c in self.money_cols:
            disp[c] = self.table[c].map(money_fmt)
        for c in self.count_cols:
            disp[c] = self.table[c].map(lambda v: f"{int(round(float(v))):,}")

        cols = list(disp.columns)
        n_rows, n_cols = disp.shape
        norm = self._color_norm()

        HEADER_BOLD_FACTOR = 1.18
        col_widths = [
            max(len(str(col)) * HEADER_BOLD_FACTOR,
                *(len(str(v)) for v in disp[col].values)) + 3
            for col in cols]
        total = sum(col_widths)
        char_inch = 0.105 + 0.006 * (font_size - 11)
        fig_w = max(7.0, total * char_inch + 0.6)
        fig_h = (n_rows + 1) * row_height + 0.9
        col_fracs = [w / total for w in col_widths]

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        # Cell background colors: money cols scaled by att_sequential, others zebra.
        money_set = set(self.money_cols)
        text_colors = {}
        cell_colors = []
        for r in range(n_rows):
            zebra = ATT_COLORS["gray_100"] if r % 2 == 1 else ATT_COLORS["white"]
            row_colors = []
            for ci, col in enumerate(cols):
                if col in money_set:
                    rgba = self._cell_rgba(self.table.iloc[r][col], norm)
                    row_colors.append(rgba)
                    text_colors[(r + 1, ci)] = self._text_color(rgba)
                else:
                    row_colors.append(zebra)
            cell_colors.append(row_colors)

        tbl = ax.table(
            cellText=disp.values.tolist(), colLabels=cols,
            cellColours=cell_colors, colWidths=col_fracs,
            cellLoc="center", colLoc="center", loc="upper left",
            bbox=[0, 0.12, 1, 0.88])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(font_size)
        for (r, c), cell in tbl.get_celld().items():
            cell.PAD = 0.04
            if r == 0:
                cell.set_facecolor(ATT_COLORS["deep_blue"])
                cell.set_text_props(color="white", weight="bold")
                cell.set_edgecolor("white")
            else:
                cell.set_edgecolor(ATT_COLORS["gray_300"])
                if (r, c) in text_colors:
                    cell.set_text_props(color=text_colors[(r, c)])

        if self.title:
            fig.text(0.5, 0.04, self.title, ha="center", va="center",
                     fontsize=font_size - 1, style="italic",
                     color=ATT_COLORS["gray_700"])
        return fig

    def figure(self):
        return self._figure()

    def to_png(self, path, *, dpi: int = 220):
        from DSSP2026.core.figure import save_figure
        return save_figure(self._figure(), path, dpi=dpi)

    def to_html(self, path, **kwargs):
        html = self.styler().to_html(**kwargs)
        with open(path, "w") as f:
            f.write(html)
        return str(path)

    def to_csv(self, path, **kwargs):
        self.table.to_csv(path, index=False, **kwargs)
        return str(path)

    def _repr_png_(self):
        import tempfile, os
        fd, tmp = tempfile.mkstemp(suffix=".png"); os.close(fd)
        self.to_png(tmp, dpi=150)
        with open(tmp, "rb") as f:
            data = f.read()
        os.unlink(tmp)
        return data
