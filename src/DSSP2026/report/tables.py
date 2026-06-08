from __future__ import annotations

import numpy as np
import pandas as pd

from DSSP2026.report.plots import ConfusionPlot

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
        # Generic path: format only columns whose non-null values are all
        # numeric floats. Guard with a try/except so a rogue string value
        # never crashes notebook display.
        def _is_float_col(col):
            if pd.api.types.is_float_dtype(col):
                return True
            # object dtype that actually holds floats (e.g. from JSON round-trip)
            non_null = col.dropna()
            if len(non_null) == 0:
                return False
            try:
                non_null.astype(float)
                return not pd.api.types.is_integer_dtype(col)
            except (ValueError, TypeError):
                return False

        def _safe_float_fmt(v):
            if pd.isna(v):
                return ""
            try:
                return f"{float(v):,.4f}"
            except (ValueError, TypeError):
                return str(v)

        float_cols = [c for c in self.df.columns if _is_float_col(self.df[c])]
        fmt = {c: _safe_float_fmt for c in float_cols}
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

        def _safe_float_fmt(v):
            if pd.isna(v):
                return ""
            try:
                return f"{float(v):,.4f}"
            except (ValueError, TypeError):
                return str(v)

        float_cols = [c for c in self.df.columns
                      if pd.api.types.is_float_dtype(self.df[c])]
        col_fmts = {c: _safe_float_fmt for c in float_cols}
        highlight = None
        if self.best_by and self.best_by in self.df.columns:
            highlight = int(self.df[self.best_by].idxmax())
        return save_generic_table_png(
            self.df, path, title=self.title, col_fmts=col_fmts,
            highlight_row=highlight, context=self.context, dpi=dpi, **kwargs)
