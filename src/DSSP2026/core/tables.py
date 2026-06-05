"""
core/tables.py — model-agnostic table rendering.

Responsibilities
----------------
- File-format dispatch (save a DataFrame/Styler to csv/xlsx/html/png)
- Generic matplotlib PNG table renderer
- Cross-family stylers that aren't tied to one model (model comparison,
  tree metrics)
"""

from typing import Optional, Union
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from DSSP2026.core.style import ATT_COLORS, att_table_styles, _SIZE_PRESETS


_R2_VARIANTS = ("R²", "R2")


def r2_col(df: pd.DataFrame) -> str:
    """Return whichever R² column name is present in *df*.

    Accepts the superscript unicode variant (``R²``) and the plain-ASCII
    fallback (``R2``). Raises KeyError with a helpful message if neither is
    found.
    """
    for name in _R2_VARIANTS:
        if name in df.columns:
            return name
    raise KeyError(
        f"DataFrame has no R² column. Expected one of {_R2_VARIANTS}; "
        f"got columns: {list(df.columns)}"
    )


_CAPTION_STYLE = [
    {"selector": "caption",
     "props": [("caption-side", "bottom"),
               ("padding-top", "10px"),
               ("color", ATT_COLORS["gray_700"]),
               ("font-style", "italic"),
               ("text-align", "left")]},
]


def format_pvalue(p):
    try:
        return "< 0.001" if p < 0.001 else f"{p:.3f}"
    except TypeError:
        return str(p)


# ---------------------------------------------------------------------------
# File-format dispatch
# ---------------------------------------------------------------------------

def save_table_by_extension(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    styler=None,
    png_renderer=None,
    **kwargs,
) -> str:
    """Save a table, dispatching on file extension.

    csv/xlsx → pandas writers. html → styler HTML if a styler is given, else
    df.to_html. png → png_renderer(df, path, **kwargs) if provided, else the
    generic renderer in this module.
    """
    path_lower = str(path).lower()
    if path_lower.endswith(".csv"):
        df.to_csv(path, index=False, **kwargs)
    elif path_lower.endswith((".xlsx", ".xls")):
        df.to_excel(path, index=False, **kwargs)
    elif path_lower.endswith(".html"):
        if styler is not None:
            with open(path, "w", encoding=kwargs.pop("encoding", "utf-8")) as f:
                f.write(styler.to_html())
        else:
            df.to_html(path, index=False, **kwargs)
    elif path_lower.endswith(".png"):
        renderer = png_renderer or save_generic_table_png
        renderer(df, path, **kwargs)
    else:
        raise ValueError("path must end with .csv, .xlsx, .xls, .html, or .png")
    return str(path)


# ---------------------------------------------------------------------------
# Generic matplotlib PNG renderer
# ---------------------------------------------------------------------------

def save_generic_table_png(
    df,
    path,
    *,
    title="",
    col_fmts=None,
    round_digits=3,
    highlight_row=None,
    context="report",
    dpi=220,
):
    """Render a DataFrame as a styled PNG table (header band, zebra rows)."""
    sizes = _SIZE_PRESETS[context]
    font_size = sizes["table"]
    row_height = 0.52 if context == "report" else 0.65

    display = df.copy()
    # Explicit per-column formatters take priority.
    explicit = set()
    if col_fmts:
        for col, fmt in col_fmts.items():
            if col in display.columns:
                display[col] = display[col].map(fmt)
                explicit.add(col)
    
        # Auto-round any remaining numeric column for display. Integers are left
    # alone (no spurious ".000"); floats are rounded and rendered with a
    # consistent number of decimals.
    if round_digits is not None:
        for col in display.columns:
            if col in explicit:
                continue
            s = display[col]
            if pd.api.types.is_float_dtype(s):
                display[col] = s.map(
                    lambda v: f"{v:.{round_digits}f}" if pd.notna(v) else v
                )

    cols = list(display.columns)
    n_rows, n_cols = display.shape

    # Per-column width in "characters". Headers are bold, which renders ~18%
    # wider per glyph than the regular cell text the char->inch factor is tuned
    # for, so the header contribution is scaled up before taking the max against
    # the data. The flat +3 is padding (cell margins on both sides).
    HEADER_BOLD_FACTOR = 1.18
    col_widths_chars = [
        max(
            len(str(col)) * HEADER_BOLD_FACTOR,
            *(len(str(v)) for v in display[col].values),
        ) + 3
        for col in cols
    ]
    total_chars = sum(col_widths_chars)
    char_inch = 0.105 + 0.006 * (font_size - 11)
    fig_w = max(7.0, total_chars * char_inch + 0.6)
    fig_h = (n_rows + 1) * row_height + 0.9
    col_fracs = [w / total_chars for w in col_widths_chars]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    cell_colors = []
    for r in range(n_rows):
        if r == highlight_row:
            row_color = ATT_COLORS["pale_sky"]
        elif r % 2 == 1:
            row_color = ATT_COLORS["gray_100"]
        else:
            row_color = ATT_COLORS["white"]
        cell_colors.append([row_color] * n_cols)

    tbl = ax.table(
        cellText=display.values.tolist(),
        colLabels=cols,
        cellColours=cell_colors,
        colWidths=col_fracs,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0, 0.12, 1, 0.88],
    )
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
            if r - 1 == highlight_row:
                cell.set_text_props(weight="bold")

    if title:
        fig.text(0.5, 0.04, title, ha="center", va="center",
                 fontsize=font_size - 1, style="italic",
                 color=ATT_COLORS["gray_700"])

    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=ATT_COLORS["white"])
    plt.close(fig)
    return str(path)


def style_classification_report(report_df: pd.DataFrame, *, context: str = "report"):
    summary_rows = {"accuracy", "macro avg", "weighted avg"}

    fmt = {"Precision": "{:,.3f}", "Recall": "{:,.3f}",
           "F1": "{:,.3f}", "Support": "{:,.0f}"}

    def _emphasize_summary(row):
        if str(row.get("Class", "")) in summary_rows:
            return [f"background-color: {ATT_COLORS['gray_100']}; font-style: italic;"
                    for _ in row]
        return ["" for _ in row]

    return (
        report_df.style
            .set_table_styles(att_table_styles(context=context))
            .hide(axis="index")
            .format(fmt, na_rep="—")
            .apply(_emphasize_summary, axis=1)
            .set_caption("Classification report — precision / recall / F1 by class")
            .set_table_styles(_CAPTION_STYLE, overwrite=False)
    )


def save_classification_report_png(report_df: pd.DataFrame, path, *,
                                   context: str = "report", dpi: int = 220) -> str:
    fmt = {
        "Precision": lambda v: "—" if pd.isna(v) else f"{v:,.3f}",
        "Recall": lambda v: "—" if pd.isna(v) else f"{v:,.3f}",
        "F1": lambda v: "—" if pd.isna(v) else f"{v:,.3f}",
        "Support": lambda v: "—" if pd.isna(v) else f"{int(v):,}",
    }
    return save_generic_table_png(
        report_df, path,
        title="Classification report — precision / recall / F1 by class",
        col_fmts={k: v for k, v in fmt.items() if k in report_df.columns},
        context=context, dpi=dpi)


def style_classification_comparison(comparison_df: pd.DataFrame, *,
                                    best_by: str = "F1", context: str = "report"):
    metric_cols = [c for c in comparison_df.columns if c != "Model"]
    if best_by not in comparison_df.columns:
        best_by = metric_cols[-1] if metric_cols else None
    best_idx = comparison_df[best_by].idxmax() if best_by else None

    fmt = {c: "{:,.4f}" for c in metric_cols}

    def _highlight_best(row):
        if best_idx is not None and row.name == best_idx:
            return [f"background-color: {ATT_COLORS['pale_sky']}; font-weight: bold;"
                    for _ in row]
        return ["" for _ in row]

    return (
        comparison_df.style
            .set_table_styles(att_table_styles(context=context))
            .hide(axis="index")
            .format(fmt)
            .apply(_highlight_best, axis=1)
            .set_caption(f"Classifier comparison — highlighted row = best by {best_by}")
            .set_table_styles(_CAPTION_STYLE, overwrite=False)
    )


def save_classification_comparison_png(comparison_df: pd.DataFrame, path, *,
                                       best_by: str = "F1", context: str = "report",
                                       dpi: int = 220) -> str:
    metric_cols = [c for c in comparison_df.columns if c != "Model"]
    if best_by not in comparison_df.columns:
        best_by = metric_cols[-1] if metric_cols else None
    highlight = int(comparison_df[best_by].idxmax()) if best_by else None
    col_fmts = {c: "{:,.4f}".format for c in metric_cols}
    return save_generic_table_png(
        comparison_df, path,
        title=f"Classifier comparison — highlighted row = best by {best_by}",
        col_fmts=col_fmts, highlight_row=highlight, context=context, dpi=dpi)


def style_metrics_table(metrics_df: pd.DataFrame, *, context: str = "report",
                        caption: str = "Model performance"):
    fmt = {"MSPE": "{:,.3f}", "R2": "{:,.3f}", "R²": "{:,.3f}"}
    return (
        metrics_df.style
            .set_table_styles(att_table_styles(context=context))
            .hide(axis="index")
            .format(fmt, na_rep="")
            .set_caption(caption)
            .set_table_styles(_CAPTION_STYLE, overwrite=False)
    )


# ---------------------------------------------------------------------------
# Cross-family stylers
# ---------------------------------------------------------------------------

def style_model_comparison(comparison_df: pd.DataFrame, *, context: str = "report"):
    """Styler for the MSPE/R² model-comparison table; best (min MSPE) row highlighted."""
    best_idx = comparison_df["MSPE"].idxmin()
    r2 = r2_col(comparison_df)

    def _highlight_best(row):
        if row.name == best_idx:
            return [f"background-color: {ATT_COLORS['pale_sky']}; font-weight: bold;"
                    for _ in row]
        return ["" for _ in row]

    fmt = {"MSPE": "{:,.4f}", r2: "{:,.4f}"}

    return (
        comparison_df.style
            .set_table_styles(att_table_styles(context=context))
            .hide(axis="index")
            .format(fmt)
            .apply(_highlight_best, axis=1)
            .set_caption("60/40 External CV — lower MSPE is better; highlighted row = best model")
            .set_table_styles(_CAPTION_STYLE, overwrite=False)
    )


def save_model_comparison_table_png(
    comparison_df: pd.DataFrame,
    path: Union[str, Path],
    *,
    context: str = "report",
    dpi: int = 220,
) -> str:
    best_idx = int(comparison_df["MSPE"].idxmin())
    r2 = r2_col(comparison_df)
    fmt = {"MSPE": "{:,.4f}".format, r2: "{:,.4f}".format}
    return save_generic_table_png(
        comparison_df, path,
        title="60/40 External CV — highlighted row = best model by MSPE",
        col_fmts=fmt,
        highlight_row=best_idx,
        context=context,
        dpi=dpi,
    )