"""
regression/tables.py — presentation-quality tables for OLS regression.

Coefficient tables, model-comparison tables, and prediction summaries —
HTML Stylers plus matplotlib PNG renderers.
"""

import textwrap
from typing import Optional, Union
from pathlib import Path

import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper

from modules.core.style import ATT_COLORS, att_table_styles, _SIZE_PRESETS
from modules.core.tables import save_table_by_extension
from modules.regression.fit import _detect_extrapolation


_CAPTION_STYLE = [
    {"selector": "caption",
     "props": [("caption-side", "bottom"),
               ("padding-top", "10px"),
               ("color", ATT_COLORS["gray_700"]),
               ("font-style", "italic"),
               ("text-align", "left")]},
]


# ---------------------------------------------------------------------------
# Coefficient table
# ---------------------------------------------------------------------------

def style_ols_coefficients(coef_df: pd.DataFrame, model: RegressionResultsWrapper,
                           *, context: str = "report"):
    """Styler for the OLS coefficient table; significant p-values highlighted."""
    p_col = "p-value"

    def _format_p(p):
        return "< 0.001" if p < 0.001 else f"{p:.3f}"

    def _highlight_sig(val):
        try:
            if val < 0.05:
                return f"background-color: {ATT_COLORS['pale_sky']}; font-weight: bold;"
        except TypeError:
            pass
        return ""

    fmt = {
        "Coefficient": "{:,.4f}",
        "Std Error":   "{:,.4f}",
        "t":           "{:,.2f}",
        p_col:         _format_p,
    }
    for col in coef_df.columns:
        if col.startswith("CI "):
            fmt[col] = "{:,.4f}"

    caption = (
        f"OLS coefficients — N = {int(model.nobs)}, "
        f"R² = {model.rsquared:.3f}, Adj. R² = {model.rsquared_adj:.3f}, "
        f"F = {model.fvalue:.2f} (p = {model.f_pvalue:.3g})"
    )

    return (
        coef_df.style
            .set_table_styles(att_table_styles(context=context))
            .hide(axis="index")
            .format(fmt)
            .map(_highlight_sig, subset=[p_col])
            .set_caption(caption)
            .set_table_styles(_CAPTION_STYLE, overwrite=False)
    )


def save_ols_coefficients_png(coef_df: pd.DataFrame, model: RegressionResultsWrapper,
                              path: Union[str, Path], *, context: str = "report") -> str:
    """Render the coefficient table as a PNG with a fit-stats caption."""
    import matplotlib.pyplot as plt

    display_df = coef_df.copy()
    for col in display_df.columns:
        if col == "p-value":
            display_df[col] = display_df[col].apply(
                lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}")
        elif col == "t":
            display_df[col] = display_df[col].map("{:,.2f}".format)
        elif col in ("Coefficient", "Std Error") or col.startswith("CI "):
            display_df[col] = display_df[col].map("{:,.4f}".format)

    font_size, row_height = {"report": (11, 0.55), "presentation": (14, 0.65)}[context]
    cols = list(display_df.columns)
    col_widths_chars = [
        max([len(str(col))] + [len(str(v)) for v in display_df[col].values]) + 3
        for col in cols
    ]
    total_chars = sum(col_widths_chars)
    char_inch = 0.095 + 0.006 * (font_size - 11)
    fig_w = max(8.0, total_chars * char_inch + 0.6)
    col_widths_frac = [w / total_chars for w in col_widths_chars]
    n_rows = len(display_df)
    fig_h = (n_rows + 1) * row_height + 0.9

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    cell_colors = [
        [ATT_COLORS["gray_100"] if r % 2 == 1 else ATT_COLORS["white"] for _ in cols]
        for r in range(n_rows)
    ]
    table = ax.table(
        cellText=display_df.values.tolist(), colLabels=cols,
        cellColours=cell_colors, colWidths=col_widths_frac,
        cellLoc="left", colLoc="left", loc="upper left", bbox=[0, 0.12, 1, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    p_col_idx = cols.index("p-value") if "p-value" in cols else None
    for (r, c), cell in table.get_celld().items():
        cell.PAD = 0.04
        if r == 0:
            cell.set_facecolor(ATT_COLORS["att_blue"])
            cell.set_text_props(color="white", weight="bold")
            cell.set_edgecolor("white")
        else:
            cell.set_edgecolor(ATT_COLORS["gray_300"])
            if p_col_idx is not None and c == p_col_idx:
                raw_p = coef_df.iloc[r - 1]["p-value"]
                if raw_p < 0.05:
                    cell.set_facecolor(ATT_COLORS["pale_sky"])
                    cell.set_text_props(weight="bold")

    caption = (
        f"OLS coefficients — N = {int(model.nobs)},  "
        f"R² = {model.rsquared:.3f},  Adj. R² = {model.rsquared_adj:.3f},  "
        f"F = {model.fvalue:.2f} (p = {model.f_pvalue:.3g})"
    )
    fig.text(0.5, 0.04, caption, ha="center", va="center",
             fontsize=font_size - 1, style="italic", color=ATT_COLORS["gray_700"])
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=ATT_COLORS["white"])
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def style_ols_model_comparison(comparison_df: pd.DataFrame, *, best_by: str,
                               context: str = "report"):
    """Styler for the OLS model-comparison table; best model (row 0) highlighted."""
    fmt = {"n_parameters": "{:,.0f}", "Adjusted R2": "{:,.3f}",
           "AIC": "{:,.2f}", "BIC": "{:,.2f}"}

    def _highlight_best(row):
        if row.name == 0:
            return [f"background-color: {ATT_COLORS['pale_sky']}; font-weight: bold;"
                    for _ in row]
        return ["" for _ in row]

    return (
        comparison_df.style
            .set_table_styles(att_table_styles(context=context))
            .hide(axis="index")
            .format(fmt)
            .apply(_highlight_best, axis=1)
            .set_caption(f"OLS model comparison — best model selected by {best_by}")
            .set_table_styles(_CAPTION_STYLE, overwrite=False)
    )


def save_ols_model_comparison_table(comparison_df: pd.DataFrame, styler,
                                    path: Union[str, Path], *, best_by: str,
                                    context: str = "report", **kwargs) -> str:
    """Save the comparison table; PNG path uses the wrapping renderer below."""
    return save_table_by_extension(
        comparison_df, path, styler=styler,
        png_renderer=lambda df, p, **kw: _save_comparison_table_png(
            df, p, best_by=best_by, context=context, **kw),
        **kwargs,
    )


def _save_comparison_table_png(comparison_df, path, *, best_by, context, dpi=300):
    import matplotlib.pyplot as plt

    display_df = comparison_df.copy()
    fmt = {"n_parameters": "{:,.0f}", "Adjusted R2": "{:,.3f}",
           "AIC": "{:,.2f}", "BIC": "{:,.2f}"}
    for col, formatter in fmt.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].map(formatter.format)

    FORMULA_WRAP = 42
    if "formula" in display_df.columns:
        display_df["formula"] = display_df["formula"].apply(
            lambda f: "\n".join(textwrap.wrap(str(f), FORMULA_WRAP)))

    font_size, row_height_base = {"report": (11, 0.55), "presentation": (14, 0.65)}[context]
    cols = list(display_df.columns)

    def _max_line_len(val):
        return max(len(line) for line in str(val).split("\n"))

    col_widths_chars = [
        max([len(str(col))] + [_max_line_len(v) for v in display_df[col].values]) + 3
        for col in cols
    ]

    def _n_lines(val):
        return str(val).count("\n") + 1

    row_line_counts = [
        max(_n_lines(v) for v in display_df.iloc[r].values)
        for r in range(len(display_df))
    ]

    total_chars = sum(col_widths_chars)
    char_inch = 0.095 + 0.006 * (font_size - 11)
    fig_w = max(8.0, total_chars * char_inch + 0.6)
    col_widths_frac = [w / total_chars for w in col_widths_chars]
    fig_h = row_height_base + sum(row_height_base * max(1, lc) for lc in row_line_counts) + 0.9

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    cell_colors = []
    for r in range(len(display_df)):
        if r == 0:
            row_color = ATT_COLORS["pale_sky"]
        elif r % 2 == 1:
            row_color = ATT_COLORS["gray_100"]
        else:
            row_color = ATT_COLORS["white"]
        cell_colors.append([row_color for _ in cols])

    table = ax.table(
        cellText=display_df.values.tolist(), colLabels=cols,
        cellColours=cell_colors, colWidths=col_widths_frac,
        cellLoc="left", colLoc="left", loc="upper left", bbox=[0, 0.12, 1, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (r, c), cell in table.get_celld().items():
        cell.PAD = 0.04
        cell.get_text().set_wrap(True)
        cell.get_text().set_multialignment("left")
        if r == 0:
            cell.set_facecolor(ATT_COLORS["att_blue"])
            cell.set_text_props(color="white", weight="bold")
            cell.set_edgecolor("white")
        else:
            cell.set_edgecolor(ATT_COLORS["gray_300"])
            if r == 1:
                cell.set_text_props(weight="bold")
        if r > 0:
            n_lines = row_line_counts[r - 1]
            if n_lines > 1:
                cell.set_height(cell.get_height() * n_lines)

    fig.text(0.5, 0.04, f"OLS model comparison — best model selected by {best_by}",
             ha="center", va="center", fontsize=font_size - 1, style="italic",
             color=ATT_COLORS["gray_700"])
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=ATT_COLORS["white"])
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Prediction summary table
# ---------------------------------------------------------------------------

def style_prediction_summary(predictions: pd.DataFrame, *, conf_level: int,
                             new_data: Optional[pd.DataFrame] = None,
                             training_data: Optional[pd.DataFrame] = None,
                             model: Optional[RegressionResultsWrapper] = None,
                             y_name: Optional[str] = None):
    """Wide-format prediction summary Styler. Returns (styler, summary_df)."""
    be_col = "Best Estimate"
    ci_col = f"{conf_level}% CI"
    pi_col = f"{conf_level}% PI"
    response_label = y_name if y_name else "Response"

    if new_data is not None and model is not None:
        model_raw_cols = set()
        for name in model.model.exog_names:
            if name == "Intercept":
                continue
            if "(" in name and ")" in name:
                raw = name[name.index("(") + 1: name.index(")")]
            else:
                raw = name.split("[")[0].split(":")[0]
            model_raw_cols.add(raw)
        predictor_cols = [c for c in new_data.columns if c in model_raw_cols]
    elif new_data is not None:
        predictor_cols = list(new_data.columns)
    else:
        predictor_cols = []

    extrap_data = new_data[predictor_cols] if new_data is not None and predictor_cols else new_data
    if extrap_data is not None:
        row_is_extrap, col_flags = _detect_extrapolation(extrap_data, training_data)
    else:
        row_is_extrap = [False] * len(predictions)
        col_flags = [{} for _ in range(len(predictions))]
    any_extrap = any(row_is_extrap)

    rows = {}
    for col in predictor_cols:
        col_vals = []
        for i, val in enumerate(new_data[col].values):
            try:
                s = f"{float(val):,.2f}"
            except (TypeError, ValueError):
                s = str(val)
            if col_flags[i].get(col):
                s = f"{s} *"
            col_vals.append(s)
        rows[col] = col_vals

    rows[be_col] = [f"{v:,.2f}" for v in predictions["mean"].values]
    rows[ci_col] = [f"[{lo:,.2f}, {hi:,.2f}]"
                    for lo, hi in zip(predictions["ci_lower"], predictions["ci_upper"])]
    rows[pi_col] = [f"[{lo:,.2f}, {hi:,.2f}]"
                    for lo, hi in zip(predictions["pi_lower"], predictions["pi_upper"])]
    summary_df = pd.DataFrame(rows).reset_index(drop=True)

    predictor_group = "Predictors"
    response_group = response_label
    tuples = (
        [(predictor_group, c) for c in predictor_cols] +
        [(response_group, be_col), (response_group, ci_col), (response_group, pi_col)]
    )
    summary_df.columns = pd.MultiIndex.from_tuples(tuples)

    sizes = _SIZE_PRESETS["report"]
    magenta_light = "#FEF0F4"
    table_styles = [
        {"selector": "th.col_heading.level0",
         "props": [("background-color", ATT_COLORS["navy"]), ("color", ATT_COLORS["white"]),
                   ("font-size", f"{sizes['table'] + 1}pt"), ("font-weight", "bold"),
                   ("text-align", "center"), ("padding", "8px 14px"),
                   ("border-bottom", f"2px solid {ATT_COLORS['att_blue']}")]},
        {"selector": "th.col_heading.level1",
         "props": [("background-color", ATT_COLORS["deep_blue"]), ("color", ATT_COLORS["white"]),
                   ("font-size", f"{sizes['table']}pt"), ("font-weight", "bold"),
                   ("text-align", "center"), ("padding", "8px 14px")]},
        {"selector": "td",
         "props": [("background-color", ATT_COLORS["white"]), ("font-size", f"{sizes['table']}pt"),
                   ("color", ATT_COLORS["gray_900"]), ("padding", "8px 14px"),
                   ("text-align", "right")]},
        {"selector": "tr:nth-child(even) td",
         "props": [("background-color", ATT_COLORS["gray_100"])]},
        {"selector": "tr:hover td",
         "props": [("background-color", ATT_COLORS["pale_sky"])]},
        {"selector": "table",
         "props": [("border-collapse", "collapse"),
                   ("border", f"1px solid {ATT_COLORS['gray_300']}"),
                   ("border-bottom", f"2px solid {ATT_COLORS['gray_900']}"),
                   ("width", "100%")]},
        {"selector": "caption",
         "props": [("caption-side", "bottom"), ("background-color", ATT_COLORS["white"]),
                   ("color", ATT_COLORS["black"]), ("font-size", f"{sizes['table'] - 1}pt"),
                   ("font-style", "normal"), ("font-weight", "normal"), ("text-align", "left"),
                   ("padding", "6px 4px 2px 4px"),
                   ("border-top", f"1px solid {ATT_COLORS['gray_300']}")]},
    ]

    styler = summary_df.style.set_table_styles(table_styles).hide(axis="index")
    styler = styler.set_properties(subset=[(response_group, be_col)],
                                   **{"font-weight": "bold", "color": ATT_COLORS["navy"]})
    styler = styler.set_properties(subset=[(response_group, ci_col)],
                                   **{"color": ATT_COLORS["orange"], "font-weight": "semibold"})
    styler = styler.set_properties(subset=[(response_group, pi_col)],
                                   **{"color": ATT_COLORS["navy"], "font-weight": "semibold"})

    for i, extrap in enumerate(row_is_extrap):
        if extrap:
            styler = styler.set_properties(subset=pd.IndexSlice[i, :],
                                           **{"background-color": magenta_light})
    if any_extrap:
        styler = styler.set_caption("* Extrapolated predictor")

    return styler, summary_df


def save_prediction_table_png(df: pd.DataFrame, path: Union[str, Path], *,
                              font_size: int = 12, dpi: int = 300) -> str:
    """Render the prediction summary (possibly MultiIndex columns) as a PNG."""
    import matplotlib.pyplot as plt

    display_df = df.copy()
    if isinstance(display_df.columns, pd.MultiIndex):
        display_df.columns = [
            "\n".join(str(part) for part in col if str(part) != "")
            for col in display_df.columns
        ]

    cols = list(display_df.columns)
    n_rows = len(display_df)
    col_widths_chars = [
        max([len(str(col))] + [len(str(v)) for v in display_df[col].values]) + 4
        for col in cols
    ]
    total_chars = sum(col_widths_chars) if col_widths_chars else 1
    fig_w = max(8.0, total_chars * 0.11)
    fig_h = max(1.8, (n_rows + 1) * 0.55 + 0.4)
    col_widths_frac = [w / total_chars for w in col_widths_chars]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    cell_colors = [
        [ATT_COLORS["gray_100"] if r % 2 == 1 else ATT_COLORS["white"] for _ in cols]
        for r in range(n_rows)
    ]
    table = ax.table(
        cellText=display_df.values.tolist(), colLabels=cols,
        cellColours=cell_colors, colWidths=col_widths_frac,
        cellLoc="center", colLoc="center", loc="upper left", bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (r, c), cell in table.get_celld().items():
        cell.PAD = 0.04
        if r == 0:
            cell.set_facecolor(ATT_COLORS["deep_blue"])
            cell.set_text_props(color="white", weight="bold")
            cell.set_edgecolor("white")
        else:
            cell.set_edgecolor(ATT_COLORS["gray_300"])
            if cols[c].endswith("Best Estimate"):
                cell.set_text_props(color=ATT_COLORS["navy"], weight="bold")
            if "CI" in cols[c]:
                cell.set_text_props(color=ATT_COLORS["orange"], weight="semibold")
            if "PI" in cols[c]:
                cell.set_text_props(color=ATT_COLORS["navy"], weight="semibold")

    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=ATT_COLORS["white"])
    plt.close(fig)
    return str(path)