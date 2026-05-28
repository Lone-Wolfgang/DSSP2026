"""
logistic/tables.py — presentation-quality tables for logistic regression.

The coefficient table on the odds-ratio scale is the logistic-specific view.
Classification report and confusion-matrix tables are family-agnostic and
live in core/tables.py — use those for evaluation output.
"""

from typing import Union
from pathlib import Path

import pandas as pd
from statsmodels.discrete.discrete_model import BinaryResultsWrapper

from modules.core.style import ATT_COLORS, att_table_styles
from modules.core.tables import _CAPTION_STYLE, save_generic_table_png


def style_logit_coefficients(coef_df: pd.DataFrame, model: BinaryResultsWrapper,
                             *, context: str = "report"):
    """Styler for the logistic coefficient table (odds-ratio scale).

    Significant rows (p < 0.05) are highlighted, and the odds-ratio column is
    emphasized since it's the headline quantity. An OR of 1.0 means no effect;
    the caption reminds the reader of that reference line.
    """
    p_col = "p-value"
    or_col = "Odds Ratio"

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
        "z":           "{:,.2f}",
        p_col:         _format_p,
        or_col:        "{:,.3f}",
    }
    for col in coef_df.columns:
        if col.startswith("OR CI "):
            fmt[col] = "{:,.3f}"

    pseudo_r2 = getattr(model, "prsquared", float("nan"))
    caption = (
        f"Logistic coefficients — N = {int(model.nobs)}, "
        f"pseudo-R² = {pseudo_r2:.3f}, "
        f"LLR p = {model.llr_pvalue:.3g}. "
        f"Odds Ratio > 1 raises the odds of the positive outcome; < 1 lowers it."
    )

    styler = (
        coef_df.style
            .set_table_styles(att_table_styles(context=context))
            .hide(axis="index")
            .format(fmt)
            .map(_highlight_sig, subset=[p_col])
            .set_properties(subset=[or_col],
                            **{"font-weight": "bold", "color": ATT_COLORS["navy"]})
            .set_caption(caption)
            .set_table_styles(_CAPTION_STYLE, overwrite=False)
    )
    return styler


def save_logit_coefficients_png(coef_df: pd.DataFrame, model: BinaryResultsWrapper,
                                path: Union[str, Path], *, context: str = "report",
                                dpi: int = 220) -> str:
    """Render the odds-ratio coefficient table as a PNG."""
    p_col = "p-value"
    col_fmts = {
        "Coefficient": "{:,.4f}".format,
        "Std Error":   "{:,.4f}".format,
        "z":           "{:,.2f}".format,
        p_col:         lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}",
        "Odds Ratio":  "{:,.3f}".format,
    }
    for col in coef_df.columns:
        if col.startswith("OR CI "):
            col_fmts[col] = "{:,.3f}".format

    pseudo_r2 = getattr(model, "prsquared", float("nan"))
    title = (f"Logistic coefficients — N = {int(model.nobs)}, "
             f"pseudo-R² = {pseudo_r2:.3f}  (Odds Ratio > 1 means higher odds)")
    return save_generic_table_png(
        coef_df, path, title=title,
        col_fmts={k: v for k, v in col_fmts.items() if k in coef_df.columns},
        context=context, dpi=dpi)