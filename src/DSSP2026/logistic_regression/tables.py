"""
logistic/tables.py — presentation-quality tables for logistic regression.

The coefficient table on the odds-ratio scale is the logistic-specific view.
Classification report and confusion-matrix tables are family-agnostic and
live in core/tables.py — use those for evaluation output.

Notebook vs. disk
-----------------
- ``style_logit_coefficients(...)``  -> a Styler that renders in a notebook.
- ``save_logit_coefficients(..., path)`` -> dispatches on the file extension:
  .html keeps the full brand styling (highlighted significant rows, caption),
  .csv/.xlsx write the raw table, .png renders the matplotlib table image.

Both the notebook view and every saved format are built from one source of
truth (``_coef_formats`` / ``_coef_caption``) so they can't drift apart.
"""

from typing import Union
from pathlib import Path

import pandas as pd
from statsmodels.discrete.discrete_model import BinaryResultsWrapper

from DSSP2026.core.style import ATT_COLORS, att_table_styles
from DSSP2026.core.tables import (
    _CAPTION_STYLE, format_pvalue, save_generic_table_png, save_table_by_extension,
)


# ---------------------------------------------------------------------------
# Shared formatting — one source of truth for both the Styler and the PNG/HTML
# ---------------------------------------------------------------------------

def _coef_formats(coef_df: pd.DataFrame, *, as_callables: bool):
    """Per-column formatters for the coefficient table.

    ``as_callables=False`` returns ``str.format`` spec strings (what
    ``Styler.format`` wants); ``as_callables=True`` returns bound callables
    (what ``save_generic_table_png``'s ``col_fmts`` wants). Built once, used by
    both paths so the notebook view and the saved file always agree.
    """
    base = {
        "Coefficient": "{:,.4f}",
        "Std Error":   "{:,.4f}",
        "z":           "{:,.2f}",
        "Odds Ratio":  "{:,.3f}",
    }
    for col in coef_df.columns:
        if col.startswith("OR CI "):
            base[col] = "{:,.3f}"

    if as_callables:
        fmt = {k: v.format for k, v in base.items()}
    else:
        fmt = dict(base)
    fmt["p-value"] = format_pvalue  # callable in both modes
    return {k: v for k, v in fmt.items() if k in coef_df.columns}


def _coef_caption(model: BinaryResultsWrapper, *, short: bool = False) -> str:
    """Caption/title text describing model fit. `short` drops the LLR p term
    (used for the PNG title, which has less room)."""
    pseudo_r2 = getattr(model, "prsquared", float("nan"))
    if short:
        return (f"Logistic coefficients — N = {int(model.nobs)}, "
                f"pseudo-R² = {pseudo_r2:.3f}  (Odds Ratio > 1 means higher odds)")
    return (f"Logistic coefficients — N = {int(model.nobs)}, "
            f"pseudo-R² = {pseudo_r2:.3f}, LLR p = {model.llr_pvalue:.3g}. "
            f"Odds Ratio > 1 raises the odds of the positive outcome; < 1 lowers it.")


# ---------------------------------------------------------------------------
# Notebook view
# ---------------------------------------------------------------------------

def style_logit_coefficients(coef_df: pd.DataFrame, model: BinaryResultsWrapper,
                             *, context: str = "report"):
    """Styler for the logistic coefficient table (odds-ratio scale).

    Significant rows (p < 0.05) are highlighted, and the odds-ratio column is
    emphasized since it's the headline quantity. An OR of 1.0 means no effect;
    the caption reminds the reader of that reference line.
    """
    or_col = "Odds Ratio"

    def _highlight_sig(val):
        try:
            if val < 0.05:
                return f"background-color: {ATT_COLORS['pale_sky']}; font-weight: bold;"
        except TypeError:
            pass
        return ""

    return (
        coef_df.style
            .set_table_styles(att_table_styles(context=context))
            .hide(axis="index")
            .format(_coef_formats(coef_df, as_callables=False))
            .map(_highlight_sig, subset=["p-value"])
            .set_properties(subset=[or_col],
                            **{"font-weight": "bold", "color": ATT_COLORS["navy"]})
            .set_caption(_coef_caption(model))
            .set_table_styles(_CAPTION_STYLE, overwrite=False)
    )


# ---------------------------------------------------------------------------
# Disk — one saver, dispatches on extension
# ---------------------------------------------------------------------------

def save_logit_coefficients(coef_df: pd.DataFrame, model: BinaryResultsWrapper,
                            path: Union[str, Path], *, context: str = "report",
                            dpi: int = 220) -> str:
    """Save the coefficient table, dispatching on file extension.

    .html -> the styled Styler (brand colors, highlighted rows, caption).
    .csv / .xlsx -> the raw table via pandas writers.
    .png -> the matplotlib table image, formatted to match the notebook view.

    Returns the path written, as a string.
    """
    return save_table_by_extension(
        coef_df, path,
        styler=style_logit_coefficients(coef_df, model, context=context),
        png_renderer=lambda df, p, **kw: save_logit_coefficients_png(
            df, model, p, context=context, dpi=dpi),
    )


def save_logit_coefficients_png(coef_df: pd.DataFrame, model: BinaryResultsWrapper,
                                path: Union[str, Path], *, context: str = "report",
                                dpi: int = 220) -> str:
    """Render the odds-ratio coefficient table as a PNG (kept as a direct entry
    point; ``save_logit_coefficients`` routes here for .png paths)."""
    return save_generic_table_png(
        coef_df, path,
        title=_coef_caption(model, short=True),
        col_fmts=_coef_formats(coef_df, as_callables=True),
        context=context, dpi=dpi)