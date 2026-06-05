import math

from DSSP2026.core.style import ATT_COLORS, _SIZE_PRESETS


def att_format_table(
    df,
    *,
    context: str = "report",
    descriptor_col: str = None,
    sig_figs: int = 3,
    sci_upper: float = 1e5,
    sci_lower: float = 1e-3,
):
    """Apply AT&T default formatting to a DataFrame for notebook display.

    - Header row: centered, brand-blue background, white bold text.
    - Descriptor column (first string-like column by default): bold, left-aligned.
    - Numeric columns: formatted to `sig_figs` significant figures.
      Values >= sci_upper or < sci_lower (and non-zero) use scientific notation.
    - Zebra rows, explicit white background on odd rows (no transparency).
    - Row hover highlight in pale AT&T blue.

    Parameters
    ----------
    df : pd.DataFrame
    context : {"report", "presentation"}
    descriptor_col : str or False, optional
        Name of the column to bold as a row label. If None (default), the
        first column whose dtype is object/string is used automatically;
        if no such column exists, no descriptor is bolded. Pass False to
        skip auto-detection and bolding entirely.
    sig_figs : int, default 3
        Significant figures for numeric values.
    sci_upper : float, default 1e5
        Values >= this threshold render in scientific notation.
    sci_lower : float, default 1e-3
        Non-zero values whose absolute value < this render in scientific notation.

    Returns
    -------
    pandas Styler — call .hide(axis="index") if you don't want the index shown.

    Examples
    --------
    att_format_table(df)
    att_format_table(df, descriptor_col="Metric", context="presentation")
    att_format_table(df).hide(axis="index")
    """
    import pandas as pd

    # ---- Identify the descriptor column ----------------------------------
    # None  -> auto-detect the first object/string column (may stay None).
    # False -> skip auto-detection and bolding entirely.
    # str   -> use that column as given.
    if descriptor_col is None:
        for col in df.columns:
            if (pd.api.types.is_object_dtype(df[col])
                    or pd.api.types.is_string_dtype(df[col])):
                descriptor_col = col
                break

    # ---- Build per-column format functions -------------------------------
    def _fmt_number(val):
        """Format a single number to sig_figs with sci notation when needed."""
        try:
            if pd.isna(val):
                return "—"
        except (TypeError, ValueError):
            return str(val)
        if val == 0:
            return "0"
        abs_val = abs(val)
        if abs_val >= sci_upper or abs_val < sci_lower:
            # Scientific notation, sig_figs significant digits
            decimals = sig_figs - 1
            return f"{val:.{decimals}e}"
        # Fixed sig figs: figure out how many decimal places we need
        mag = math.floor(math.log10(abs_val))
        decimal_places = max(0, sig_figs - 1 - mag)
        return f"{val:,.{decimal_places}f}"

    def _fmt_integer(val):
        return "—" if pd.isna(val) else f"{int(val):,}"

    fmt = {}
    for col in df.columns:
        if col == descriptor_col:
            continue
        if pd.api.types.is_integer_dtype(df[col]):
            # Integers: comma thousands separator, no decimals.
            fmt[col] = _fmt_integer
        elif pd.api.types.is_numeric_dtype(df[col]):
            fmt[col] = _fmt_number

    # ---- Base table styles (centers headers) -----------------------------
    sizes = _SIZE_PRESETS[context]
    table_styles = [
        {"selector": "th",
         "props": [("background-color", ATT_COLORS["deep_blue"]),
                   ("color",            ATT_COLORS["white"]),
                   ("font-size",        f"{sizes['table']}pt"),
                   ("font-weight",      "bold"),
                   ("text-align",       "center"),       # <-- centered
                   ("padding",          "10px 14px")]},
        {"selector": "th.col_heading",
         "props": [("text-align", "center")]},
        {"selector": "th.row_heading",
         "props": [("text-align", "left")]},
        {"selector": "td",
         "props": [("background-color", ATT_COLORS["white"]),
                   ("font-size",        f"{sizes['table']}pt"),
                   ("color",            ATT_COLORS["gray_900"]),
                   ("padding",          "8px 14px"),
                   ("text-align",       "right")]},        # numbers right-aligned
        {"selector": "tr:nth-child(even) td",
         "props": [("background-color", ATT_COLORS["gray_100"])]},
        {"selector": "tr:hover td",
         "props": [("background-color", ATT_COLORS["pale_sky"])]},
        {"selector": "table",
         "props": [("border-collapse", "collapse"),
                   ("border",          f"1px solid {ATT_COLORS['gray_300']}"),
                   ("width",           "100%")]},
    ]

    styler = df.style.set_table_styles(table_styles).format(fmt, na_rep="—")

    # ---- Bold the descriptor column, left-align it -----------------------
    if descriptor_col and descriptor_col in df.columns:
        styler = styler.set_properties(
            subset=[descriptor_col],
            **{"font-weight": "bold",
               "text-align": "left",
               "color": ATT_COLORS["gray_900"]},
        )

    return styler