"""
core/style.py — AT&T plotting style for matplotlib & seaborn.

Usage:
    from att_style import apply_att_style, ATT_COLORS, ATT_PALETTE
    apply_att_style()                          # default, report sizing
    apply_att_style(context="presentation")    # bigger, for slides
"""

import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from cycler import cycler


# ---------------------------------------------------------------------------
# BRAND COLORS
# ---------------------------------------------------------------------------
# Brand reference: AT&T blue (#00A8E0) and black are the official logo colors.
# On white plot backgrounds att_blue is cyan-bright and washes out, so the
# default categorical palette LEADS with deep_blue and reserves att_blue for
# brand accents (titles, table headers, highlights).

ATT_COLORS = {
    # Primary brand
    "att_blue":   "#00A8E0",   # bright logo blue — accent / brand mark
    "black":      "#000000",
    "white":      "#FFFFFF",

    # Plot-readable blues (high contrast on white)
    "deep_blue":  "#0057B8",   # primary plot color
    "navy":       "#002A5C",
    "sky":        "#7FD3EF",
    "pale_sky":   "#D6F0FA",

    # Accent / categorical
    "orange":     "#E55A0B",   # darkened from #F47B20 for contrast
    "teal":       "#00857C",
    "gold":       "#C99000",   # darkened from #FFB81C for contrast
    "magenta":    "#C8102E",
    "purple":     "#6E3FA3",
    "green":      "#2E7031",   # darkened from #3A8C3A for contrast

    # Neutrals (darker than before for legibility)
    "gray_900":   "#1A1A1A",
    "gray_700":   "#333333",   # darkened from #4D4D4D
    "gray_500":   "#666666",   # darkened from #808080
    "gray_300":   "#BBBBBB",   # slightly darker for visible spines/grids
    "gray_100":   "#F2F2F2",
}

# Categorical palette — deep_blue first for contrast, att_blue as a brand
# touch in the second slot, then high-contrast accents.
ATT_PALETTE = [
    ATT_COLORS["deep_blue"],
    ATT_COLORS["orange"],
    ATT_COLORS["teal"],
    ATT_COLORS["att_blue"],
    ATT_COLORS["magenta"],
    ATT_COLORS["gold"],
    ATT_COLORS["purple"],
    ATT_COLORS["green"],
]

ATT_SEQUENTIAL = [
    ATT_COLORS["pale_sky"],
    ATT_COLORS["sky"],
    ATT_COLORS["att_blue"],
    ATT_COLORS["deep_blue"],
    ATT_COLORS["navy"],
]

ATT_DIVERGING = [
    ATT_COLORS["magenta"],
    "#E88896",
    ATT_COLORS["gray_100"],
    ATT_COLORS["sky"],
    ATT_COLORS["deep_blue"],
]


# ---------------------------------------------------------------------------
# TYPOGRAPHY
# ---------------------------------------------------------------------------
# Sizes bumped across the board for readability.
_SIZE_PRESETS = {
    "report": {
        "title":      22,
        "suptitle":   24,
        "axis_label": 16,
        "tick":       14,
        "legend":     14,
        "annotation": 13,
        "table":      13,
    },
    "presentation": {
        "title":      28,
        "suptitle":   32,
        "axis_label": 22,
        "tick":       18,
        "legend":     18,
        "annotation": 16,
        "table":      16,
    },
}

_SEABORN_CONTEXT = {"report": "notebook", "presentation": "talk"}

# Font preference order. The first available wins. We check at apply-time and
# warn if we have to fall back past Helvetica/Arial — that tells the user
# their figures won't match the brand's typographic intent.
_FONT_STACK = [
    "Helvetica Neue",   # macOS
    "Helvetica",        # macOS / many Linux distros via fonts-mathjax etc.
    "Arial",            # Windows, and many Linux setups
    "Liberation Sans",  # FOSS Arial-metric-compatible
    "Nimbus Sans",      # ghostscript-fonts
    "DejaVu Sans",      # matplotlib's bundled fallback
]


def _resolve_font():
    """Pick the first available font from the stack. Warn if it's the last resort."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _FONT_STACK:
        if name in available:
            if name in ("Liberation Sans", "Nimbus Sans", "DejaVu Sans"):
                warnings.warn(
                    f"Using '{name}' — Helvetica/Arial not found. Install one for "
                    f"the intended AT&T look. (macOS: built-in. Ubuntu/Debian: "
                    f"`sudo apt install fonts-liberation` or `ttf-mscorefonts-installer`.)",
                    stacklevel=3,
                )
            return name
    return "DejaVu Sans"


# ---------------------------------------------------------------------------
# APPLY STYLE
# ---------------------------------------------------------------------------
def apply_att_style(context: str = "report"):
    """Apply the AT&T plotting style globally.

    Parameters
    ----------
    context : {"report", "presentation"}
        Use "report" for figures in documents/notebooks, "presentation"
        for slide decks (everything is sized larger).
    """
    if context not in _SIZE_PRESETS:
        raise ValueError(f"context must be one of {list(_SIZE_PRESETS)}")
    sizes = _SIZE_PRESETS[context]
    font_name = _resolve_font()

    # Start from a clean seaborn base, then override.
    sns.set_theme(style="whitegrid", context=_SEABORN_CONTEXT[context])

    mpl.rcParams.update({
        # ---- Fonts ----
        "font.family":        "sans-serif",
        "font.sans-serif":    [font_name] + _FONT_STACK,
        "font.size":          sizes["tick"],
        "text.color":         ATT_COLORS["gray_900"],

        # ---- Titles & labels ----
        "axes.titlesize":     sizes["title"],
        "axes.titleweight":   "bold",
        "axes.titlepad":      16,
        "axes.titlecolor":    ATT_COLORS["black"],

        "figure.titlesize":   sizes["suptitle"],
        "figure.titleweight": "bold",

        "axes.labelsize":     sizes["axis_label"],
        "axes.labelweight":   "semibold",
        "axes.labelcolor":    ATT_COLORS["gray_900"],
        "axes.labelpad":      10,

        # ---- Ticks ----
        "xtick.labelsize":    sizes["tick"],
        "ytick.labelsize":    sizes["tick"],
        "xtick.color":        ATT_COLORS["gray_900"],
        "ytick.color":        ATT_COLORS["gray_900"],
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  1.2,
        "ytick.major.width":  1.2,

        # ---- Legend ----
        "legend.fontsize":       sizes["legend"],
        "legend.title_fontsize": sizes["legend"],
        "legend.frameon":        True,
        "legend.framealpha":     0.95,
        "legend.edgecolor":      ATT_COLORS["gray_300"],
        "legend.facecolor":      ATT_COLORS["white"],

        # ---- Axes / spines / grid ----
        "axes.facecolor":     ATT_COLORS["white"],
        "axes.edgecolor":     ATT_COLORS["gray_900"],
        "axes.linewidth":     1.4,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "grid.color":         ATT_COLORS["gray_300"],
        "grid.linestyle":     "-",
        "grid.linewidth":     0.7,
        "grid.alpha":         0.6,

        # ---- Figure ----
        "figure.facecolor":   ATT_COLORS["white"],
        "figure.figsize":     (10, 6),
        "figure.dpi":         110,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  ATT_COLORS["white"],

        # ---- Lines & markers ----
        "lines.linewidth":    2.8,
        "lines.markersize":   9,
        "patch.edgecolor":    ATT_COLORS["white"],
        "patch.linewidth":    0.8,

        # ---- Color cycle ----
        "axes.prop_cycle":    cycler(color=ATT_PALETTE),
    })

    _register_cmap("att_sequential", ATT_SEQUENTIAL)
    _register_cmap("att_diverging",  ATT_DIVERGING)
    sns.set_palette(ATT_PALETTE)


def _register_cmap(name, colors):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(name, colors)
    try:
        mpl.colormaps.register(cmap=cmap, name=name, force=True)
    except AttributeError:
        mpl.cm.register_cmap(name=name, cmap=cmap)


# ---------------------------------------------------------------------------
# TABLE STYLING
# ---------------------------------------------------------------------------
def att_table_styles(context: str = "report"):
    """Return CSS rules for `DataFrame.style.set_table_styles`."""
    sizes = _SIZE_PRESETS[context]
    return [
        {"selector": "th",
         "props": [("background-color", ATT_COLORS["deep_blue"]),
                   ("color", ATT_COLORS["white"]),
                   ("font-size", f"{sizes['table']}pt"),
                   ("font-weight", "bold"),
                   ("text-align", "left"),
                   ("padding", "10px 14px")]},
        {"selector": "td",
         "props": [("background-color", ATT_COLORS["white"]),   # explicit — no transparency
                   ("font-size", f"{sizes['table']}pt"),
                   ("color", ATT_COLORS["gray_900"]),
                   ("padding", "8px 14px")]},
        {"selector": "tr:nth-child(even) td",
         "props": [("background-color", ATT_COLORS["gray_100"])]},
        {"selector": "tr:hover td",
         "props": [("background-color", ATT_COLORS["pale_sky"])]},  # hover highlight
        {"selector": "table",
         "props": [("border-collapse", "collapse"),
                   ("border", f"1px solid {ATT_COLORS['gray_300']}"),
                   ("width", "100%")]},
    ]


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
    descriptor_col : str, optional
        Name of the column to bold as a row label. If None, the first
        column whose dtype is object/string is used automatically.
        Pass False to skip bolding entirely.
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
    if descriptor_col is None:
        for col in df.columns:
            if df[col].dtype == object or hasattr(df[col], "str"):
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
        else:
            # Fixed sig figs: figure out how many decimal places we need
            import math
            mag = math.floor(math.log10(abs_val))
            decimal_places = max(0, sig_figs - 1 - mag)
            return f"{val:,.{decimal_places}f}"

    fmt = {}
    numeric_cols = []
    for col in df.columns:
        if col == descriptor_col:
            continue
        if pd.api.types.is_integer_dtype(df[col]):
            # Integers: just add comma thousands separator, no decimals
            fmt[col] = lambda v: "—" if pd.isna(v) else f"{int(v):,}"
            numeric_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            fmt[col] = _fmt_number
            numeric_cols.append(col)

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