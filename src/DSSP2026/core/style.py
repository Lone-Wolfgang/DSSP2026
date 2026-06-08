"""
core/style.py — AT&T plotting style for matplotlib & seaborn.

Usage:
    from att_style import apply_att_style, ATT_COLORS, ATT_PALETTE
    apply_att_style()                          # default, report sizing
    apply_att_style(context="presentation")    # bigger, for slides
"""

import math
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
    "magenta_light": "#FEF0F4",
    "purple":     "#6E3FA3",
    "green_light": "#A4CEA6",
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
    ATT_COLORS["magenta_light"],
    ATT_COLORS["pale_sky"],
    ATT_COLORS["green_light"],
    ATT_COLORS["green"],
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

# Relative size ratios, taken from the "report" preset with `tick` as the base
# (ratio 1.0). When `base_size` is passed to apply_att_style, every element is
# `base_size * ratio`, so one number scales the whole figure while preserving
# the type hierarchy (suptitle > title > axis_label > tick ≈ legend > table).
_SIZE_RATIOS = {
    "suptitle":   24 / 14,
    "title":      22 / 14,
    "axis_label": 16 / 14,
    "tick":       14 / 14,
    "legend":     14 / 14,
    "annotation": 13 / 14,
    "table":      13 / 14,
}


def _sizes_from_base(base_size: float) -> dict:
    """Scale every typographic element off a single base size, preserving ratios."""
    return {k: base_size * ratio for k, ratio in _SIZE_RATIOS.items()}

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
def apply_att_style(context: str = "report", base_size: float = None):
    """Apply the AT&T plotting style globally.

    Parameters
    ----------
    context : {"report", "presentation"}
        Use "report" for figures in documents/notebooks, "presentation"
        for slide decks (everything is sized larger). Also selects the seaborn
        context ("notebook" vs "talk").
    base_size : float, optional
        Override the preset sizing with a single base font size (in points).
        Every text element scales proportionally off this — ticks sit at
        ``base_size``, titles/labels above it, table/annotation slightly below
        — preserving the type hierarchy. ``base_size=14`` reproduces the
        "report" preset; ``~18`` approximates "presentation". When omitted,
        the named ``context`` preset is used unchanged (backward compatible).
    """
    if context not in _SIZE_PRESETS:
        raise ValueError(f"context must be one of {list(_SIZE_PRESETS)}")
    if base_size is not None:
        if base_size <= 0:
            raise ValueError("base_size must be a positive number of points.")
        sizes = _sizes_from_base(base_size)
    else:
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



def att_format_table(*args, **kwargs):
    from DSSP2026.core.tables import att_format_table as _att_format_table
    return _att_format_table(*args, **kwargs)
