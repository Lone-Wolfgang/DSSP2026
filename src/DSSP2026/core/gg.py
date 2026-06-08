"""
eda/gg.py — AT&T-branded ggplot2 grammar via plotnine.

This module is a thin identity layer over `plotnine
<https://plotnine.org>`_, which is a faithful Python port of ggplot2. plotnine
already gives you the full grammar of graphics — a base ``ggplot(df, aes(...))``
plus composable layers (``geom_point``, ``geom_jitter``, ``geom_smooth``,
``geom_bar``, ``geom_boxplot``, …), stats, scales, facets, and coords — so this
module's only job is to make all of that *look* like the rest of the package:
the AT&T palette, sequential/diverging gradients, fonts, and grid.

Usage
-----
Re-export everything from plotnine through here so notebooks have one import and
the brand helpers are alongside::

    from DSSP2026.core.gg import *     # ggplot, aes, geom_*, theme_att, att, …
    set_att_theme()                          # brand theme globally (fonts/grid/…)

    (ggplot(flights, aes("Distance", "DepDelay", color="DayName"))
        + geom_point(alpha=0.4)
        + geom_smooth(method="lm")
        + scale_color_att()                  # brand palette for this plot
        + labs(title="Delay vs distance by day"))

``set_att_theme()`` makes the *theme* (fonts, panel, grid, titles, facet strips)
a global default reliably across plotnine versions. Brand *colors* are added per
plot — either an explicit ``scale_color_att()`` / ``scale_color_att_c()``, or the
one-shot ``att()`` bundle which adds theme + color (+ fill) scales together::

    ggplot(df, aes("x", "y", color="grp")) + geom_point() + att()

What this module does NOT do
----------------------------
Reimplement the grammar. geoms/stats/scales/facets all come from plotnine. If
plotnine doesn't support something, neither does this; file it upstream.
"""

from typing import Sequence

# Re-export the whole plotnine surface so callers import from one place.
from plotnine import *           # noqa: F401,F403  (ggplot, aes, geom_*, …)
import plotnine as p9


from DSSP2026.core.style import (
    ATT_COLORS, ATT_PALETTE, ATT_SEQUENTIAL, ATT_DIVERGING, _resolve_font,
)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def theme_att(base_size: float = 14, context: str = "report"):
    """An AT&T-branded plotnine theme.

    Mirrors the look of ``core.style.apply_att_style``: brand sans-serif font,
    white panel, light-gray major grid, dark axis text, bold brand-black titles,
    no top/right spines.

    Parameters
    ----------
    base_size : float
        Base font size in points; other text sizes scale from it.
    context : {"report", "presentation"}
        ``"presentation"`` bumps the base size for slides.
    """
    if context == "presentation":
        base_size = max(base_size, 18)
    font = _resolve_font()
    s = base_size

    return (
        p9.theme_minimal(base_size=s, base_family=font)
        + p9.theme(
            text=p9.element_text(color=ATT_COLORS["gray_900"], family=font),
            plot_title=p9.element_text(
                size=s * 1.5, weight="bold", color=ATT_COLORS["black"],
                ha="left", margin={"b": 8}),
            plot_subtitle=p9.element_text(size=s * 1.05, color=ATT_COLORS["gray_700"],
                                          ha="left"),
            plot_caption=p9.element_text(size=s * 0.8, color=ATT_COLORS["gray_700"],
                                         style="italic", ha="right"),
            axis_title=p9.element_text(size=s * 1.1, weight="bold",
                                       color=ATT_COLORS["gray_900"]),
            axis_text=p9.element_text(size=s * 0.9, color=ATT_COLORS["gray_900"]),
            legend_title=p9.element_text(size=s, weight="bold",
                                         color=ATT_COLORS["gray_900"]),
            legend_text=p9.element_text(size=s * 0.9, color=ATT_COLORS["gray_900"]),
            # Panel / grid
            panel_background=p9.element_rect(fill=ATT_COLORS["white"]),
            plot_background=p9.element_rect(fill=ATT_COLORS["white"]),
            panel_grid_major=p9.element_line(color=ATT_COLORS["gray_300"],
                                             size=0.6, alpha=0.7),
            panel_grid_minor=p9.element_blank(),
            axis_line=p9.element_line(color=ATT_COLORS["gray_900"], size=1.0),
            axis_ticks=p9.element_line(color=ATT_COLORS["gray_700"], size=0.8),
            # Facet strips
            strip_text=p9.element_text(size=s * 0.95, weight="bold",
                                       color=ATT_COLORS["white"]),
            strip_background=p9.element_rect(fill=ATT_COLORS["deep_blue"]),
            legend_key=p9.element_rect(fill=ATT_COLORS["white"]),
        )
    )


# ---------------------------------------------------------------------------
# Color / fill scales
# ---------------------------------------------------------------------------
# Discrete -> ATT_PALETTE (categorical). Continuous -> ATT_SEQUENTIAL gradient.
# Diverging helpers are offered explicitly since plotnine can't guess intent.


def _maybe_reverse(colors, reverse):
    return list(colors)[::-1] if reverse else list(colors)

def scale_color_att_c(*, reverse=False, **kwargs):
    """Continuous AT&T sequential color scale."""
    return p9.scale_color_gradientn(colors=_maybe_reverse(ATT_SEQUENTIAL, reverse), **kwargs)

def scale_fill_att_c(*, reverse=False, **kwargs):
    return p9.scale_fill_gradientn(colors=_maybe_reverse(ATT_SEQUENTIAL, reverse), **kwargs)

def scale_color_att_diverging(*, reverse=False, **kwargs):
    return p9.scale_color_gradientn(colors=_maybe_reverse(ATT_DIVERGING, reverse), **kwargs)

def scale_fill_att_diverging(*, reverse=False, **kwargs):
    return p9.scale_fill_gradientn(colors=_maybe_reverse(ATT_DIVERGING, reverse), **kwargs)

def scale_color_att(**kwargs):
    """Discrete AT&T categorical color scale (maps to ``ATT_PALETTE``)."""
    return p9.scale_color_manual(values=list(ATT_PALETTE), **kwargs)

def scale_fill_att(**kwargs):
    """Discrete AT&T categorical fill scale (maps to ``ATT_PALETTE``)."""
    return p9.scale_fill_manual(values=list(ATT_PALETTE), **kwargs)



# British spellings, to match plotnine's own aliasing.
scale_colour_att = scale_color_att
scale_colour_att_c = scale_color_att_c
scale_colour_att_diverging = scale_color_att_diverging


# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------
# The theme can be set globally and reliably via plotnine's theme_set. Default
# *color scales*, however, are resolved deep in plotnine's internals in a way
# that's brittle to override globally across versions — so rather than
# monkeypatch, we make the brand scales trivial to add per plot, and offer
# `att()` to add theme + scales in one go.


def set_att_theme(context: str = "report", base_size: float = 14):
    """Set the AT&T theme as the global plotnine default (reliable, version-safe).

    This covers fonts, panel, grid, titles, and facet strips for *every*
    subsequent plot with no per-plot code. Brand **colors** are not global
    defaults (see :func:`att`); add them per plot, which is one short line.

    Parameters
    ----------
    context : {"report", "presentation"}
        Sizing preset passed through to :func:`theme_att`.
    base_size : float
        Base font size in points; other text sizes scale from it. Note
        ``context="presentation"`` raises this to at least 18.
    """
    p9.theme_set(theme_att(context=context, base_size=base_size))


def att(*, kind: str = "discrete", fill: bool = False, diverging: bool = False,
        context: str = "report", base_size: float = 14):
    """Return a list of layers that brand a plot: AT&T theme + matching scale.

    Add it to any ggplot with ``+``. Because plotnine raises if a continuous
    scale meets a discrete variable (and vice-versa), you tell ``att`` which kind
    of color mapping the plot uses::

        # categorical color/fill (default)
        ggplot(df, aes("Distance","DepDelay", color="DayName")) + geom_point() + att()

        # continuous color/fill
        ggplot(df, aes("Distance","DepDelay", color="DepDelay")) + geom_point() + att(kind="continuous")

        # bigger text for a specific plot
        ggplot(df, aes("Distance","DepDelay", color="DayName")) + geom_point() + att(base_size=18)

    If a plot has no color/fill mapping at all, just use ``theme_att()`` (or the
    global :func:`set_att_theme`) — there's no scale to add.

    Parameters
    ----------
    kind : {"discrete", "continuous"}
        Which color/fill scale to include, matching the mapped variable's type.
    fill : bool
        Also add the matching fill scale (for bar/box/area geoms using ``fill=``).
    diverging : bool
        For ``kind="continuous"``, use the diverging gradient (data centered on
        zero) instead of sequential.
    context : {"report", "presentation"}
        Theme sizing preset. ``"presentation"`` raises ``base_size`` to at least 18.
    base_size : float
        Base font size in points; all other text scales from it. Forwarded to
        :func:`theme_att`.

    Returns
    -------
    list
        Layers to add to a ggplot object.
    """
    if kind not in ("discrete", "continuous"):
        raise ValueError("kind must be 'discrete' or 'continuous'")
    layers = [theme_att(context=context, base_size=base_size)]
    if kind == "discrete":
        layers.append(scale_color_att())
        if fill:
            layers.append(scale_fill_att())
    else:
        layers.append(scale_color_att_diverging() if diverging else scale_color_att_c())
        if fill:
            layers.append(scale_fill_att_diverging() if diverging else scale_fill_att_c())
    return layers


def att_palette(n: int = None) -> Sequence:
    """Return the AT&T categorical palette (optionally the first *n* colors)."""
    pal = list(ATT_PALETTE)
    if n is None:
        return pal
    return (pal * (n // len(pal) + 1))[:n]