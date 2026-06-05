"""
core/viz.py — AT&T-branded plotting via seaborn's Objects API.

Replaces the earlier plotnine-based ``gg`` engine. seaborn's Objects API keeps
the layered, compositional style — ``so.Plot(df, x=, y=).add(so.Dots()).add(
so.Line(), so.PolyFit())`` — but as plain seaborn, with no extra grammar
dependency and no fighting a separate theme system.

Usage
-----
Import the objects namespace and the brand helpers from one place::

    from DSSP2026.core.viz import so, apply_att_seaborn, att_nominal
    apply_att_seaborn()                      # brand theme globally, once

    (so.Plot(flights, x="Distance", y="DepDelay", color="DayName")
        .add(so.Dots(alpha=0.4))
        .add(so.Line(), so.PolyFit())        # regression line (geom_smooth analogue)
        .scale(color=att_nominal())          # brand palette, on demand
        .label(title="Delay vs distance by day"))

``apply_att_seaborn()`` sets the global theme (fonts, panel, grid, spines) so
every ``so.Plot`` inherits it. Brand **color scales** are added per plot via
``.scale(color=...)`` using the helpers below — "scales on demand", because
seaborn (rightly) can't guess whether a mapping is categorical or continuous.

Branding cheatsheet
-------------------
- discrete color/fill  -> ``.scale(color=att_nominal())``
- continuous color     -> ``.scale(color=att_continuous())``
- diverging color      -> ``.scale(color=att_diverging())``

The continuous scales reference the ``att_sequential`` / ``att_diverging``
colormaps that ``core.style.apply_att_style`` registers, so call that (or
``apply_att_seaborn``, which calls it) before using them.
"""

from typing import Optional

import matplotlib as mpl
import seaborn as sns
import seaborn.objects as so   # noqa: F401  (re-exported)
from matplotlib.colors import LinearSegmentedColormap

from DSSP2026.core.style import (
    apply_att_style, ATT_COLORS, ATT_PALETTE, ATT_SEQUENTIAL, ATT_DIVERGING,
)


def _att_cmap(colors, reverse: bool):
    """Build (and return the name of) an AT&T colormap, reversed if asked.

    seaborn's ``Continuous(values=...)`` takes a registered colormap *name*, and
    the reversed ``*_r`` variants of our custom maps aren't auto-registered — so
    register a reversed copy on demand and hand back its name.
    """
    if not reverse:
        return colors  # caller passes the already-registered base name
    name = f"{colors}_r"
    if name not in mpl.colormaps:
        base = mpl.colormaps[colors]
        mpl.colormaps.register(base.reversed(), name=name, force=True)
    return name


# ---------------------------------------------------------------------------
# Global theme
# ---------------------------------------------------------------------------
def apply_att_seaborn(context: str = "report", base_size: float = None):
    """Apply the AT&T look globally for both classic seaborn and the Objects API.

    Calls :func:`core.style.apply_att_style` (which sets matplotlib rcParams,
    the seaborn theme, fonts, and registers the ``att_sequential`` /
    ``att_diverging`` colormaps), then mirrors the key rcParams into the Objects
    API's global theme so bare ``so.Plot(...)`` calls inherit the brand styling.

    Parameters
    ----------
    context : {"report", "presentation"}
        Sizing preset, forwarded to :func:`apply_att_style`.
    base_size : float, optional
        Override the preset with a single base font size (points); scales every
        text element proportionally. Forwarded to :func:`apply_att_style`.

        IMPORTANT: the Objects API reads its theme from a dict seeded *here*, not
        from live rcParams — so to change ``so.Plot`` text sizes you must pass
        ``base_size`` to *this* function (re-running ``apply_att_style`` alone
        won't reach already-created Plots).
    """
    apply_att_style(context=context, base_size=base_size)

    # The Objects API keeps its own theme dict; re-seed it from the live
    # rcParams so so.Plot output reflects the current sizing each time this runs.
    so.Plot.config.theme.update(mpl.rcParams.copy())

    # Default categorical palette for the Objects API color scale.
    sns.set_palette(list(ATT_PALETTE))


# ---------------------------------------------------------------------------
# Color scales — pass to .scale(color=...) / .scale(fill=...)
# ---------------------------------------------------------------------------
def att_nominal(order=None, reverse: bool = False):
    """Discrete AT&T categorical color scale for ``.scale(color=...)``.

    Parameters
    ----------
    order : list, optional
        Explicit category order.
    reverse : bool
        Reverse the palette assignment.
    """
    palette = list(ATT_PALETTE)
    if reverse:
        palette = palette[::-1]
    return so.Nominal(values=palette, order=order)


def att_continuous(norm=None, trans: Optional[str] = None, reverse: bool = False):
    """Continuous AT&T sequential color scale (uses the ``att_sequential`` cmap).

    Parameters
    ----------
    norm : tuple, optional
        ``(vmin, vmax)`` to fix the scale's domain.
    trans : str, optional
        A transform, e.g. ``"log"`` for skewed data.
    reverse : bool
        Use the reversed colormap (``att_sequential_r``).
    """
    cmap = _att_cmap("att_sequential", reverse)
    return so.Continuous(values=cmap, norm=norm, trans=trans)


def att_diverging(norm=None, trans: Optional[str] = None, reverse: bool = False):
    """Continuous AT&T diverging color scale (uses the ``att_diverging`` cmap).

    For data centered on a meaningful midpoint (e.g. correlations around 0).
    Set ``norm`` symmetrically (e.g. ``(-1, 1)``) so the midpoint stays centered.
    """
    cmap = _att_cmap("att_diverging", reverse)
    return so.Continuous(values=cmap, norm=norm, trans=trans)


def att_palette(n: int = None):
    """Return the AT&T categorical palette (optionally the first *n* colors)."""
    pal = list(ATT_PALETTE)
    if n is None:
        return pal
    return (pal * (n // len(pal) + 1))[:n]


def att_sequential_colors(n: int, *, reverse: bool = False):
    """Sample *n* evenly-spaced hex colors along the ``att_sequential`` ramp.

    Unlike :func:`att_palette` (the categorical brand palette: deep_blue,
    orange, teal …), this returns *related* colors drawn from the sequential
    blue ramp — pale_sky → sky → att_blue → deep_blue → navy. Useful when bars
    or categories should read as a single colour family rather than maximally
    distinct hues. For a single class it returns the mid-ramp colour.
    """
    from matplotlib.colors import to_hex
    cmap = mpl.colormaps[_att_cmap("att_sequential", reverse)]
    if n <= 1:
        return [to_hex(cmap(0.5))]
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def att_sequential_nominal(n: int, *, order=None, reverse: bool = False):
    """Discrete colour scale of *n* colours sampled from the sequential ramp.

    Pass to ``.scale(color=...)`` when you want each category coloured from the
    brand blue ramp instead of the categorical palette::

        .scale(color=att_sequential_nominal(n_classes))
    """
    return so.Nominal(values=att_sequential_colors(n, reverse=reverse), order=order)
