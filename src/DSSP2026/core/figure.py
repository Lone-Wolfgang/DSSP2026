"""
core/figure.py — the one shared matplotlib figure helper.

``save_figure`` is used by every matplotlib-rendering module (``core.heatmap``,
``core.residuals``, and each family's ``plots.py``). It lives here, on its own,
rather than inside any one plotting file, so importing the save helper never
drags in heatmap or residual code. Grammar-of-graphics plots (plotnine, via
``core.gg``) save themselves with ``ggplot.save`` and don't need this.
"""

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt

from DSSP2026.core.style import ATT_COLORS


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    *,
    dpi: int = 220,
    facecolor: str = ATT_COLORS["white"],
    **kwargs,
) -> str:
    """Save *fig* to *path* and close it. Returns the path as a string."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=facecolor, **kwargs)
    plt.close(fig)
    return str(path)
