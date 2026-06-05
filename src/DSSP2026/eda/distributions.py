from typing import Optional, Sequence

import pandas as pd
import matplotlib.pyplot as plt

from DSSP2026.core.style import ATT_COLORS
from DSSP2026.core.figure import save_figure
from DSSP2026.core.color_scales import so, att_nominal, att_sequential_colors


def plot_class_distribution(
    labels: Sequence,
    *,
    title: str = "Class distribution",
    x_label: str = "Count",
    y_label: str = "Class",
    order: Optional[Sequence] = None,
    figsize: Optional[tuple] = None,
    color="deep_blue",
    ax=None,
):
    counts = pd.Series(labels).value_counts()
    if order is not None:
        counts = counts.reindex(order).fillna(0).astype(int)
    count_df = counts.rename_axis("Class").reset_index(name="Count")
    count_df["Class"] = count_df["Class"].astype(str)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or (8, max(2.2, 0.7 * len(count_df) + 1.5)))
    else:
        fig = ax.figure

    bar_order = count_df["Class"].tolist()

    single_fill, color_scale = _resolve_bar_color(color, count_df)

    plot = so.Plot(count_df, x="Count", y="Class")
    if color_scale is not None:
        plot = (plot.add(so.Bar(), color="Class", legend=False)
                    .scale(y=so.Nominal(order=bar_order), color=color_scale))
    else:
        plot = (plot.add(so.Bar(color=single_fill))
                    .scale(y=so.Nominal(order=bar_order)))
    plot.label(x=x_label, y=y_label, title=title).on(ax).plot()

    total = int(count_df["Count"].sum())
    for _, row in count_df.iterrows():
        pct = row["Count"] / total if total else 0
        ax.text(row["Count"], row["Class"], f"  {int(row['Count']):,}\n({pct:.0%})",
                va="center", ha="left", fontsize=14, color=ATT_COLORS["gray_900"])

    ax.margins(x=0.12)
    fig.tight_layout()
    return fig


def _resolve_bar_color(color, count_df):
    if isinstance(color, (so.Nominal, so.Continuous)):
        return None, color

    if color in ("sequential", "sequential_r"):
        n = len(count_df)
        ramp = att_sequential_colors(n)
        ranks = count_df["Count"].rank(method="first").astype(int) - 1
        if color == "sequential":
            class_to_color = {cls: ramp[r] for cls, r in
                              zip(count_df["Class"], ranks)}
        else:
            class_to_color = {cls: ramp[n - 1 - r] for cls, r in
                              zip(count_df["Class"], ranks)}
        return None, so.Nominal(values=class_to_color)

    if color == "palette":
        return None, att_nominal()

    return ATT_COLORS.get(color, color), None


def save_class_distribution_png(labels, path, *, title="Class distribution",
                                x_label="Count", y_label="Class",
                                order: Optional[Sequence] = None, dpi: int = 220,
                                figsize: Optional[tuple] = None,
                                color="deep_blue") -> str:
    return save_figure(
        plot_class_distribution(labels, title=title, x_label=x_label,
                                y_label=y_label, order=order, figsize=figsize,
                                color=color),
        path, dpi=dpi)
