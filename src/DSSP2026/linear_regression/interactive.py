from DSSP2026.core.style import ATT_COLORS
from DSSP2026.linear_regression.fit import _detect_extrapolation


def apply_plotly_att_style(
    fig,
    *,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 1000,
    height: int = 600,
    equal_aspect: bool = False,
):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=22, color=ATT_COLORS["black"])),
        xaxis_title=x_label,
        yaxis_title=y_label,
        plot_bgcolor=ATT_COLORS["white"],
        paper_bgcolor=ATT_COLORS["white"],
        font=dict(family="Helvetica, Arial, sans-serif",
                  size=14, color=ATT_COLORS["gray_900"]),
        hoverlabel=dict(
            bgcolor=ATT_COLORS["white"],
            bordercolor=ATT_COLORS["navy"],
            font=dict(family="Helvetica, Arial, sans-serif",
                      size=13, color=ATT_COLORS["gray_900"]),
            align="left",
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=ATT_COLORS["gray_300"],
            borderwidth=1,
        ),
        width=width, height=height,
        margin=dict(l=70, r=40, t=80, b=70),
    )
    fig.update_xaxes(showgrid=True, gridcolor=ATT_COLORS["gray_300"],
                     zeroline=False, linecolor=ATT_COLORS["gray_900"],
                     title_font=dict(size=16))
    y_kwargs = dict(showgrid=True, gridcolor=ATT_COLORS["gray_300"],
                    zeroline=False, linecolor=ATT_COLORS["gray_900"],
                    title_font=dict(size=16))
    if equal_aspect:
        y_kwargs.update(scaleanchor="x", scaleratio=1)
    fig.update_yaxes(**y_kwargs)
    return fig


def _build_hover_text(new_data, predictions, conf_level, training_data=None):
    _, col_flags = _detect_extrapolation(new_data, training_data)
    lines_per_row = []
    cols = list(new_data.columns)
    for idx in range(len(new_data)):
        extrap_cols = col_flags[idx]
        header = "<b>New observation</b>"
        if extrap_cols:
            header += "  <b style='color:#C8102E'>⚠ Extrapolation</b>"
        pieces = [header]
        for col in cols:
            val = new_data[col].iloc[idx]
            val_str = f"{val:,.3g}" if isinstance(val, float) else str(val)
            if col in extrap_cols:
                pieces.append(f"  <b>{col}: {val_str} [Extrapolation]</b>")
            else:
                pieces.append(f"  {col}: {val_str}")
        row = predictions.iloc[idx]
        pieces.append("")
        pieces.append(f"<b>Best estimate:</b> {row['mean']:,.2f}")
        pieces.append(f"<b>{conf_level}% CI:</b> [{row['ci_lower']:,.2f}, {row['ci_upper']:,.2f}]")
        pieces.append(f"<b>{conf_level}% PI:</b> [{row['pi_lower']:,.2f}, {row['pi_upper']:,.2f}]")
        lines_per_row.append("<br>".join(pieces))
    return lines_per_row


def _predict_plot_interactive(*, new_data, new_filled, predictions, training_data,
                              x, y_name, interval, conf_level, show_training, formula):
    import numpy as np
    import plotly.graph_objects as go

    if x == "actual":
        return _predict_plot_interactive_vs_actual(
            new_data=new_data, predictions=predictions, y_name=y_name,
            interval=interval, conf_level=conf_level, show_training=show_training,
            formula=formula, training_data=training_data)

    ci_color = ATT_COLORS["orange"]
    pi_color = ATT_COLORS["navy"]
    show_ci = interval in ("confidence", "all")
    show_pi = interval in ("prediction", "all")

    x_vals = new_filled[x].to_numpy()
    fig = go.Figure()

    if show_training and training_data is not None and x in training_data.columns:
        fig.add_trace(go.Scatter(
            x=training_data[x], y=training_data[y_name], mode="markers",
            marker=dict(size=6, color=ATT_COLORS["gray_500"], opacity=0.35, line=dict(width=0)),
            name="Training data", hoverinfo="skip"))

    row_is_extrap, _ = _detect_extrapolation(new_data, training_data)
    point_colors = [ATT_COLORS["magenta"] if e else ATT_COLORS["att_blue"] for e in row_is_extrap]
    point_borders = [ATT_COLORS["magenta"] if e else ATT_COLORS["navy"] for e in row_is_extrap]
    hover_text = _build_hover_text(new_data, predictions, conf_level, training_data=training_data)
    interval_hover = [f"{text}<extra></extra>" for text in hover_text]
    marker_size = 12
    offset = (float(np.ptp(x_vals)) * 0.014) if (show_ci and show_pi and len(x_vals) > 1) else 0.0
    ci_colors = [ATT_COLORS["magenta"] if e else ci_color for e in row_is_extrap]
    pi_colors = [ATT_COLORS["magenta"] if e else pi_color for e in row_is_extrap]

    if any(row_is_extrap):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=ATT_COLORS["magenta"],
                        line=dict(color=ATT_COLORS["magenta"], width=2)),
            name="Extrapolation", showlegend=True))
    if show_ci:
        fig.add_trace(go.Scatter(
            x=x_vals, y=predictions["ci_upper"], mode="markers",
            marker=dict(symbol="triangle-up-open", size=marker_size, color=ci_colors,
                        line=dict(color=ci_colors, width=2)),
            name=f"{conf_level}% CI", legendgroup="ci",
            text=hover_text, hovertemplate=interval_hover))
        fig.add_trace(go.Scatter(
            x=x_vals, y=predictions["ci_lower"], mode="markers",
            marker=dict(symbol="triangle-down-open", size=marker_size, color=ci_colors,
                        line=dict(color=ci_colors, width=2)),
            name=f"{conf_level}% CI", legendgroup="ci", showlegend=False,
            text=hover_text, hovertemplate=interval_hover))
    if show_pi:
        pi_x = x_vals + offset
        fig.add_trace(go.Scatter(
            x=pi_x, y=predictions["pi_upper"], mode="markers",
            marker=dict(symbol="triangle-up", size=marker_size, color=pi_colors,
                        line=dict(color=pi_colors, width=1.2)),
            name=f"{conf_level}% PI", legendgroup="pi",
            text=hover_text, hovertemplate=interval_hover))
        fig.add_trace(go.Scatter(
            x=pi_x, y=predictions["pi_lower"], mode="markers",
            marker=dict(symbol="triangle-down", size=marker_size, color=pi_colors,
                        line=dict(color=pi_colors, width=1.2)),
            name=f"{conf_level}% PI", legendgroup="pi", showlegend=False,
            text=hover_text, hovertemplate=interval_hover))
    fig.add_trace(go.Scatter(
        x=x_vals, y=predictions["mean"], mode="markers",
        marker=dict(size=12, color=point_colors, line=dict(color=point_borders, width=2)),
        name="New observations", text=hover_text, hovertemplate="%{text}<extra></extra>"))

    title = f"Predictions on new data ({x})"
    if formula:
        title = f"{title}: {formula}"
    return apply_plotly_att_style(fig, title=title, x_label=x, y_label=y_name,
                                  width=1000, height=600)


def _predict_plot_interactive_vs_actual(*, new_data, predictions, y_name, interval,
                                        conf_level, show_training, formula, training_data):
    import numpy as np
    import plotly.graph_objects as go

    if y_name not in new_data.columns:
        raise ValueError(
            f"x='actual' requires the target column '{y_name}' present in new_data.")

    actual = new_data[y_name].to_numpy()
    mean = predictions["mean"].to_numpy()
    show_ci = interval in ("confidence", "all")
    show_pi = interval in ("prediction", "all")
    fig = go.Figure()

    axis_values = [actual, mean]
    if show_ci:
        axis_values.extend([predictions["ci_lower"].to_numpy(), predictions["ci_upper"].to_numpy()])
    if show_pi:
        axis_values.extend([predictions["pi_lower"].to_numpy(), predictions["pi_upper"].to_numpy()])
    lo_axis = float(min(np.min(v) for v in axis_values))
    hi_axis = float(max(np.max(v) for v in axis_values))
    fig.add_trace(go.Scatter(
        x=[lo_axis, hi_axis], y=[lo_axis, hi_axis], mode="lines",
        line=dict(color=ATT_COLORS["gray_700"], width=1.5, dash="dash"),
        name="y = x", hoverinfo="skip"))

    hover_text = _build_hover_text(new_data, predictions, conf_level)
    interval_hover = [f"{text}<extra></extra>" for text in hover_text]
    marker_size = 12
    ci_color = ATT_COLORS["orange"]
    pi_color = ATT_COLORS["navy"]
    offset = (float(np.ptp(actual)) * 0.014) if (show_ci and show_pi and len(actual) > 1) else 0.0

    if show_ci:
        fig.add_trace(go.Scatter(
            x=actual, y=predictions["ci_upper"], mode="markers",
            marker=dict(symbol="triangle-up-open", size=marker_size, color=ci_color,
                        line=dict(color=ci_color, width=2)),
            name=f"{conf_level}% CI", legendgroup="ci",
            text=hover_text, hovertemplate=interval_hover))
        fig.add_trace(go.Scatter(
            x=actual, y=predictions["ci_lower"], mode="markers",
            marker=dict(symbol="triangle-down-open", size=marker_size, color=ci_color,
                        line=dict(color=ci_color, width=2)),
            name=f"{conf_level}% CI", legendgroup="ci", showlegend=False,
            text=hover_text, hovertemplate=interval_hover))
    if show_pi:
        pi_x = actual + offset
        fig.add_trace(go.Scatter(
            x=pi_x, y=predictions["pi_upper"], mode="markers",
            marker=dict(symbol="triangle-up", size=marker_size, color=pi_color,
                        line=dict(color=pi_color, width=1.2)),
            name=f"{conf_level}% PI", legendgroup="pi",
            text=hover_text, hovertemplate=interval_hover))
        fig.add_trace(go.Scatter(
            x=pi_x, y=predictions["pi_lower"], mode="markers",
            marker=dict(symbol="triangle-down", size=marker_size, color=pi_color,
                        line=dict(color=pi_color, width=1.2)),
            name=f"{conf_level}% PI", legendgroup="pi", showlegend=False,
            text=hover_text, hovertemplate=interval_hover))
    fig.add_trace(go.Scatter(
        x=actual, y=mean, mode="markers",
        marker=dict(size=12, color=ATT_COLORS["att_blue"], line=dict(color=ATT_COLORS["navy"], width=2)),
        name="Predicted", text=hover_text, hovertemplate="%{text}<extra></extra>"))

    title = "Predictions vs. actual"
    if formula:
        title = f"{title}: {formula}"
    return apply_plotly_att_style(fig, title=title, x_label=f"Actual ({y_name})",
                                  y_label=f"Predicted ({y_name})", width=800, height=700,
                                  equal_aspect=True)
