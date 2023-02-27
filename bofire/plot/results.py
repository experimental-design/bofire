from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bofire.domain.domain import Domain
from bofire.utils.multiobjective import get_pareto_front


class Layout:
    colorstyles = {
        "evonik": {
            "plotbgcolor": "#E8E5DF",
            "bgcolor": "#FFFFFF",
            "fontcolor": "#000000",
            "highlight": "#8C3080",
            "highlight2": "#A29B93",
            "highlight3": "red",
        },
        "basf": {
            "plotbgcolor": "#e6f2ff",
            "bgcolor": "#FFFFFF",
            "fontcolor": "#3F3F3F",
            "highlight": "#004A96",
            "highlight2": "#21A0D2",
            "highlight3": "red",
        },
        "standard": {
            "plotbgcolor": "#C8C9C7",
            "bgcolor": "#FFFFFF",
            "fontcolor": "#000000",
            "highlight": "#FC9547",
            "highlight2": "#55CFBE",
            "highlight3": "#9A2006",
        },
    }


def plot_scatter_matrix(
    domain: Domain,
    experiments: pd.DataFrame,
    objectives: Optional[List[str]] = [],
    display_pareto_only: bool = False,
    ref_point: Dict = {},
    labels: Dict = {},
    colorstyle="standard",
    diagonal_visible: bool = False,
    showupperhalf: bool = True,
    show_amimation: bool = True,
    ms_per_frame: int = 500
):
    if objectives == []:
        objectives = domain.output_features.get_keys()  # type: ignore

    if len(objectives) < 2:
        raise ValueError("Specify at least two features, that should be plotted.")
    elif len(objectives) == 2 and (showupperhalf is False and diagonal_visible is False):
        raise ValueError("For two features either showupperhalf or diaginal_visible must be set to True.")

    if labels == {}:
        for key in objectives:  # type: ignore
            labels[key] = "<b>" + key + "</b>"

    # set colorstyles
    colorstyles = Layout().colorstyles
    if colorstyle in colorstyles.keys():
        colorstyle = colorstyles.get(colorstyle)
    else:
        raise ValueError("Provide one of the following colorstyles: {colorstyles.keys}")

    # create frames for animation
    fig = go.Figure()

    # one frame for each row in experiments dataframe
    for i, _ in experiments.iterrows():
        dimensions_pareto_points = []
        pareto_front = get_pareto_front(
            domain=domain,
            experiments=experiments
        )
        for key in objectives:  # type: ignore
            dimension = dict(label=labels[key], values=pareto_front[key].iloc[0 : i + 1])
            dimensions_pareto_points.append(dimension)

        pareto_trace = go.Splom(
            name="pareto front",
            dimensions=dimensions_pareto_points,
            diagonal_visible=diagonal_visible,
            showupperhalf=showupperhalf,
            marker=dict(
                size=12,
                line=dict(width=1, color=colorstyle["fontcolor"]),
                color=colorstyle["highlight"],
            ),
        )

        if display_pareto_only is False:
            dimensions_points = []
            for key in objectives:  # type: ignore
                dimension = dict(label=labels[key], values=experiments[key].iloc[0 : i + 1])
                dimensions_points.append(dimension)

            common_trace = go.Splom(
                name="all points",
                dimensions=dimensions_points,
                diagonal_visible=diagonal_visible,
                showupperhalf=showupperhalf,
                marker=dict(
                    size=12,
                    line=dict(width=1, color=colorstyle["fontcolor"]),
                    color=colorstyle["highlight2"],
                    symbol="x"
                ),
            )
            frame = go.Frame(data=[common_trace, pareto_trace])
        else:
            frame = go.Frame(data=pareto_trace)

        fig.frames = fig.frames + (frame,)

    if display_pareto_only is False:
        fig.add_trace(common_trace)

    fig.add_trace(pareto_trace)

# create trace for the ref point if specified
    dimensions_ref_point = []
    if ref_point != {}:
        for obj in objectives:
            if obj not in ref_point.keys():
                ref_point[obj] = np.nan
            dimension = dict(values=[ref_point[key]])
            dimensions_ref_point.append(dimension)
        experiments = pd.concat([experiments, pd.DataFrame(ref_point, index=[0])], axis=0)  # append to experiments for bound calculation of displayed axes

        ref_point_trace = go.Splom(
            name="ref point",
            dimensions=dimensions_ref_point,
            diagonal_visible=diagonal_visible,
            showupperhalf=showupperhalf,
            marker=dict(
                size=12,
                line=dict(width=1, color=colorstyle["fontcolor"]),
                color=colorstyle["highlight3"],
            )
        )
        fig.add_trace(ref_point_trace)

    # specify plot layout
    fig.update_layout(
        paper_bgcolor=colorstyle["bgcolor"],
        plot_bgcolor=colorstyle["plotbgcolor"],
        showlegend=True,
        hovermode="closest",
        title=dict(
            text="<b>Feature Scatter Matrix</b>",
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=0.98,
        ),
        font=dict(family="Arial", size=18, color=colorstyle["fontcolor"]),
        width=400 * len(objectives),  # type: ignore
        height=400 * len(objectives),  # type: ignore
    )

    # set ranges for axis
    axis_specs = {}
    ax_count = 1
    for objective in objectives:
        scale = experiments[objective].mean() * 0.2
        for x in ["x", "y"]:
            key = x + "axis" + str(ax_count)
            axis_specs[key] = dict(
                range=[
                    experiments[objective].min() - scale,
                    experiments[objective].max() + scale,
                ]
            )
        ax_count += 1

    fig.update_layout(axis_specs)

    if show_amimation:
        # create and add animation button
        button = {
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": ms_per_frame}}],
                }
            ],
        }
        fig.update_layout(
            updatemenus=[button]
        )

    return fig
