from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from bofire.domain.domain import Domain
from bofire.utils.multiobjective import get_pareto_mask, get_pareto_front


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
    display_pareto_only=True,
    ref_point: dict = {},
    labels: dict = {},
    colorstyle="standard",
    diagonal_visible=False,
    showupperhalf=False,
):

    if objectives == []:
        objectives = domain.output_features.get_keys()  # type: ignore

    if labels == {}:
        for key in objectives:  # type: ignore
            labels[key] = "<b>" + key + "</b>"

    experiments["point type"] = "point"

    hover_data = {}
    for key in domain.get_feature_keys():
        hover_data[key] = ":.f1"

    colorstyles = Layout().colorstyles
    if colorstyle in colorstyles.keys():
        colorstyle = colorstyles.get(colorstyle)
    else:
        raise ValueError("Provide one of the following colorstyles: {colorstyles.keys}")

    # create frames for animation
    fig = go.Figure()
    slider_steps = []

    # create custom hover data for all objectives
    hover_data_tuple = ()
    for key in domain.inputs.get_keys():
        hover_data_tuple = hover_data_tuple + (experiments[key],)
    custom_hover_data = np.dstack(hover_data_tuple)

    for i, _ in experiments.iterrows():
        if display_pareto_only is False:
            dimensions_points = []
            for key in objectives:  # type: ignore
                dimension = dict(label=labels[key], values=experiments[key].iloc[0 : i + 1])
                dimensions_points.append(dimension)

            points = go.Splom(
                name="all points",
                dimensions=dimensions_points,
                diagonal_visible=diagonal_visible,
                showupperhalf=showupperhalf,
                marker=dict(
                    size=12,
                    line=dict(width=1, color=colorstyle["fontcolor"]),
                    color=colorstyle["highlight2"],
                ),
                # customdata=custom_hover_data,
                # hovertemplate="<b>z1:%{f_0:.3f}</b><br>z2:%{custom_hover_data[0]:.3f} <br>z3: %{customdata[1]:.3f} ",
            )

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
            # customdata=custom_hover_data,
            # hovertemplate="<b>z1:%{f_0:.3f}</b><br>z2:%{custom_hover_data[0]:.3f} <br>z3: %{customdata[1]:.3f} ",
        )

        # fig.add_trace(points)
        frame = go.Frame(data=[points, pareto_trace])
        fig.frames = fig.frames + (frame,)

        visible_position = [False] * len(experiments)

        visible_position[i] = True
        step = dict(
            method="update",
            label="Sim: " + str(i),
            args=[
                {"visible": visible_position},
            ],  # layout attribute
        )
        slider_steps.append(step)

    fig.add_traces([points, pareto_trace])   

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

    # scatter_matrix = px.scatter_matrix(
    #     hover_data=hover_data,
    # )

    # create the button
    button = {
        "type": "buttons",
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 250}}],
            }
        ],
    }

    fig.update_layout(
        paper_bgcolor=colorstyle["bgcolor"],
        plot_bgcolor=colorstyle["plotbgcolor"],
        showlegend=True,
        hovermode="x unified",
        title=dict(
            text="<b>Feature Scatter Matrix</b>",
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=0.98,
        ),
        font=dict(family="Arial", size=18, color=colorstyle["fontcolor"]),
        updatemenus=[button],
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

    return fig
