from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from bofire.domain.domain import Domain
from bofire.utils.multiobjective import get_pareto_front, get_pareto_mask


class Layout:
    colorstyles = {
        "evonik": {
            "plotbgcolor": "#FFFFFF",
            "bgcolor": "#E8E5DF",
            "fontcolor": "#000000",
            "highlight": "#8C3080",
            "highlight2": "#A29B93",
        },
        "basf": {
            "plotbgcolor": "#e6f2ff",
            "bgcolor": "#FFFFFF",
            "fontcolor": "#3F3F3F",
            "highlight": "#004A96",
            "highlight2": "#21A0D2",
        },
        "standard": {
            "plotbgcolor": "#C8C9C7",
            "bgcolor": "#FFFFFF",
            "fontcolor": "#000000",
            "highlight": "#FC9547",
            "highlight2": "#55CFBE",
        },
    }


def plot_scatter_matrix(
    domain: Domain,
    experiments: pd.DataFrame,
    objectives: Optional[List[str]] = [],
    display_pareto=True,
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

    if display_pareto:
        pareto_mask = get_pareto_mask(domain=domain, experiments=experiments)
        for index, _ in experiments.iterrows():
            if pareto_mask[index] == True:
                experiments["point type"].iloc[index] = "pareto optimal"

    if ref_point != {}:
        ref_point_df = pd.DataFrame(ref_point, index=[0])
        ref_point_df["point type"] = "ref point"
        experiments = pd.concat([experiments, ref_point_df], axis=0)

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

    for i, _ in experiments.iterrows():
        dimensions = []
        for key in objectives:  # type: ignore
            dimension = dict(label=labels[key], values=experiments[key].iloc[0:i])
            dimensions.append(dimension)

        points = go.Splom(
            dimensions=dimensions,
            diagonal_visible=diagonal_visible,
            showupperhalf=showupperhalf,
            marker=dict(
                size=12,
                line=dict(width=1, color=colorstyle["fontcolor"]),
                color=colorstyle["highlight"],
            ),
        )
        fig.add_trace(points)
        # frames.append(go.Frame(data=[points]))

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

    # scatter_matrix = px.scatter_matrix(
    #     # color="VW",
    #     color_discrete_sequence=[colorstyle["highlight"], colorstyle["highlight2"]],
    #     symbol="point type",
    #     symbol_sequence=["x", "circle", "square"],
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
        title=dict(
            text="<b>Feature Scatter Matrix</b>",
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=0.98,
        ),
        font=dict(family="Arial", size=18, color=colorstyle["fontcolor"]),
        # updatemenus=[button],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": slider_steps,
            }
        ],
        width=400 * len(objectives),  # type: ignore
        height=400 * len(objectives),  # type: ignore
    )

    return fig
