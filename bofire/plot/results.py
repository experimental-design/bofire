from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from bofire.domain.domain import Domain
from bofire.utils.multiobjective import get_pareto_mask


class Layout:
    colorstyles = {
        "evonik": {
            "plotbgcolor": "#E8E5DF",
            "bgcolor": "#FFFFFF",
            "fontcolor": "#000000",
            "highlight": "#8C3080",
            "highlight2": "#A29B93",
            "highlight3": "#9A2006",
        },
        "basf": {
            "plotbgcolor": "#e6f2ff",
            "bgcolor": "#FFFFFF",
            "fontcolor": "#3F3F3F",
            "highlight": "#004A96",
            "highlight2": "#21A0D2",
            "highlight3": "#9A2006",
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
            if pareto_mask[index]:
                experiments["point type"].iloc[index] = "pareto optimal"

    if ref_point != {}:
        ref_point_df = pd.DataFrame(ref_point, index=[0])
        ref_point_df["point type"] = "ref point"
        experiments = pd.concat([experiments, ref_point_df], axis=0)
        experiments.reset_index(inplace=True, drop=True)

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

    # set symbols for point types
    point_types = np.unique(experiments["point type"].values).tolist()
    point_code = {point_types[k]: k for k in range(len(point_types))}
    symbol_vals = [point_code[cl] for cl in experiments["point type"]]

    # create custom hover data for all objectives
    hover_data_tuple = ()
    for key in domain.inputs.get_keys():
        hover_data_tuple = hover_data_tuple + (experiments[key],)
    custom_hover_data = np.dstack(hover_data_tuple)

    print(custom_hover_data[0])

    for i, _ in experiments.iterrows():
        dimensions = []
        for key in objectives:  # type: ignore
            dimension = dict(label=labels[key], values=experiments[key].iloc[0 : i + 1])
            dimensions.append(dimension)

        points = go.Splom(
            dimensions=dimensions,
            diagonal_visible=diagonal_visible,
            showupperhalf=showupperhalf,
            marker=dict(
                size=12,
                line=dict(width=1, color=colorstyle["fontcolor"]),
                color=symbol_vals,
                colorscale=[
                    colorstyle["highlight2"],
                    colorstyle["highlight"],
                    colorstyle["highlight3"],
                ],
            ),
            customdata=custom_hover_data,
            hovertemplate="<b>z1:%{f_0:.3f}</b><br>z2:%{custom_hover_data[0]:.3f} <br>z3: %{customdata[1]:.3f} ",
        )
        # fig.add_trace(points)
        frame = go.Frame(data=points)
        fig.frames = fig.frames + (frame,)

        visible_position = [False] * len(experiments)
        # visible_position[-1] = True

        visible_position[i] = True
        step = dict(
            method="update",
            label="Sim: " + str(i),
            args=[
                {"visible": visible_position},
            ],  # layout attribute
        )
        slider_steps.append(step)

    fig.add_trace(points)

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
        # sliders=[
        #     {
        #         "active": 0,
        #         "yanchor": "top",
        #         "xanchor": "left",
        #         "currentvalue": {
        #             "font": {"size": 20},
        #             "visible": True,
        #             "xanchor": "right",
        #         },
        #         "transition": {"duration": 300, "easing": "cubic-in-out"},
        #         "pad": {"b": 10, "t": 50},
        #         "len": 0.9,
        #         "x": 0.1,
        #         "y": 0,
        #         "steps": slider_steps,
        #     }
        # ],
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
