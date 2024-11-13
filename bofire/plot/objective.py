from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from bofire.data_models.features.api import ContinuousOutput


def plot_objective_plotly(
    feature: ContinuousOutput,
    lower: float,
    upper: float,
    values: Optional[pd.Series] = None,
    adapt_values: Optional[pd.Series] = None,
    layout_options: Optional[Dict] = None,
):
    """Plot the assigned objective.

    Args:
        feature (ContinuousOutput): Output feature, whose objective should be visualized.
        lower (float): lower bound for the plot
        upper (float): upper bound for the plot
        values (Optional[pd.Series], optional): If provided, scatter also the historical data in the plot. Defaults to None.
        adapt_values (Optional[pd.Series], optional): If provided, adapt the objective function to the passed values.
            Defaults to None.
        layout_options: (Dict, optional): Options that are passed to plotlys `update_layout`.

    """
    if feature.objective is None:
        raise ValueError(
            f"No objective assigned for ContinuousOutputFeature with key {feature.key}.",
        )

    x = pd.Series(np.linspace(lower, upper, 5000))
    reward = feature.objective.__call__(x, x_adapt=adapt_values)

    fig1 = px.line(x=x, y=reward, title=feature.key)

    if values is not None:
        fig2 = px.scatter(
            x=values,
            y=feature.objective.__call__(values, x_adapt=adapt_values),
        )
        fig = go.Figure(data=fig1.data + fig2.data)
    else:
        fig = fig1

    if layout_options is not None:
        fig.update_layout(layout_options)

    return fig
