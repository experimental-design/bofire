from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_duplicates_plotly(
    experiments: pd.DataFrame,
    duplicates: List[str],
    key: str,
    scale: float = 0.1,
) -> go.Figure:
    """Plots duplicates using Plotly.

    Args:
        experiments: The dataframe containing the experiments data.
        duplicates: A list of strings representing the labcodes of the duplicates.
        key: The key of the output feature that should be plotted.
        scale: The scale of the noise added to the x-axis. Default is 0.1.

    Returns:
        fig: The Plotly figure object representing the plot.

    """
    fig = go.Figure()

    # plot everything
    fig.add_shape(
        type="line",
        x0=0,
        y0=min(experiments[key]),
        x1=0,
        y1=max(experiments[key]),
        line={"color": "white"},
    )

    fig.add_trace(
        go.Scatter(
            x=np.random.normal(scale=scale, size=len(experiments)),
            y=experiments[key],
            mode="markers",
            name="total",
        ),
    )

    # loop over the duplicates
    for i, d in enumerate(duplicates):
        fig.add_shape(
            type="line",
            x0=i + 1,
            y0=min(experiments[key]),
            x1=i + 1,
            y1=max(experiments[key]),
            line={"color": "white"},
        )

        fig.add_trace(
            go.Scatter(
                x=np.random.normal(size=len(d), scale=scale) + i + 1,
                y=experiments.loc[experiments.labcode.isin(d), key],
                mode="markers",
                name="-".join(d),
                hovertext=d,
            ),
        )

    fig.update_layout(
        title=f"Duplicates {key}",
        yaxis_title=key,
        xaxis_showticklabels=False,
    )

    return fig
