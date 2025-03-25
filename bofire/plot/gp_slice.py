from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.surrogates.single_task_gp import SingleTaskGPSurrogate


def _create_contour_slice(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    zmin: float,
    zmax: float,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    fixed_input_features: List[ContinuousInput],
    fixed_values: List[float],
    input_features: List[ContinuousInput],
    output_feature: ContinuousOutput,
    observed_data: Optional[pd.DataFrame] = None,
) -> go.Figure:
    fig = go.Figure()

    # plot the contour plot
    fig.add_trace(
        go.Contour(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z,
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",
            showscale=True,
            opacity=0.8,
            contours={
                "showlabels": True,
                "labelfont": {
                    "family": "Raleway",
                    "size": 12,
                    "color": "white",
                },
            },
        )
    )

    # add datapoints in and out of the slice if observed data is provided
    if observed_data is not None:
        fixed_input_keys = [inp.key for inp in fixed_input_features]
        mask = observed_data[[inp.key for inp in fixed_input_features]] == fixed_values
        mask = mask.all(axis=1)
        observed_data_in_slice = observed_data[mask]

        fig.add_trace(
            go.Scatter(
                x=observed_data_in_slice[input_features[0].key],
                y=observed_data_in_slice[input_features[1].key],
                mode="markers",
                marker={"size": 8, "color": "red", "symbol": "circle-open"},
                name="Observed data in slice",
                showlegend=True,
                hoverinfo="text",
                text=[
                    f"Index: {index}<br>"
                    + "<br>".join([f"{key}: {row[key]}" for key in fixed_input_keys])
                    + f"<br>{input_features[0].key}: {row[input_features[0].key]}<br>{input_features[1].key}: {row[input_features[1].key]}<br>{output_feature.key}: {row[output_feature.key]}"
                    for index, row in observed_data_in_slice.iterrows()
                ],
            )
        )

        observed_data_not_in_slice = observed_data[~mask]

        fig.add_trace(
            go.Scatter(
                x=observed_data_not_in_slice[input_features[0].key],
                y=observed_data_not_in_slice[input_features[1].key],
                mode="markers",
                marker={
                    "size": 8,
                    "color": observed_data_not_in_slice[output_feature.key],
                    "cmin": zmin,
                    "cmax": zmax,
                    "colorscale": "Viridis",
                    "symbol": "cross",
                },
                name="Observed data not in slice",
                showlegend=True,
                hoverinfo="text",
                text=[
                    f"Index: {index}<br>"
                    + "<br>".join([f"{key}: {row[key]}" for key in fixed_input_keys])
                    + f"<br>{input_features[0].key}: {row[input_features[0].key]}<br>{input_features[1].key}: {row[input_features[1].key]}<br>{output_feature.key}: {row[output_feature.key]}"
                    for index, row in observed_data_not_in_slice.iterrows()
                ],
            )
        )

    # set the axis labels and title
    fig.update_xaxes(
        title_text=xaxis_title,
        range=[
            X.min() - 0.01 * (X.max() - X.min()),
            X.max() + 0.01 * (X.max() - X.min()),
        ],
    )
    fig.update_yaxes(
        title_text=yaxis_title,
        range=[
            Y.min() - 0.01 * (Y.max() - Y.min()),
            Y.max() + 0.01 * (Y.max() - Y.min()),
        ],
    )
    fig.update_layout(
        title=title, legend={"yanchor": "top", "y": -0.2, "xanchor": "left", "x": 0.01}
    )

    return fig


def plot_gp_slice_plotly(
    surrogate: SingleTaskGPSurrogate,
    fixed_input_features: List[ContinuousInput],
    fixed_values: List[float],
    varied_input_features: List[ContinuousInput],
    output_feature: ContinuousOutput,
    resolution: int = 100,
    observed_data: Optional[pd.DataFrame] = None,
) -> Tuple[go.Figure, go.Figure]:
    """
    Plot a slice of the Gaussian Process model.
    Where all but two input features are fixed, the other two input features are varied and gp predictions of the output features are plotted.
    If observed data is provided it is plotted in the mean prediction plot with a distinction between data points in the slice and data points not in the slice.

    Args:
        model: The trained Gaussian Process model.
        fixed_input_features: A list of ContinuousInput features that are fixed.
        fixed_values: A list of values for the fixed input features.
        varied_input_features: A list of two ContinuousInput features that are varied.
        output_feature: a ContinuousOutput.
        resolution: The resolution of the plot.
        observed_data: A dataframe with observed data.

    Returns:
        A plotly figure showing the mean prediction of the slice.
        A plotly figure showing the standard deviation of the slice.
    """

    # Raise a Valuerror if there are more than two input features
    if len(varied_input_features) != 2:
        raise ValueError("This function requires two input features.")

    # Raise a value error if fixed_values and fixed_input_features do not have the same length
    if len(fixed_values) != len(fixed_input_features):
        raise ValueError(
            "The length of fixed_values and fixed_input_features should be the same."
        )

    # check if input features are in the model
    for feature in fixed_input_features + varied_input_features:
        if feature not in surrogate.inputs.features:
            raise ValueError(f"Input feature {feature.key} not in model")

    # check if output feature is in the model
    if output_feature not in surrogate.outputs.features:
        raise ValueError(f"Output feature {output_feature} not in model")

    # check if the in and output features keys are in the observed data
    if observed_data is not None:
        for feature in fixed_input_features + varied_input_features + [output_feature]:
            if feature.key not in observed_data.columns:
                raise ValueError(f"Feature {feature.key} not in observed data")

    fixed_input_keys = [inp.key for inp in fixed_input_features]

    x1 = np.linspace(
        varied_input_features[0].bounds[0],
        varied_input_features[0].bounds[1],
        resolution,
    )
    x2 = np.linspace(
        varied_input_features[1].bounds[0],
        varied_input_features[1].bounds[1],
        resolution,
    )

    X, Y = np.meshgrid(x1, x2)

    samples = pd.DataFrame(
        {
            varied_input_features[0].key: X.flatten(),
            varied_input_features[1].key: Y.flatten(),
        },
        index=range(resolution**2),
    )
    rows = pd.DataFrame(
        [pd.Series(index=[inp.key for inp in fixed_input_features], data=fixed_values)],
        index=range(resolution**2),
    )
    samples = pd.concat([samples, rows], axis=1)

    y = surrogate.predict(samples)
    output_pred = y[f"{output_feature.key}_pred"]
    output_sd = y[f"{output_feature.key}_sd"]

    if observed_data is not None:
        output_min = min(output_pred.min(), observed_data[output_feature.key].min())
        output_max = max(output_pred.max(), observed_data[output_feature.key].max())
    else:
        output_min = output_pred.min()
        output_max = output_pred.max()

    title_mean = f"{output_feature.key} slice with fixed features: " + ", ".join(
        [f"{key}={value:.2f}" for key, value in zip(fixed_input_keys, fixed_values)]
    )
    fig_mean = _create_contour_slice(
        X,
        Y,
        output_pred,
        output_min,
        output_max,
        title_mean,
        varied_input_features[0].key,
        varied_input_features[1].key,
        fixed_input_features,
        fixed_values,
        varied_input_features,
        output_feature,
        observed_data,
    )

    title_sd = (
        f"{output_feature.key} standard deviation slice with fixed features: "
        + ", ".join(
            [f"{key}={value:.2f}" for key, value in zip(fixed_input_keys, fixed_values)]
        )
    )
    fig_sd = _create_contour_slice(
        X,
        Y,
        output_sd,
        0,
        output_sd.max(),
        title_sd,
        varied_input_features[0].key,
        varied_input_features[1].key,
        fixed_input_features,
        fixed_values,
        varied_input_features,
        output_feature,
    )

    return fig_mean, fig_sd
