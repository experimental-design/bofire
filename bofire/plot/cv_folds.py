import plotly.graph_objects as go

from bofire.surrogates.diagnostics import CvResults


def plot_cv_folds_plotly(
    cv: CvResults,
    plot_uncertainties: bool = True,
    plot_labcodes: bool = False,
    plot_X: bool = False,
) -> go.Figure:
    """Plot the predicted vs true values for each fold in a CVResults object.

    Checks if uncertainties, labcodes and X are available and ignores them if not.

    Args:
        cv: The CvResults object to plot.
        plot_uncertainties: Boolean to indicate whether to plot uncertainties. Default is True.
        plot_labcodes: Boolean to indicate whether to plot labcodes. Default is False.
        plot_X: Boolean to indicate whether to plot X. Default is False.
    Returns:
        go.Figure: Figure of cross-validation folds results.

    """
    fig = go.Figure()
    # Add traces for each fold
    for i, fold in enumerate(cv.results):
        fig.add_trace(
            go.Scatter(
                x=fold.observed,
                y=fold.predicted,
                error_y={"array": fold.standard_deviation}
                if plot_uncertainties and fold.standard_deviation is not None
                else None,
                mode="markers",
                name=f"Fold {i+1}",
                visible=False,  # Initially hide all traces
                text=fold.labcodes
                if plot_labcodes and fold.labcodes is not None
                else None,
                hovertemplate=(
                    "Labcode: %{text}<br>True: %{x}<br>Predicted: %{y}<br>"
                    + "<br>".join(
                        [
                            f"{col}: %{{customdata[{j}]}}"
                            for j, col in enumerate(fold.X.columns)
                        ]
                    )
                    if plot_labcodes
                    and plot_X
                    and fold.labcodes is not None
                    and fold.X is not None
                    else "Labcode: %{text}<br>True: %{x}<br>Predicted: %{y}<br>"
                    if plot_labcodes and fold.labcodes is not None
                    else "True: %{x}<br>Predicted: %{y}<br>"
                    + "<br>".join(
                        [
                            f"{col}: %{{customdata[{j}]}}"
                            for j, col in enumerate(fold.X.columns)
                        ]
                    )
                    if plot_X and fold.X is not None
                    else "True: %{x}<br>Predicted: %{y}<br>"
                ),
                customdata=fold.X.to_numpy() if fold.X is not None else None,
            )
        )

    # combine all folds
    all_folds = cv._combine_folds()

    # trace for all the folds
    fig.add_trace(
        go.Scatter(
            x=all_folds.observed,
            y=all_folds.predicted,
            error_y={"array": all_folds.standard_deviation}
            if plot_uncertainties and all_folds.standard_deviation is not None
            else None,
            mode="markers",
            name="All Folds",
            visible=True,  # Initially show the combined trace
            text=all_folds.labcodes
            if plot_labcodes and all_folds.labcodes is not None
            else None,
            hovertemplate=(
                "Labcode: %{text}<br>True: %{x}<br>Predicted: %{y}<br>"
                + "<br>".join(
                    [
                        f"{col}: %{{customdata[{j}]}}"
                        for j, col in enumerate(all_folds.X.columns)
                    ]
                )
                if plot_labcodes
                and plot_X
                and all_folds.labcodes is not None
                and all_folds.X is not None
                else "Labcode: %{text}<br>True: %{x}<br>Predicted: %{y}<br>"
                if plot_labcodes and all_folds.labcodes is not None
                else "True: %{x}<br>Predicted: %{y}<br>"
                + "<br>".join(
                    [
                        f"{col}: %{{customdata[{j}]}}"
                        for j, col in enumerate(all_folds.X.columns)
                    ]
                )
                if plot_X and all_folds.X is not None
                else "True: %{x}<br>Predicted: %{y}<br>"
            ),
            customdata=all_folds.X.to_numpy() if all_folds.X is not None else None,
        )
    )

    # Add a diagonal line for reference
    min_value = min(all_folds.observed.min(), all_folds.predicted.min())
    max_value = max(all_folds.observed.max(), all_folds.predicted.max())

    fig.add_shape(
        type="line",
        x0=min_value,
        y0=min_value,
        x1=max_value,
        y1=max_value,
        line={"color": "Red"},
        xref="x",
        yref="y",
    )

    # Create dropdown menu
    buttons = [
        {
            "label": "All Folds",
            "method": "update",
            "args": [{"visible": [False] * len(cv.results) + [True]}],
        }
    ]
    for i in range(len(cv.results)):
        visible = [False] * len(cv.results) + [False]
        visible[i] = True
        buttons.append(
            {"label": f"Fold {i+1}", "method": "update", "args": [{"visible": visible}]}
        )

    fig.update_layout(
        title="Predicted vs True Values",
        xaxis_title="True Values",
        yaxis_title="Predicted Values",
        showlegend=True,
        updatemenus=[{"active": 0, "buttons": buttons}],
    )

    return fig
