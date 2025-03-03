import numpy as np
import plotly.graph_objects as go

from bofire.surrogates.diagnostics import CvResults


def plot_cv_folds_plotly(cv: CvResults) -> go.Figure:
    """Plot the predicted vs true values for each fold in a CVResults object.

    Args:
        cv: The CvResults object to plot.

    Returns:
        go.Figure: Figure of cross-validation folds results.

    """
    fig = go.Figure()

    # Add traces for each fold
    for i, fold in enumerate(cv.results):
        true_values_gp = fold.observed
        predicted_values_gp = fold.predicted
        uncertainties_gp = fold.standard_deviation

        fig.add_trace(
            go.Scatter(
                x=true_values_gp,
                y=predicted_values_gp,
                error_y={"array": uncertainties_gp},
                mode="markers",
                name=f"Fold {i+1}",
                visible=False,  # Initially hide all traces
            )
        )

    # Add a trace for all folds combined
    all_true_values_gp = np.concatenate([fold.observed for fold in cv.results])
    all_predicted_values_gp = np.concatenate([fold.predicted for fold in cv.results])
    all_uncertainties_gp = np.concatenate(
        [fold.standard_deviation for fold in cv.results]
    )

    fig.add_trace(
        go.Scatter(
            x=all_true_values_gp,
            y=all_predicted_values_gp,
            error_y={"array": all_uncertainties_gp},
            mode="markers",
            name="All Folds",
            visible=True,  # Initially show the combined trace
        )
    )

    # Add a diagonal line for reference
    fig.add_shape(
        type="line",
        x0=min(all_true_values_gp),
        y0=min(all_true_values_gp),
        x1=max(all_true_values_gp),
        y1=max(all_true_values_gp),
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
