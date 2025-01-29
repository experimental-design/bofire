from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objs as go


def compose_annotation(
    caption: str,
    x: float = 0.0,
    y: float = -0.15,
) -> List[Dict[str, Any]]:
    if not caption:
        return []
    return [
        {
            "showarrow": False,
            "text": caption,
            "x": x,
            "xanchor": "left",
            "xref": "paper",
            "y": y,
            "yanchor": "top",
            "yref": "paper",
            "align": "left",
        },
    ]


def plot_feature_importance_by_feature_plotly(
    sensitivity_values: Dict[str, pd.DataFrame],
    relative: bool = False,
    show_std: bool = False,
    caption: str = "",
    importance_measure: str = "",
) -> go.Figure:
    """One plot per metric, showing importances by feature.

    Main part of this code is taken from Ax (https://github.com/facebook/Ax/blob/main/ax/plot/feature_importances.py).

    Args:
        sensitivity_values (Dict[str, pd.DataFrame]): The sensitivity values for each metric in a dict format.
            It takes the following format if only the sensitivity value is plotted:
            `{"metric1":pd.DataFrame}` where the columns of the dataframe are the
            features and the index is called `mean` and `std`.
        relative (bool, optional): Whether to normalize feature importances so that they add to 1. Defaults to False.
        show_std (bool, optional): Whether to show the standard deviation in the plot. Defaults to False.
        caption: An HTML-formatted string to place at the bottom of the plot.
        importance_measure: The name of the importance metric to be added to the title.

    Returns:
        go.Figure: Figure of feature importances.

    """
    traces = []
    dropdown = []
    for i, metric_name in enumerate(sorted(sensitivity_values.keys())):
        importances = sensitivity_values[metric_name]
        df = pd.DataFrame(
            [
                {"Feature": feature, "Importance": importances.loc["mean", feature]}
                for feature in importances.columns
            ],
        )
        if show_std and "std" in importances.index:
            error_x = {
                "type": "data",
                "array": [
                    importances.loc["std", feature] for feature in importances.columns
                ],
                "visible": True,
            }
        else:
            error_x = None
        if relative:
            df["Importance"] = df["Importance"].div(df["Importance"].sum())
        traces.append(
            go.Bar(
                name="Importance",
                orientation="h",
                visible=i == 0,
                x=df["Importance"],
                y=df["Feature"],
                error_x=error_x,
                opacity=0.8,
            ),
        )

        is_visible = [False] * len(sensitivity_values)
        is_visible[i] = True
        dropdown.append(
            {
                "args": ["visible", is_visible],
                "label": metric_name,
                "method": "restyle",
            },
        )
    if not traces:
        raise NotImplementedError("No traces found for metric")

    updatemenus = [
        {
            "x": 0,
            "y": 1,
            "yanchor": "top",
            "xanchor": "left",
            "buttons": dropdown,
            "pad": {
                "t": -40,
            },  # hack to put dropdown below title regardless of number of features
        },
    ]
    features = traces[0].y
    title = (
        "Relative Feature Importances" if relative else "Absolute Feature Importances"
    )
    if importance_measure:
        title = title + " based on " + importance_measure
    layout = go.Layout(
        height=200 + len(features) * 20,
        hovermode="closest",
        margin=go.layout.Margin(l=8 * min(max(len(idx) for idx in features), 75)),
        showlegend=False,
        title=title,
        updatemenus=updatemenus,
        annotations=compose_annotation(caption=caption),
    )

    if relative:
        layout.update({"xaxis": {"tickformat": ".0%"}})

    return go.Figure(data=traces, layout=layout)
