from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bofire.data_models.domain.domain import Domain
from bofire.utils.multiobjective import get_pareto_front


class Layout:
    colorstyles: Dict = {
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

    def calculate_axis_ranges(self, features: List, experiments: pd.DataFrame) -> Dict:
        """Calcualtes the range for each axis and writes them into a dictionary that can be passed to fig.update_layout().

        Args:
            features (List): List of all features that are shown on an axis.
            experiments (pd.DataFrame): Dataframe with the values for each dimension.

        Returns:
            Dict: Dictionary with specifications for all axis.
        """
        axis_specs = {}
        ax_count = 1
        for objective in features:
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
        return axis_specs

    def create_button(self, ms_per_frame: int) -> Dict:
        """Creates a play button for the animation.

        Args:
            ms_per_frame (int): Duration per frame in milliseconds.

        Returns:
            Dict: Dictionary with the specifications for the play button. Can be passed directly to fig.update_layout(updatemenus).
        """
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
        return button

    def make_dimensions(
        self,
        features: List,
        labels: Dict,
        experiments: pd.DataFrame,
        from_row: int = 0,
        to_row: int = 0,
    ) -> List:
        """Makes dimensions for the scatter matrix.

        Args:
            features (List): Keys for features that should be plotted by the scatter matrix.
            labels (Dict): Labels that set the x and y titles next to the axis of each feature.
            experiments (pd.DataFrame): Dataframe with the values for each dimension.
            from_row (int, optional): Sets beginning of dataframe slice. Defaults to 0.
            to_row (int, optional): Sets end of dataframe slice. Defaults to 0.

        Returns:
            List: List containing one dictionary for each dimension that can be plotted by graph_objects.Splom.
        """
        dimensions = []
        for key in features:  # type: ignore
            dimension = dict(
                label=labels[key], values=experiments[key].iloc[from_row : to_row + 1]  # type: ignore
            )
            dimensions.append(dimension)
        return dimensions


def plot_scatter_matrix(
    domain: Domain,
    experiments: pd.DataFrame,
    features: List[str] = [],
    display_pareto_only: bool = False,
    ref_point: Dict = {},
    labels: Dict = {},
    scale: int = 1,
    colorstyle="standard",
    diagonal_visible: bool = False,
    showupperhalf: bool = True,
    show_animation: bool = False,
    ms_per_frame: int = 750,
):
    """Creates a scatter matrix with an animation that shows the point exploration of the strategy.

    Args:
        domain (Domain): Domain of the function that was used to generate the results.
        experiments (pd.DataFrame): Generated points by the function.
        features (List[str], optional): List of features that should be plotted in the scatter matrix.
        These can include output and input features as well as desired values that are in the experiments dataframe
        Defaults to output features of the function.
        display_pareto_only (bool, optional): Displays only pareto-optimal points. Defaults to False.
        ref_point (Dict, optional): Coordinates of the ref_point. Defaults to {}.
        labels (Dict, optional): Labels that set the x and y titles next to the axis of each feature.
        Has to be in the form: {"feature_key": "desired label"}. Defaults to {}.
        scale (int, optional): Enlargens or shrinks the overall size of the window.
        colorstyle (str, optional): Colorstyle of the scatter matrix. Defaults to "standard".
        diagonal_visible (bool, optional): Whether the diagonal is visible. Defaults to False.
        showupperhalf (bool, optional): Whether the upper half is visible. Defaults to True.
        show_animation (bool, optional): Whether the animation is visible. Defaults to False.
        ms_per_frame (int, optional): Duration per animation frame in milliseconds.. Defaults to 750.

    Raises:
        ValueError: If not enough features are given to be plotted.
        ValueError: If both showupperhalf and diagonal_visible are set to be False.
        ValueError: If an invalid colorstyle is given

    Returns:
        _type_: _description_
    """

    if features == []:
        features = domain.output_features.get_keys()  # type: ignore

    if len(features) < 2:
        raise ValueError("Specify at least two features, that should be plotted.")
    elif len(features) == 2 and (showupperhalf is False and diagonal_visible is False):
        raise ValueError(
            "For two features either showupperhalf or diaginal_visible must be set to True."
        )

    if labels == {}:
        for key in features:
            labels[key] = "<b>" + key + "</b>"

    # set colorstyles
    colorstyles = Layout().colorstyles
    if colorstyle in colorstyles.keys():
        style = colorstyles.get(colorstyle)
    else:
        raise ValueError("Provide one of the following colorstyles: {colorstyles.keys}")

    # create frames for animation
    fig = go.Figure()

    # one frame for each row in experiments dataframe
    pareto_front = get_pareto_front(domain=domain, experiments=experiments)
    for i, _ in experiments.iterrows():
        if show_animation is False:
            i = len(experiments)

        if i not in pareto_front.index:
            pareto_front.loc[i] = np.nan  # type: ignore
            pareto_front.sort_index(inplace=True)

        dimensions_pareto_points = Layout().make_dimensions(
            features=features,
            labels=labels,
            experiments=pareto_front,
            to_row=i,  # type: ignore
        )

        pareto_trace = go.Splom(
            name="pareto front",
            dimensions=dimensions_pareto_points,
            diagonal_visible=diagonal_visible,
            showupperhalf=showupperhalf,
            marker=dict(
                size=12 * scale,
                line=dict(width=1, color=style["fontcolor"]),  # type: ignore
                color=style["highlight"],  # type: ignore
            ),
        )

        if display_pareto_only is False:
            dimensions_points = Layout().make_dimensions(
                features=features,
                labels=labels,
                experiments=experiments,
                to_row=i,  # type: ignore
            )

            common_trace = go.Splom(
                name="all points",
                dimensions=dimensions_points,
                diagonal_visible=diagonal_visible,
                showupperhalf=showupperhalf,
                marker=dict(
                    size=10 * scale,
                    line=dict(
                        width=1 * scale, color=style["fontcolor"]  # type: ignore
                    ),
                    color=style["highlight2"],  # type: ignore
                    symbol="x",
                ),
            )
            frame = go.Frame(data=[common_trace, pareto_trace])
        else:
            frame = go.Frame(data=pareto_trace)

        fig.frames = fig.frames + (frame,)

        if show_animation is False:  # stop loop after first run without animation
            break

    if display_pareto_only is False:
        fig.add_trace(common_trace)  # type: ignore

    fig.add_trace(pareto_trace)  # type: ignore

    # create trace for the ref point if specified
    dimensions_ref_point = []
    if ref_point != {}:
        for obj in features:
            if obj not in ref_point.keys():
                ref_point[obj] = np.nan
            dimension = dict(values=[ref_point[obj]])
            dimensions_ref_point.append(dimension)
        experiments = pd.concat(
            [experiments, pd.DataFrame(ref_point, index=[0])], axis=0
        )  # append to experiments for bound calculation of displayed axis

        ref_point_trace = go.Splom(
            name="ref point",
            dimensions=dimensions_ref_point,
            diagonal_visible=diagonal_visible,
            showupperhalf=showupperhalf,
            marker=dict(
                size=12 * scale,
                line=dict(width=1, color=style["fontcolor"]),  # type: ignore
                color=style["highlight3"],  # type: ignore
            ),
        )
        fig.add_trace(ref_point_trace)

    # specify plot layout
    fig.update_layout(
        paper_bgcolor=style["bgcolor"],  # type: ignore
        plot_bgcolor=style["plotbgcolor"],  # type: ignore
        showlegend=True,
        hovermode="closest",
        title=dict(
            text="<b>Feature Scatter Matrix</b>",
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=0.98,
        ),
        font=dict(family="Arial", size=18 * scale, color=style["fontcolor"]),  # type: ignore
        width=400 * scale * len(features),
        height=400 * scale * len(features),
    )

    # set ranges for axis
    fig.update_layout(
        Layout().calculate_axis_ranges(features=features, experiments=experiments)
    )

    if show_animation:
        # create and add animation button
        button = Layout().create_button(ms_per_frame=ms_per_frame)
        fig.update_layout(updatemenus=[button])

    return fig
