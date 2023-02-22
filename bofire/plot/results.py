from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px

from bofire.domain.domain import Domain
from bofire.utils.multiobjective import get_pareto_front, get_pareto_mask


def plot_scatter_matrix(
    domain: Domain,
    experiments: pd.DataFrame,
    dimensions: Optional[List[str]] = [],
    display_pareto=True,
    ref_point: dict = {},
    labels: dict = {},
    diagonal_visible=False,
    showupperhalf=False,
):

    if dimensions == []:
        dimensions = domain.output_features.get_keys()  # type: ignore

    experiments["point type"] = "point"

    if display_pareto:
        pareto_mask = get_pareto_mask(domain=domain, experiments=experiments)
        for index, _ in experiments.iterrows():
            if pareto_mask[index] == True:
                experiments["point type"][index] = "pareto optimal"

    if ref_point != {}:
        ref_point_df = pd.DataFrame(ref_point, index=[0])
        ref_point_df["point type"] = "ref point"
        experiments = pd.concat([experiments, ref_point_df], axis=0)

    scatter_matrix = px.scatter_matrix(
        data_frame=experiments,
        dimensions=dimensions,
        # color="VW",
        color_discrete_sequence=["#991D85", "blue", "#000000", "red"],
        symbol="point type",
        symbol_sequence=["x", "circle", "square"],
        labels=labels,
        width=1000,
        height=1000,
    )

    scatter_matrix.update_layout(
        paper_bgcolor="#E9E6DF",
        plot_bgcolor="white",
        title=dict(
            text="<b>Feature Scatter Matrix</b>",
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=0.98,
        ),
        font=dict(family="Arial", size=18, color="black"),
    )
    scatter_matrix.update_traces(
        diagonal_visible=diagonal_visible,
        showupperhalf=showupperhalf,
        marker=dict(size=12, line=dict(width=1, color="black")),
        selector=dict(mode="markers"),
    )

    return scatter_matrix
