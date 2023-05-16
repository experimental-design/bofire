from typing import Dict, Optional

import numpy as np
import plotly.express as px
import torch

import bofire.priors.api as priors
from bofire.data_models.priors.api import AnyPrior


def plot_prior_pdf_plotly(
    prior: AnyPrior,
    lower: float,
    upper: float,
    layout_options: Optional[Dict] = None,
):
    """Plot the probability density function of the prior with plotly.

    Args:
        prior (AnyPrior): The prior that should be plotted.
        lower (float): lower bound for computing the prior pdf.
        upper (float): upper bound for computing the prior pdf.
        layout_options (Dict, optional): Layout options passed to plotly. Defaults to {}.

    Returns:
        fig, ax objects of the plot.
    """

    x = np.linspace(lower, upper, 1000)

    fig = px.line(
        x=x,
        y=np.exp(priors.map(prior).log_prob(torch.from_numpy(x)).numpy()),
    )

    if layout_options is not None:
        fig.update_layout(layout_options)

    return fig
