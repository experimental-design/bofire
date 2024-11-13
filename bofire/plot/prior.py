from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import torch
from gpytorch.priors import Prior


def plot_prior_pdf_plotly(
    priors: List[Prior],
    lower: float,
    upper: float,
    layout_options: Optional[Dict] = None,
    labels: Optional[List[str]] = None,
):
    """Plot the probability density function of a gyptorch prior with plotly.

    Args:
        prior: The prior that should be plotted.
        lower: lower bound for computing the prior pdf.
        upper: upper bound for computing the prior pdf.
        layout_options: Layout options passed to plotly. Defaults to None.
        labels: Labels for the priors, that are shown in the plot. Defaults to None.

    Returns:
        fig, ax objects of the plot.

    """
    use_labels = labels is not None and len(labels) == len(priors)
    x = np.linspace(lower, upper, 1000)
    fig = go.Figure()
    for i, prior in enumerate(priors):
        y = np.exp(prior.log_prob(torch.from_numpy(x)).numpy())
        label = labels[i] if use_labels else prior.__class__.__name__  # type: ignore
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label))
    if layout_options is not None:
        fig.update_layout(layout_options)
    return fig
