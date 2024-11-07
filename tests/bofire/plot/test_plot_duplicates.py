import pandas as pd
import plotly.graph_objects as go

from bofire.plot.api import plot_duplicates_plotly


def test_plot_duplicates_plotly():
    experiments = pd.DataFrame(
        {
            "labcode": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "output": [1, 2, 3, 4, 5, 6, 7, 8],
        },
    )
    plot = plot_duplicates_plotly(
        experiments=experiments,
        duplicates=[["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]],
        key="output",
    )
    assert isinstance(plot, go.Figure)
