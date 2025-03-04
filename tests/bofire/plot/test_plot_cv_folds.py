import itertools

import plotly.graph_objs as go
import pytest

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate
from bofire.plot.cv_folds import plot_cv_folds_plotly


@pytest.mark.parametrize("folds", [2, 5, 3, 10, -1])
def test_cv_folds(folds):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=100)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    experiments = experiments.sample(10)
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)
    _, test_cv, _ = model.cross_validate(experiments, folds=folds)

    permutations = list(itertools.product([True, False], repeat=3))

    for plot_uncertainties, plot_labcodes, plot_X in permutations:
        fig = plot_cv_folds_plotly(
            test_cv,
            plot_uncertainties=plot_uncertainties,
            plot_labcodes=plot_labcodes,
            plot_X=plot_X,
        )
        assert isinstance(fig, go.Figure)
