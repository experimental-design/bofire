import numpy as np
import pandas as pd
import pytest

from bofire.data_models.features.api import ContinuousOutput
from bofire.data_models.objectives.api import MaximizeSigmoidObjective
from bofire.plot.objective import plot_objective_plotly


@pytest.mark.parametrize(
    "feature, data",
    [
        (
            ContinuousOutput(
                key="of1",
                objective=MaximizeSigmoidObjective(w=1, tp=15, steepness=0.5),
            ),
            None,
        ),
        (
            ContinuousOutput(
                key="of1",
                objective=MaximizeSigmoidObjective(w=1, tp=15, steepness=0.5),
            ),
            pd.DataFrame(
                columns=["of1", "of2", "of3"],
                index=range(5),
                data=np.random.uniform(size=(5, 3)),
            ),
        ),
    ],
)
def test_output_feature_plot(feature, data):
    print(feature)
    plot_objective_plotly(
        feature=feature,
        lower=0,
        upper=30,
        values=data["of1"] if data is not None else None,
    )
