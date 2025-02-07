import pandas as pd
import numpy as np

from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate
from bofire.data_models.domain.api import Domain, Inputs, Outputs
import bofire.surrogates.api as surrogates
import plotly.graph_objects as go


from bofire.plot.api import plot_gp_slice_plotly


def test_plot_gp_slice_plotly():
    x1 = ContinuousInput(key="x1", bounds=(0, 1))
    x2 = ContinuousInput(key="x2", bounds=(0, 1))
    x3 = ContinuousInput(key="x3", bounds=(0, 1))
    x4 = ContinuousInput(key="x4", bounds=(0, 1))
    y = ContinuousOutput(key="y")

    # Define the domain
    inputs = Inputs(features=[x1, x2, x3, x4])
    outputs = Outputs(features=[y])
    domain = Domain(inputs=inputs, outputs=outputs)

    # generate some data
    # Generate some synthetic data
    data = pd.DataFrame({
        "x1": np.random.rand(100),
        "x2": np.random.rand(100),
        "x3": np.random.rand(100),
        "x4": np.random.rand(100),
    })

    # add a datapoint in the slice to test the plotting
    new_data = pd.DataFrame([{
        "x1": 0.5,
        "x2": 0.5,
        "x3": 0.5,
        "x4": 0.5,
    }])

    data = pd.concat([data, new_data], ignore_index=True)

    data["y"] = data["x1"] + data["x2"] + np.random.normal(0, 0.1, 101)
    data["valid_y"] = 1

    # Define the surrogate model
    surrogate_data = SingleTaskGPSurrogate(inputs=domain.inputs.get_by_keys(domain.inputs.get_keys()), outputs=domain.outputs)
    surrogate_gp = surrogates.map(surrogate_data)
    surrogate_gp.fit(data)

    input_features = [x1, x2]
    fixed_input_features = [x3, x4]
    fixed_values = [0.5, 0.5]
    fig, fig_sd = plot_gp_slice_plotly(surrogate_gp, fixed_input_features, fixed_values, input_features, y, observed_data=data)

    assert isinstance(fig, go.Figure)
    assert isinstance(fig_sd, go.Figure)
