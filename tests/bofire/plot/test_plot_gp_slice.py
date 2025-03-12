import numpy as np
import pandas as pd
import pytest
from plotly.graph_objects import Figure

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate
from bofire.plot.api import plot_gp_slice_plotly


@pytest.fixture
def setup_surrogate():
    """Set up a real SingleTaskGPSurrogate with continuous input and output features."""
    # Create input features with bounds between 0 and 10
    input_features = [
        ContinuousInput(key="x1", bounds=(0, 1)),
        ContinuousInput(key="x2", bounds=(0, 1)),
        ContinuousInput(key="x3", bounds=(0, 1)),
    ]

    output_feature = ContinuousOutput(key="y")

    domain = Domain(inputs=input_features, outputs=output_feature)

    # Generate some synthetic data
    data = pd.DataFrame(
        {
            "x1": np.random.rand(100),
            "x2": np.random.rand(100),
            "x3": np.random.rand(100),
        }
    )
    data["y"] = data["x1"] + data["x2"] + np.random.normal(0, 0.1, 100)

    # add a valid_y column
    data["valid_y"] = 1

    # Define the surrogate model
    surrogate_data = SingleTaskGPSurrogate(
        inputs=domain.inputs.get_by_keys(domain.inputs.get_keys()),
        outputs=domain.outputs,
    )
    surrogate = surrogates.map(surrogate_data)
    surrogate.fit(data)

    return surrogate, input_features, output_feature, data


def test_valid_inputs(setup_surrogate):
    """Test plot generation with valid input features and observed data."""
    surrogate, input_features, output_feature, data = setup_surrogate
    fixed_input_features = [input_features[2]]  # Fix x3
    fixed_values = [0.5]  # Fix x3 at 0.5
    varied_input_features = [input_features[0], input_features[1]]  # Vary x1 and x2

    # Generate the plots
    fig_mean, fig_sd = plot_gp_slice_plotly(
        surrogate=surrogate,
        fixed_input_features=fixed_input_features,
        fixed_values=fixed_values,
        varied_input_features=varied_input_features,
        output_feature=output_feature,
        resolution=50,
        observed_data=data,
    )

    # Assertions to verify the output
    assert isinstance(fig_mean, Figure)
    assert isinstance(fig_sd, Figure)
    assert len(fig_mean.data) > 0  # Ensure the figure contains at least one trace
    assert len(fig_sd.data) > 0


def test_error_on_more_than_two_varied_inputs(setup_surrogate):
    """Test that an error is raised when more than two input features are varied."""
    surrogate, input_features, output_feature, data = setup_surrogate
    fixed_input_features = [input_features[2]]  # Fix x3
    fixed_values = [0.5]
    varied_input_features = [
        input_features[0],
        input_features[1],
        input_features[2],
    ]  # Three varied inputs

    # Expect a ValueError when more than two input features are varied
    with pytest.raises(ValueError, match="This function requires two input features."):
        plot_gp_slice_plotly(
            surrogate=surrogate,
            fixed_input_features=fixed_input_features,
            fixed_values=fixed_values,
            varied_input_features=varied_input_features,
            output_feature=output_feature,
            resolution=50,
        )


def test_error_on_fixed_values_length_mismatch(setup_surrogate):
    """Test that an error is raised when fixed_values length does not match fixed_input_features."""
    surrogate, input_features, output_feature, data = setup_surrogate
    fixed_input_features = [input_features[2]]  # Fix x3
    fixed_values = [0.5, 0.7]  # Mismatch: 2 values for 1 fixed feature

    # Expect a ValueError due to the mismatch
    with pytest.raises(
        ValueError,
        match="The length of fixed_values and fixed_input_features should be the same.",
    ):
        plot_gp_slice_plotly(
            surrogate=surrogate,
            fixed_input_features=fixed_input_features,
            fixed_values=fixed_values,
            varied_input_features=[input_features[0], input_features[1]],
            output_feature=output_feature,
            resolution=50,
        )


def test_error_on_feature_not_in_model(setup_surrogate):
    """Test that an error is raised when an input feature is not in the surrogate model."""
    surrogate, input_features, output_feature, data = setup_surrogate
    extra_input = ContinuousInput(
        key="x_extra", bounds=(0, 10)
    )  # Feature not in the surrogate
    fixed_input_features = [input_features[2]]
    fixed_values = [0.5]
    varied_input_features = [
        input_features[0],
        extra_input,
    ]  # Use the extra input feature

    # Expect a ValueError because x_extra is not part of the surrogate model
    with pytest.raises(ValueError, match="Input feature .* not in model"):
        plot_gp_slice_plotly(
            surrogate=surrogate,
            fixed_input_features=fixed_input_features,
            fixed_values=fixed_values,
            varied_input_features=varied_input_features,
            output_feature=output_feature,
            resolution=50,
            observed_data=data,
        )


# test error where the in and output features are not in the observed data
def test_error_on_feature_not_in_observed_data(setup_surrogate):
    """Test that an error is raised when an input or output feature is not in the observed data."""
    surrogate, input_features, output_feature, data = setup_surrogate
    fixed_input_features = [input_features[2]]
    fixed_values = [0.5]
    varied_input_features = [input_features[0], input_features[1]]

    # only include x1 and x2 in the observed data
    data = data[["x1", "x2", "y"]]

    # Expect a ValueError because x2 is not in the observed data
    with pytest.raises(ValueError, match="Feature .* not in observed data"):
        plot_gp_slice_plotly(
            surrogate=surrogate,
            fixed_input_features=fixed_input_features,
            fixed_values=fixed_values,
            varied_input_features=varied_input_features,
            output_feature=output_feature,
            resolution=50,
            observed_data=data,
        )
