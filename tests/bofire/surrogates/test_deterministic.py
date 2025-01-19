import pandas as pd
from pandas.testing import assert_frame_equal

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.surrogates.api import (
    CategoricalDeterministicSurrogate,
    LinearDeterministicSurrogate,
)


def test_linear_deterministic_surrogate():
    surrogate_data = LinearDeterministicSurrogate(
        inputs=Inputs(
            features=[
                ContinuousInput(key="a", bounds=(0, 1)),
                ContinuousInput(key="b", bounds=(0, 1)),
            ],
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        intercept=2.0,
        coefficients={"b": 3.0, "a": -2.0},
    )
    surrogate = surrogates.map(surrogate_data)
    assert surrogate.input_preprocessing_specs == {}
    experiments = pd.DataFrame(data={"a": [1.0, 2.0], "b": [0.5, 4.0]})
    preds = surrogate.predict(experiments)
    assert_frame_equal(preds, pd.DataFrame(data={"y_pred": [1.5, 10.0], "y_sd": 0.0}))


def test_categorical_deterministic_surrogate():
    surrogate_data = CategoricalDeterministicSurrogate(
        inputs=Inputs(
            features=[
                CategoricalInput(key="a", categories=["A", "B"]),
            ],
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        mapping={"A": 1.0, "B": 2.0},
    )
    surrogate = surrogates.map(surrogate_data)
    experiments = pd.DataFrame(data={"a": ["A", "B"]})
    preds = surrogate.predict(experiments)
    assert_frame_equal(preds, pd.DataFrame(data={"y_pred": [1.0, 2.0], "y_sd": 0.0}))
