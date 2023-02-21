from typing import Literal

import numpy as np
import pandas as pd
import pytest
from pydantic.error_wrappers import ValidationError

from bofire.domain.feature import ContinuousInput, ContinuousOutput
from bofire.domain.features import InputFeatures, OutputFeatures
from bofire.models.model import Model, PredictedValue


class Dummy(Model):
    type: Literal["Dummy"] = "Dummy"

    def _predict(self, transformed_X: pd.DataFrame):
        preds = np.random.normal(
            loc=5, scale=1, size=(len(transformed_X), len(self.output_features))
        )
        stds = np.random.uniform(
            low=0.0, high=1.0, size=(len(transformed_X), len(self.output_features))
        )
        return preds, stds


def test_zero_input_features():
    input_features = InputFeatures(features=[])
    output_features = OutputFeatures(features=[ContinuousOutput(key="y")])
    with pytest.raises(ValueError):
        Dummy(input_features=input_features, output_features=output_features)


def test_zero_output_features():
    input_features = InputFeatures(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(5)
        ]
    )
    output_features = OutputFeatures(features=[])
    with pytest.raises(ValueError):
        Dummy(input_features=input_features, output_features=output_features)


def test_predicted_value():
    # valid
    PredictedValue(predictedValue=5.0, standardDeviation=0.1)
    PredictedValue(predictedValue=5.0, standardDeviation=0.0)
    # invalid
    with pytest.raises(ValidationError):
        PredictedValue(value=5.0, standardDeviation=-0.1)


@pytest.mark.parametrize("n_output_features", [1, 2])
def test_to_outputs(n_output_features):
    input_features = InputFeatures(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(5)
        ]
    )
    output_features = OutputFeatures(
        features=[ContinuousOutput(key=f"y_{i+1}") for i in range(n_output_features)]
    )
    model = Dummy(input_features=input_features, output_features=output_features)
    preds = model.predict(input_features.sample(10))
    output = model.to_predictions(preds)
    assert len(output) == n_output_features
    assert sorted(output.keys()) == [f"y_{i+1}" for i in range(n_output_features)]
    for key in [f"y_{i+1}" for i in range(n_output_features)]:
        assert len(output[key]) == 10
