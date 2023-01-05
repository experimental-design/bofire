import pandas as pd
import pytest

from bofire.domain.features import (
    ContinuousInput,
    ContinuousOutput,
    InputFeatures,
    OutputFeatures,
)
from bofire.models.model import Model


class Dummy(Model):
    def _predict(self, transformed_X: pd.DataFrame):
        pass


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
