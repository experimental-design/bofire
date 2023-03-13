from typing import Literal

import numpy as np
import pandas as pd
import pytest
from pydantic.error_wrappers import ValidationError

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.surrogates.api import Surrogate as SurrogateDataModel
from bofire.surrogates.api import PredictedValue, Surrogate


class DummyDataModel(SurrogateDataModel):
    type: Literal["Dummy"] = "Dummy"


class Dummy(Surrogate):
    def __init__(
        self,
        data_model: SurrogateDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)

    def _predict(self, transformed_X: pd.DataFrame):
        preds = np.random.normal(
            loc=5, scale=1, size=(len(transformed_X), len(self.output_features))
        )
        stds = np.random.uniform(
            low=0.0, high=1.0, size=(len(transformed_X), len(self.output_features))
        )
        return preds, stds

    def dumps(self):
        pass

    def loads(self, dumpstr: str):
        pass


def test_zero_input_features():
    input_features = Inputs(features=[])
    output_features = Outputs(features=[ContinuousOutput(key="y")])
    with pytest.raises(ValueError):
        data_model = DummyDataModel(
            input_features=input_features, output_features=output_features
        )
        Dummy(data_model=data_model)


def test_zero_output_features():
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(5)
        ]
    )
    output_features = Outputs(features=[])
    with pytest.raises(ValueError):
        data_model = DummyDataModel(
            input_features=input_features, output_features=output_features
        )
        Dummy(data_model=data_model)


def test_predicted_value():
    # valid
    PredictedValue(predictedValue=5.0, standardDeviation=0.1)
    PredictedValue(predictedValue=4.2, standardDeviation=0.01)
    # invalid
    with pytest.raises(ValidationError):
        PredictedValue(predictedValue=5.0, standardDeviation=-0.1)


@pytest.mark.parametrize("n_output_features", [1, 2])
def test_to_outputs(n_output_features):
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(5)
        ]
    )
    output_features = Outputs(
        features=[ContinuousOutput(key=f"y_{i+1}") for i in range(n_output_features)]
    )
    data_model = DummyDataModel(
        input_features=input_features, output_features=output_features
    )
    model = Dummy(data_model=data_model)

    model.model = "dummymodel"
    preds = model.predict(input_features.sample(10))
    output = model.to_predictions(preds)
    assert len(output) == n_output_features
    assert sorted(output.keys()) == [f"y_{i+1}" for i in range(n_output_features)]
    for key in [f"y_{i+1}" for i in range(n_output_features)]:
        assert len(output[key]) == 10


def test_is_fitted():
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(5)
        ]
    )

    output_features = Outputs(features=[ContinuousOutput(key="y")])
    data_model = DummyDataModel(
        input_features=input_features, output_features=output_features
    )
    d = Dummy(data_model=data_model)
    assert d.is_fitted is False
    d.model = "dummymodel"
    assert d.is_fitted is True
