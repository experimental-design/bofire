from typing import Literal, Type

import numpy as np
import pandas as pd
import pytest
from pydantic.error_wrappers import ValidationError

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import AnyOutput, ContinuousInput, ContinuousOutput
from bofire.data_models.surrogates.api import Surrogate as SurrogateDataModel
from bofire.surrogates.api import PredictedValue, Surrogate


class DummyDataModel(SurrogateDataModel):
    type: Literal["Dummy"] = "Dummy"

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models

        Args:
            outputs: objective functions for the surrogate
            my_type: continuous or categorical output

        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise

        """
        return True


class Dummy(Surrogate):
    def __init__(
        self,
        data_model: SurrogateDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)

    def _predict(self, transformed_X: pd.DataFrame):
        preds = np.random.normal(
            loc=5,
            scale=1,
            size=(len(transformed_X), len(self.outputs)),
        )
        stds = np.random.uniform(
            low=0.0,
            high=1.0,
            size=(len(transformed_X), len(self.outputs)),
        )
        return preds, stds

    def _dumps(self):
        pass

    def loads(self, dumpstr: str):
        pass


def test_zero_inputs():
    inputs = Inputs(features=[])
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    with pytest.raises(ValueError):
        data_model = DummyDataModel(inputs=inputs, outputs=outputs)
        Dummy(data_model=data_model)


def test_zero_outputs():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(5)
        ],
    )
    outputs = Outputs(features=[])
    with pytest.raises(ValueError):
        data_model = DummyDataModel(inputs=inputs, outputs=outputs)
        Dummy(data_model=data_model)


def test_predicted_value():
    # valid
    PredictedValue(predictedValue=5.0, standardDeviation=0.1)
    PredictedValue(predictedValue=4.2, standardDeviation=0.01)
    # invalid
    with pytest.raises(ValidationError):
        PredictedValue(predictedValue=5.0, standardDeviation=-0.1)


@pytest.mark.parametrize("n_outputs", [1, 2])
def test_to_outputs(n_outputs):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(5)
        ],
    )
    outputs = Outputs(
        features=[ContinuousOutput(key=f"y_{i+1}") for i in range(n_outputs)],
    )
    data_model = DummyDataModel(inputs=inputs, outputs=outputs)
    model = Dummy(data_model=data_model)

    model.model = "dummymodel"
    preds = model.predict(inputs.sample(10))
    output = model.to_predictions(preds)
    assert len(output) == n_outputs
    assert sorted(output.keys()) == [f"y_{i+1}" for i in range(n_outputs)]
    for key in [f"y_{i+1}" for i in range(n_outputs)]:
        assert len(output[key]) == 10


def test_is_fitted():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(5)
        ],
    )

    outputs = Outputs(features=[ContinuousOutput(key="y")])
    data_model = DummyDataModel(inputs=inputs, outputs=outputs)
    d = Dummy(data_model=data_model)
    assert d.is_fitted is False
    d.model = "dummymodel"
    assert d.is_fitted is True
