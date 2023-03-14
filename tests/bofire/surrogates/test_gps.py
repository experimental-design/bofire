import pytest
import torch
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputStandardize,
    Normalize,
    OneHotToNumeric,
)
from botorch.models.transforms.outcome import Standardize
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import HammondDistanceKernel, RBFKernel, ScaleKernel
from bofire.data_models.surrogates.api import (
    MixedSingleTaskGPSurrogate,
    ScalerEnum,
    SingleTaskGPSurrogate,
)


@pytest.mark.parametrize(
    "kernel, scaler",
    [
        (ScaleKernel(base_kernel=RBFKernel(ard=True)), ScalerEnum.NORMALIZE),
        (ScaleKernel(base_kernel=RBFKernel(ard=False)), ScalerEnum.STANDARDIZE),
    ],
)
def test_SingleTaskGPModel(kernel, scaler):
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
    )
    output_features = Outputs(features=[ContinuousOutput(key="y")])
    experiments = input_features.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    model = SingleTaskGPSurrogate(
        input_features=input_features,
        output_features=output_features,
        kernel=kernel,
        scaler=scaler,
    )
    model = surrogates.map(model)
    samples = input_features.sample(5)
    # test error on non fitted model
    with pytest.raises(ValueError):
        model.predict(samples)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    preds = model.predict(samples)
    assert preds.shape == (5, 2)
    # check that model is composed correctly
    assert isinstance(model.model, SingleTaskGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, Normalize)
    else:
        assert isinstance(model.model.input_transform, InputStandardize)
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = SingleTaskGPSurrogate(
        input_features=input_features,
        output_features=output_features,
        kernel=kernel,
        scaler=scaler,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(samples)
    assert_frame_equal(preds, preds2)


def test_MixedGPModel_invalid_preprocessing():
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
    )
    output_features = Outputs(features=[ContinuousOutput(key="y")])
    experiments = input_features.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    with pytest.raises(ValidationError):
        MixedSingleTaskGPSurrogate(
            input_features=input_features,
            output_features=output_features,
        )


@pytest.mark.parametrize(
    "kernel, scaler",
    [
        (RBFKernel(ard=True), ScalerEnum.NORMALIZE),
        (RBFKernel(ard=False), ScalerEnum.STANDARDIZE),
    ],
)
def test_MixedGPModel(kernel, scaler):
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
    )
    output_features = Outputs(features=[ContinuousOutput(key="y")])
    experiments = input_features.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments.loc[experiments.x_cat == "mama", "y"] *= 5.0
    experiments.loc[experiments.x_cat == "papa", "y"] /= 2.0
    experiments["valid_y"] = 1

    model = MixedSingleTaskGPSurrogate(
        input_features=input_features,
        output_features=output_features,
        input_preprocessing_specs={"x_cat": CategoricalEncodingEnum.ONE_HOT},
        scaler=scaler,
        continuous_kernel=kernel,
        categorical_kernel=HammondDistanceKernel(),
    )
    model = surrogates.map(model)

    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    samples = input_features.sample(5)
    preds = model.predict(samples)
    assert preds.shape == (5, 2)
    # check that model is composed correctly
    assert isinstance(model.model, MixedSingleTaskGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    assert isinstance(model.model.input_transform, ChainedInputTransform)
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform.tf1, Normalize)
    else:
        assert isinstance(model.model.input_transform.tf1, InputStandardize)
    assert torch.eq(
        model.model.input_transform.tf1.indices, torch.tensor([0, 1], dtype=torch.int64)
    ).all()
    assert isinstance(model.model.input_transform.tf2, OneHotToNumeric)
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = SingleTaskGPSurrogate(
        input_features=input_features,
        output_features=output_features,
        kernel=kernel,
        scaler=scaler,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(samples)
    assert_frame_equal(preds, preds2)
