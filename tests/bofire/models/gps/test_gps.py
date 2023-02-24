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

from bofire.domain.feature import CategoricalInput, ContinuousInput, ContinuousOutput
from bofire.domain.features import InputFeatures, OutputFeatures
from bofire.models.gps.gps import MixedSingleTaskGPModel, SingleTaskGPModel
from bofire.models.gps.kernels import HammondDistanceKernel, RBFKernel, ScaleKernel
from bofire.utils.enum import CategoricalEncodingEnum, ScalerEnum


@pytest.mark.parametrize(
    "kernel, scaler",
    [
        (ScaleKernel(base_kernel=RBFKernel(ard=True)), ScalerEnum.NORMALIZE),
        (ScaleKernel(base_kernel=RBFKernel(ard=False)), ScalerEnum.STANDARDIZE),
    ],
)
def test_SingleTaskGPModel(kernel, scaler):
    input_features = InputFeatures(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
    )
    output_features = OutputFeatures(features=[ContinuousOutput(key="y")])
    experiments = input_features.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    model = SingleTaskGPModel(
        input_features=input_features,
        output_features=output_features,
        kernel=kernel,
        scaler=scaler,
    )
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
    model2 = SingleTaskGPModel(
        input_features=input_features,
        output_features=output_features,
        kernel=kernel,
        scaler=scaler,
    )
    model2.loads(dump)
    preds2 = model2.predict(samples)
    assert_frame_equal(preds, preds2)


@pytest.mark.parametrize(
    "kernel, scaler",
    [
        (RBFKernel(ard=True), ScalerEnum.NORMALIZE),
        (RBFKernel(ard=False), ScalerEnum.STANDARDIZE),
    ],
)
def test_MixedGPModel(kernel, scaler):
    input_features = InputFeatures(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
    )
    output_features = OutputFeatures(features=[ContinuousOutput(key="y")])
    experiments = input_features.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments.loc[experiments.x_cat == "mama", "y"] *= 5.0
    experiments.loc[experiments.x_cat == "papa", "y"] /= 2.0
    experiments["valid_y"] = 1

    model = MixedSingleTaskGPModel(
        input_features=input_features,
        output_features=output_features,
        input_preprocessing_specs={"x_cat": CategoricalEncodingEnum.ONE_HOT},
        scaler=scaler,
        continuous_kernel=kernel,
        categorical_kernel=HammondDistanceKernel(),
    )

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
    model2 = SingleTaskGPModel(
        input_features=input_features,
        output_features=output_features,
        kernel=kernel,
        scaler=scaler,
    )
    model2.loads(dump)
    preds2 = model2.predict(samples)
    assert_frame_equal(preds, preds2)
