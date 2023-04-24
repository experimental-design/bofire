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
    CategoricalDescriptorInput,
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
from bofire.surrogates.single_task_gp import get_scaler
from bofire.utils.torch_tools import tkwargs


@pytest.mark.parametrize(
    "scaler_enum, input_preprocessing_specs, expected_scaler, expected_indices, expected_offset, expected_coefficient",
    [
        (
            ScalerEnum.NORMALIZE,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.ONE_HOT,
            },
            Normalize,
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([-4.0, -4.0]).to(**tkwargs),
            torch.tensor([8.0, 8.0]).to(**tkwargs),
        ),
        (
            ScalerEnum.NORMALIZE,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.DESCRIPTOR,
            },
            Normalize,
            torch.tensor([0, 1, 2], dtype=torch.int64),
            torch.tensor([-4.0, -4.0, 1.0]).to(**tkwargs),
            torch.tensor([8.0, 8.0, 5.0]).to(**tkwargs),
        ),
        (
            ScalerEnum.STANDARDIZE,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.ONE_HOT,
            },
            InputStandardize,
            torch.tensor([0, 1], dtype=torch.int64),
            None,
            None,
        ),
        (
            ScalerEnum.STANDARDIZE,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.DESCRIPTOR,
            },
            InputStandardize,
            torch.tensor([0, 1, 2], dtype=torch.int64),
            None,
            None,
        ),
    ],
)
def test_get_scaler(
    scaler_enum,
    input_preprocessing_specs,
    expected_scaler,
    expected_indices,
    expected_offset,
    expected_coefficient,
):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [
            CategoricalInput(key="x_cat", categories=["mama", "papa"]),
            CategoricalDescriptorInput(
                key="x_desc",
                categories=["alpha", "beta"],
                descriptors=["oskar"],
                values=[[1], [6]],
            ),
        ]
    )
    experiments = inputs.sample(n=10)
    scaler = get_scaler(
        inputs=inputs,
        input_preprocessing_specs=input_preprocessing_specs,
        scaler=scaler_enum,
        X=experiments[inputs.get_keys()],
    )
    assert isinstance(scaler, expected_scaler)
    assert (scaler.indices == expected_indices).all()
    if expected_offset is not None:
        assert torch.allclose(scaler.offset, expected_offset)
        assert torch.allclose(scaler.coefficient, expected_coefficient)


@pytest.mark.parametrize(
    "kernel, scaler",
    [
        (ScaleKernel(base_kernel=RBFKernel(ard=True)), ScalerEnum.NORMALIZE),
        (ScaleKernel(base_kernel=RBFKernel(ard=False)), ScalerEnum.STANDARDIZE),
    ],
)
def test_SingleTaskGPModel(kernel, scaler):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
    )
    model = surrogates.map(model)
    samples = inputs.sample(5)
    # test error on non fitted model
    with pytest.raises(ValueError):
        model.predict(samples)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    samples2 = samples.copy()
    samples2 = samples2.astype({"x_1": "object"})
    preds = model.predict(samples2)
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
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(samples)
    assert_frame_equal(preds, preds2)


def test_MixedGPModel_invalid_preprocessing():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    with pytest.raises(ValidationError):
        MixedSingleTaskGPSurrogate(
            inputs=inputs,
            outputs=outputs,
        )


@pytest.mark.parametrize(
    "kernel, scaler",
    [
        (RBFKernel(ard=True), ScalerEnum.NORMALIZE),
        (RBFKernel(ard=False), ScalerEnum.STANDARDIZE),
    ],
)
def test_MixedGPModel(kernel, scaler):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments.loc[experiments.x_cat == "mama", "y"] *= 5.0
    experiments.loc[experiments.x_cat == "papa", "y"] /= 2.0
    experiments["valid_y"] = 1

    model = MixedSingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        input_preprocessing_specs={"x_cat": CategoricalEncodingEnum.ONE_HOT},
        scaler=scaler,
        continuous_kernel=kernel,
        categorical_kernel=HammondDistanceKernel(),
    )
    model = surrogates.map(model)
    with pytest.raises(ValueError):
        model.dumps()
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    samples = inputs.sample(5)
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
    model2 = MixedSingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(samples)
    assert_frame_equal(preds, preds2)
