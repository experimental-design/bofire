import pytest
import torch
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputStandardize,
    Normalize,
)
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, RBFKernel

from bofire.domain.features import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    InputFeatures,
    OutputFeatures,
)
from bofire.models.torch_models import (
    RBF,
    ContinuousKernel,
    HammondDistanceKernel,
    Matern,
    MixedSingleTaskGPModel,
    SingleTaskGPModel,
)
from bofire.utils.enum import CategoricalEncodingEnum, ScalerEnum


@pytest.mark.parametrize(
    "kernel, ard_num_dims, active_dims, expected_kernel",
    [
        (RBF(ard=False), 10, list(range(5)), RBFKernel),
        (RBF(ard=True), 10, list(range(5)), RBFKernel),
        (Matern(ard=False), 10, list(range(5)), MaternKernel),
        (Matern(ard=True), 10, list(range(5)), MaternKernel),
        (Matern(ard=False, nu=2.5), 10, list(range(5)), MaternKernel),
        (Matern(ard=True, nu=1.5), 10, list(range(5)), MaternKernel),
    ],
)
def test_continuous_kernel(
    kernel: ContinuousKernel, ard_num_dims, active_dims, expected_kernel
):
    k = kernel.to_gpytorch(
        batch_shape=torch.Size(), ard_num_dims=ard_num_dims, active_dims=active_dims
    )
    assert isinstance(k, expected_kernel)
    if kernel.ard is False:
        assert k.ard_num_dims is None
    else:
        assert k.ard_num_dims == len(active_dims)
    assert torch.eq(k.active_dims, torch.tensor(active_dims, dtype=torch.int64)).all()

    if isinstance(kernel, Matern):
        assert kernel.nu == k.nu


@pytest.mark.parametrize(
    "kernel, scaler",
    [(RBF(ard=True), ScalerEnum.NORMALIZE), (RBF(ard=False), ScalerEnum.STANDARDIZE)],
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
    model.fit(experiments)
    preds = model.predict(experiments)
    assert preds.shape == (10, 2)
    # check that model is composed correctly
    assert isinstance(model.model, SingleTaskGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, Normalize)
    else:
        assert isinstance(model.model.input_transform, InputStandardize)


@pytest.mark.parametrize(
    "kernel, scaler",
    [(RBF(ard=True), ScalerEnum.NORMALIZE), (RBF(ard=False), ScalerEnum.STANDARDIZE)],
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
    preds = model.predict(experiments)
    assert preds.shape == (10, 2)
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
