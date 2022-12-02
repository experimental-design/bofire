import random

import pytest
import torch
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.input import InputStandardize, Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, RBFKernel

from bofire.strategies.botorch.utils.models import (
    ContKernelFactory,
    get_and_fit_model,
    get_dim_subsets,
)
from bofire.utils.enum import KernelEnum, ScalerEnum
from bofire.utils.torch_tools import tkwargs


@pytest.mark.parametrize(
    "kernel_name, expected_kernel",
    [("MATERN_25", MaternKernel), ("MATERN_15", MaternKernel), ("RBF", RBFKernel)],
)
def test_contKernelFactory(kernel_name, expected_kernel):

    kernel = ContKernelFactory(kernel=kernel_name, use_ard=False)

    assert isinstance(kernel(), expected_kernel)
    assert isinstance(
        kernel.to_mixedGP(torch.Size(), ard_num_dims=2, active_dims=[1, 2]),
        expected_kernel,
    )

    kernel = ContKernelFactory(
        kernel=kernel_name, use_ard=True, active_dims=random.sample(range(0, 10), 5)
    )

    assert isinstance(kernel(), expected_kernel)
    assert isinstance(
        kernel.to_mixedGP(torch.Size(), ard_num_dims=2, active_dims=[1, 2]),
        expected_kernel,
    )


@pytest.mark.parametrize(
    "kernel_name, expected_kernel",
    [("MATERN_25", MaternKernel), ("MATERN_15", MaternKernel), ("RBF", RBFKernel)],
)
def test_contKernelFactory_invalid(kernel_name, expected_kernel):

    with pytest.raises(TypeError):
        kernel = ContKernelFactory(kernel=kernel_name, use_ard=True)
        assert isinstance(kernel(), expected_kernel)
        assert isinstance(
            kernel.to_mixedGP(torch.Size(), ard_num_dims=2, active_dims=[1, 2]),
            expected_kernel,
        )


@pytest.mark.parametrize(
    "d, active_dims, cat_dims",
    [
        (5, [], []),
        (5, [1, 1, 2], []),
        (5, [-1, 2, 3], []),
        (5, [1, 2, 3, 5], []),
        (5, [0, 1, 2, 3, 5, 6], []),
    ],
)
def test_invalid_get_dim_subsets(d, active_dims, cat_dims):
    with pytest.raises((ValueError, TypeError, KeyError)):
        get_dim_subsets(d, active_dims, cat_dims)


@pytest.mark.parametrize(
    "d, active_dims, cat_dims, expected",
    [
        (3, [0, 1, 2], [], [[0, 1, 2], [0, 1, 2], []]),
        (3, [0, 1, 2], [2], [[0, 1], [0, 1], [2]]),
        (3, [1, 2], [2], [[0, 1], [1], [2]]),
        (3, [1], [2], [[0, 1], [1], []]),
    ],
)
def test_valid_get_dim_subsets(d, active_dims, cat_dims, expected):
    ord_dims, ord_active_dims, cat_active_dims = get_dim_subsets(
        d, active_dims, cat_dims
    )
    assert ord_dims == expected[0]
    assert ord_active_dims == expected[1]
    assert cat_active_dims == expected[2]


train_X = torch.cat([torch.rand(20, 2), torch.randint(3, (20, 1))], dim=-1).to(
    **tkwargs
)
train_Y = (
    torch.sin(train_X[..., :-1]).sum(dim=1, keepdim=True) + train_X[..., -1:]
).to(**tkwargs)


@pytest.mark.parametrize(
    "train_X, train_Y, active_dims, cat_dims, scaler_name, kernel_name, use_categorical_kernel",
    [
        (
            train_X,
            train_Y,
            [0, 1, 2],
            [],
            ScalerEnum.NORMALIZE,
            KernelEnum.MATERN_25,
            False,
        ),
        (
            train_X,
            train_Y,
            [0, 1, 2],
            [],
            ScalerEnum.NORMALIZE,
            KernelEnum.MATERN_25,
            False,
        ),
        (
            train_X,
            train_Y,
            [1, 2],
            [2],
            ScalerEnum.NORMALIZE,
            KernelEnum.MATERN_25,
            False,
        ),
    ],
)
def test_get_and_fit_model(
    train_X,
    train_Y,
    active_dims,
    cat_dims,
    scaler_name,
    kernel_name,
    use_categorical_kernel,
):
    model = get_and_fit_model(
        train_X=train_X,
        train_Y=train_Y,
        active_dims=active_dims,
        cat_dims=cat_dims,
        scaler_name=scaler_name,
        kernel_name=kernel_name,
        use_categorical_kernel=use_categorical_kernel,
    )

    if scaler_name == ScalerEnum.NORMALIZE:
        assert isinstance(model.input_transform, Normalize)
    elif scaler_name == ScalerEnum.STANDARDIZE:
        assert isinstance(model.input_transform, InputStandardize)

    assert isinstance(model.outcome_transform, Standardize)

    if len(cat_dims) != 0 and use_categorical_kernel is True:
        assert isinstance(model, MixedSingleTaskGP)
    else:
        assert isinstance(model, SingleTaskGP)
