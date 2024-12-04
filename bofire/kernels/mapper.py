from typing import List

import gpytorch
import torch
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.kernels import Kernel as GpytorchKernel

import bofire.data_models.kernels.api as data_models
import bofire.priors.api as priors
from bofire.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from bofire.kernels.shape import WassersteinKernel


def map_RBFKernel(
    data_model: data_models.RBFKernel,
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> gpytorch.kernels.RBFKernel:
    return gpytorch.kernels.RBFKernel(
        batch_shape=batch_shape,
        ard_num_dims=len(active_dims) if data_model.ard else None,
        active_dims=active_dims,  # type: ignore
        lengthscale_prior=(
            priors.map(data_model.lengthscale_prior, d=len(active_dims))
            if data_model.lengthscale_prior is not None
            else None
        ),
    )


def map_MaternKernel(
    data_model: data_models.MaternKernel,
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> gpytorch.kernels.MaternKernel:
    return gpytorch.kernels.MaternKernel(
        batch_shape=batch_shape,
        ard_num_dims=len(active_dims) if data_model.ard else None,
        active_dims=active_dims,
        nu=data_model.nu,
        lengthscale_prior=(
            priors.map(data_model.lengthscale_prior, d=len(active_dims))
            if data_model.lengthscale_prior is not None
            else None
        ),
    )


def map_InfiniteWidthBNNKernel(
    data_model: data_models.InfiniteWidthBNNKernel,
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> "InfiniteWidthBNNKernel":  # type: ignore # noqa: F821
    try:
        from botorch.models.kernels.infinite_width_bnn import (  # type: ignore
            InfiniteWidthBNNKernel,
        )

    except ImportError:
        raise ImportError(
            "InfiniteWidthBNNKernel requires botorch>=0.11.3 to be installed. "
            "This can be installed by running `pip install 'botorch>=0.11.3'`, "
            "requires python 3.10+.",
        )

    return InfiniteWidthBNNKernel(
        batch_shape=batch_shape,
        active_dims=tuple(active_dims),
        depth=data_model.depth,
    )


def map_LinearKernel(
    data_model: data_models.LinearKernel,
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> gpytorch.kernels.LinearKernel:
    return gpytorch.kernels.LinearKernel(
        batch_shape=batch_shape,
        active_dims=active_dims,
        variance_prior=(
            priors.map(data_model.variance_prior)
            if data_model.variance_prior is not None
            else None
        ),
    )


def map_PolynomialKernel(
    data_model: data_models.PolynomialKernel,
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> gpytorch.kernels.PolynomialKernel:
    return gpytorch.kernels.PolynomialKernel(
        batch_shape=batch_shape,
        active_dims=active_dims,
        power=data_model.power,
        offset_prior=(
            priors.map(data_model.offset_prior)
            if data_model.offset_prior is not None
            else None
        ),
    )


def map_AdditiveKernel(
    data_model: data_models.AdditiveKernel,
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> gpytorch.kernels.AdditiveKernel:
    return gpytorch.kernels.AdditiveKernel(
        *[  # type: ignore
            map(
                data_model=k,
                batch_shape=batch_shape,
                ard_num_dims=ard_num_dims,
                active_dims=active_dims,
            )
            for k in data_model.kernels
        ],
    )


def map_MultiplicativeKernel(
    data_model: data_models.MultiplicativeKernel,
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> gpytorch.kernels.ProductKernel:
    return gpytorch.kernels.ProductKernel(
        *[  # type: ignore
            map(
                data_model=k,
                batch_shape=batch_shape,
                ard_num_dims=ard_num_dims,
                active_dims=active_dims,
            )
            for k in data_model.kernels
        ],
    )


def map_ScaleKernel(
    data_model: data_models.ScaleKernel,
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> gpytorch.kernels.ScaleKernel:
    return gpytorch.kernels.ScaleKernel(
        base_kernel=map(
            data_model.base_kernel,
            batch_shape=batch_shape,
            ard_num_dims=ard_num_dims,
            active_dims=active_dims,
        ),
        outputscale_prior=(
            priors.map(data_model.outputscale_prior)
            if data_model.outputscale_prior is not None
            else None
        ),
    )


def map_TanimotoKernel(
    data_model: data_models.TanimotoKernel,
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> TanimotoKernel:
    return TanimotoKernel(
        batch_shape=batch_shape,
        ard_num_dims=len(active_dims) if data_model.ard else None,
        active_dims=active_dims,
    )


def map_HammingDistanceKernel(
    data_model: data_models.HammingDistanceKernel,
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> CategoricalKernel:
    return CategoricalKernel(
        batch_shape=batch_shape,
        ard_num_dims=len(active_dims) if data_model.ard else None,
        active_dims=active_dims,  # type: ignore
    )


def map_WassersteinKernel(
    data_model: data_models.WassersteinKernel,
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> WassersteinKernel:
    return WassersteinKernel(
        squared=data_model.squared,
        lengthscale_prior=(
            priors.map(data_model.lengthscale_prior, d=len(active_dims))
            if data_model.lengthscale_prior is not None
            else None
        ),
        active_dims=active_dims,
    )


KERNEL_MAP = {
    data_models.WassersteinKernel: map_WassersteinKernel,
    data_models.RBFKernel: map_RBFKernel,
    data_models.MaternKernel: map_MaternKernel,
    data_models.LinearKernel: map_LinearKernel,
    data_models.PolynomialKernel: map_PolynomialKernel,
    data_models.AdditiveKernel: map_AdditiveKernel,
    data_models.MultiplicativeKernel: map_MultiplicativeKernel,
    data_models.ScaleKernel: map_ScaleKernel,
    data_models.TanimotoKernel: map_TanimotoKernel,
    data_models.HammingDistanceKernel: map_HammingDistanceKernel,
    data_models.InfiniteWidthBNNKernel: map_InfiniteWidthBNNKernel,
}


def map(
    data_model: data_models.AnyKernel,
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> GpytorchKernel:
    return KERNEL_MAP[data_model.__class__](
        data_model,
        batch_shape,
        ard_num_dims,
        active_dims,
    )
