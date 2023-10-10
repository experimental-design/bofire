from typing import List

import gpytorch
import torch
from gpytorch.kernels import Kernel as GpytorchKernel

import bofire.data_models.kernels.api as data_models
import bofire.priors.api as priors
from bofire.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel


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
        lengthscale_prior=priors.map(data_model.lengthscale_prior)
        if data_model.lengthscale_prior is not None
        else None,
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
        lengthscale_prior=priors.map(data_model.lengthscale_prior)
        if data_model.lengthscale_prior is not None
        else None,
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
        variance_prior=priors.map(data_model.variance_prior)
        if data_model.variance_prior is not None
        else None,
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
        offset_prior=priors.map(data_model.offset_prior)
        if data_model.offset_prior is not None
        else None,
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
        ]
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
        ]
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
        outputscale_prior=priors.map(data_model.outputscale_prior)
        if data_model.outputscale_prior is not None
        else None,
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


KERNEL_MAP = {
    data_models.RBFKernel: map_RBFKernel,
    data_models.MaternKernel: map_MaternKernel,
    data_models.LinearKernel: map_LinearKernel,
    data_models.PolynomialKernel: map_PolynomialKernel,
    data_models.AdditiveKernel: map_AdditiveKernel,
    data_models.MultiplicativeKernel: map_MultiplicativeKernel,
    data_models.ScaleKernel: map_ScaleKernel,
    data_models.TanimotoKernel: map_TanimotoKernel,
}


def map(
    data_model: data_models.AnyKernel,
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> GpytorchKernel:
    return KERNEL_MAP[data_model.__class__](  # type: ignore
        data_model, batch_shape, ard_num_dims, active_dims
    )
