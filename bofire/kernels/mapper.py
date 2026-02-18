from typing import Callable, List, Optional

import gpytorch
import torch
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.kernels.infinite_width_bnn import InfiniteWidthBNNKernel
from botorch.models.kernels.positive_index import PositiveIndexKernel
from gpytorch.kernels import IndexKernel
from gpytorch.kernels import Kernel as GpytorchKernel

import bofire.data_models.kernels.api as data_models
import bofire.priors.api as priors
from bofire.kernels.aggregation import PolynomialFeatureInteractionKernel
from bofire.kernels.conditional import (
    WedgeKernel,
    build_indicator_func,
    compute_base_kernel_active_dims,
)
from bofire.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from bofire.kernels.shape import WassersteinKernel
from bofire.kernels.spherical_kernels import SphericalLinearKernel


def _compute_active_dims(
    data_model: data_models.FeatureSpecificKernel,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> List[int]:
    if data_model.features:
        if features_to_idx_mapper is None:
            raise RuntimeError(
                "features_to_idx_mapper must be defined when using only a subset of features"
            )
        active_dims = features_to_idx_mapper(data_model.features)
    return active_dims


def map_RBFKernel(
    data_model: data_models.RBFKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> gpytorch.kernels.RBFKernel:
    active_dims = _compute_active_dims(data_model, active_dims, features_to_idx_mapper)
    return gpytorch.kernels.RBFKernel(
        batch_shape=batch_shape,
        ard_num_dims=len(active_dims) if data_model.ard else None,
        active_dims=active_dims,
        lengthscale_prior=(
            priors.map(data_model.lengthscale_prior, d=len(active_dims))
            if data_model.lengthscale_prior is not None
            else None
        ),
        lengthscale_constraint=(
            priors.map(data_model.lengthscale_constraint)
            if data_model.lengthscale_constraint is not None
            else None
        ),
    )


def map_MaternKernel(
    data_model: data_models.MaternKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> gpytorch.kernels.MaternKernel:
    active_dims = _compute_active_dims(data_model, active_dims, features_to_idx_mapper)
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
        lengthscale_constraint=(
            priors.map(data_model.lengthscale_constraint)
            if data_model.lengthscale_constraint is not None
            else None
        ),
    )


def map_InfiniteWidthBNNKernel(
    data_model: data_models.InfiniteWidthBNNKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> InfiniteWidthBNNKernel:
    active_dims = _compute_active_dims(data_model, active_dims, features_to_idx_mapper)
    return InfiniteWidthBNNKernel(
        batch_shape=batch_shape,
        active_dims=tuple(active_dims),
        depth=data_model.depth,
    )


def map_LinearKernel(
    data_model: data_models.LinearKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> gpytorch.kernels.LinearKernel:
    active_dims = _compute_active_dims(data_model, active_dims, features_to_idx_mapper)
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
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> gpytorch.kernels.PolynomialKernel:
    active_dims = _compute_active_dims(data_model, active_dims, features_to_idx_mapper)
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
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> gpytorch.kernels.AdditiveKernel:
    return gpytorch.kernels.AdditiveKernel(
        *[
            map(
                data_model=k,
                batch_shape=batch_shape,
                active_dims=active_dims,
                features_to_idx_mapper=features_to_idx_mapper,
            )
            for k in data_model.kernels
        ],
    )


def map_MultiplicativeKernel(
    data_model: data_models.MultiplicativeKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> gpytorch.kernels.ProductKernel:
    return gpytorch.kernels.ProductKernel(
        *[
            map(
                data_model=k,
                batch_shape=batch_shape,
                active_dims=active_dims,
                features_to_idx_mapper=features_to_idx_mapper,
            )
            for k in data_model.kernels
        ],
    )


def map_ScaleKernel(
    data_model: data_models.ScaleKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> gpytorch.kernels.ScaleKernel:
    return gpytorch.kernels.ScaleKernel(
        base_kernel=map(
            data_model.base_kernel,
            batch_shape=batch_shape,
            active_dims=active_dims,
            features_to_idx_mapper=features_to_idx_mapper,
        ),
        outputscale_prior=(
            priors.map(data_model.outputscale_prior)
            if data_model.outputscale_prior is not None
            else None
        ),
        outputscale_constraint=(
            priors.map(data_model.outputscale_constraint)
            if data_model.outputscale_constraint is not None
            else None
        ),
    )


def map_TanimotoKernel(
    data_model: data_models.TanimotoKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> TanimotoKernel:
    active_dims = _compute_active_dims(data_model, active_dims, features_to_idx_mapper)
    return TanimotoKernel(
        batch_shape=batch_shape,
        ard_num_dims=len(active_dims) if data_model.ard else None,
        active_dims=active_dims,
    )


def map_HammingDistanceKernel(
    data_model: data_models.HammingDistanceKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> GpytorchKernel:
    active_dims = _compute_active_dims(data_model, active_dims, features_to_idx_mapper)
    return CategoricalKernel(
        batch_shape=batch_shape,
        ard_num_dims=len(active_dims) if data_model.ard else None,
        active_dims=active_dims,
        lengthscale_prior=(
            priors.map(data_model.lengthscale_prior, d=len(active_dims))
            if data_model.lengthscale_prior is not None
            else None
        ),
        lengthscale_constraint=(
            priors.map(data_model.lengthscale_constraint)
            if data_model.lengthscale_constraint is not None
            else None
        ),
    )


def map_IndexKernel(
    data_model: data_models.IndexKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> GpytorchKernel:
    active_dims = _compute_active_dims(data_model, active_dims, features_to_idx_mapper)
    return IndexKernel(
        batch_shape=batch_shape,
        num_tasks=data_model.num_categories,
        rank=data_model.rank,
        active_dims=active_dims,
        prior=(priors.map(data_model.prior) if data_model.prior is not None else None),
        var_constraint=(
            priors.map(data_model.var_constraint)
            if data_model.var_constraint is not None
            else None
        ),
    )


def map_PositiveIndexKernel(
    data_model: data_models.PositiveIndexKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> GpytorchKernel:
    active_dims = _compute_active_dims(data_model, active_dims, features_to_idx_mapper)
    return PositiveIndexKernel(
        batch_shape=batch_shape,
        num_tasks=data_model.num_categories,
        rank=data_model.rank,
        active_dims=active_dims,
        task_prior=(
            priors.map(data_model.task_prior)
            if data_model.task_prior is not None
            else None
        ),
        diag_prior=(
            priors.map(data_model.diag_prior)
            if data_model.diag_prior is not None
            else None
        ),
        normalize_covar_matrix=data_model.normalize_covar_matrix,
        var_constraint=(
            priors.map(data_model.var_constraint)
            if data_model.var_constraint is not None
            else None
        ),
        target_task_index=data_model.target_task_index,
        unit_scale_for_target=data_model.unit_scale_for_target,
    )


def map_WassersteinKernel(
    data_model: data_models.WassersteinKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
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


def map_PolynomialFeatureInteractionKernel(
    data_model: data_models.PolynomialFeatureInteractionKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> PolynomialFeatureInteractionKernel:
    ks = [
        map(
            k,
            active_dims=active_dims,
            batch_shape=batch_shape,
            features_to_idx_mapper=features_to_idx_mapper,
        )
        for k in data_model.kernels
    ]

    return PolynomialFeatureInteractionKernel(
        ks,
        max_degree=data_model.max_degree,
        include_self_interactions=data_model.include_self_interactions,
        outputscale_prior=(
            priors.map(data_model.outputscale_prior, d=len(active_dims))
            if data_model.outputscale_prior is not None
            else None
        ),
    )


def map_WedgeKernel(
    data_model: data_models.WedgeKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> WedgeKernel:
    indicator_func = build_indicator_func(data_model.conditions, features_to_idx_mapper)
    base_kernel_active_dims = compute_base_kernel_active_dims(
        data_model, active_dims, features_to_idx_mapper
    )
    # we compute the active dimensions above, then remove `features` from the base_kernel
    # to avoid re-filtering the active dimensions.
    base_kernel_data_model = data_model.base_kernel.model_copy(
        update={"features": None}
    )
    base_kernel = map(
        data_model=base_kernel_data_model,
        batch_shape=batch_shape,
        active_dims=base_kernel_active_dims,
        features_to_idx_mapper=features_to_idx_mapper,
    )
    return WedgeKernel(
        base_kernel,
        indicator_func,
        lengthscale_prior=(
            priors.map(data_model.lengthscale_prior, d=len(active_dims))
            if data_model.lengthscale_prior is not None
            else None
        ),
        lengthscale_constraint=(
            priors.map(data_model.lengthscale_constraint)
            if data_model.lengthscale_constraint is not None
            else None
        ),
        angle_prior=(
            priors.map(data_model.angle_prior, d=len(active_dims))
            if data_model.angle_prior is not None
            else None
        ),
        radius_prior=(
            priors.map(data_model.radius_prior, d=len(active_dims))
            if data_model.radius_prior is not None
            else None
        ),
    )


def map_SphericalLinearKernel(
    data_model: data_models.SphericalLinearKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> SphericalLinearKernel:
    active_dims = _compute_active_dims(data_model, active_dims, features_to_idx_mapper)
    return SphericalLinearKernel(
        batch_shape=batch_shape,
        ard_num_dims=len(active_dims) if data_model.ard else None,
        active_dims=active_dims,
        lengthscale_prior=(
            priors.map(data_model.lengthscale_prior, d=len(active_dims))
            if data_model.lengthscale_prior is not None
            else None
        ),
        lengthscale_constraint=(
            priors.map(data_model.lengthscale_constraint)
            if data_model.lengthscale_constraint is not None
            else None
        ),
        bounds=data_model.bounds,
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
    data_models.SphericalLinearKernel: map_SphericalLinearKernel,
    data_models.TanimotoKernel: map_TanimotoKernel,
    data_models.HammingDistanceKernel: map_HammingDistanceKernel,
    data_models.IndexKernel: map_IndexKernel,
    data_models.PositiveIndexKernel: map_PositiveIndexKernel,
    data_models.InfiniteWidthBNNKernel: map_InfiniteWidthBNNKernel,
    data_models.PolynomialFeatureInteractionKernel: map_PolynomialFeatureInteractionKernel,
    data_models.WedgeKernel: map_WedgeKernel,
}


def map(
    data_model: data_models.AnyKernel,
    batch_shape: torch.Size,
    active_dims: List[int],
    features_to_idx_mapper: Optional[Callable[[List[str]], List[int]]],
) -> GpytorchKernel:
    return KERNEL_MAP[data_model.__class__](
        data_model,
        batch_shape,
        active_dims,
        features_to_idx_mapper,
    )
