from functools import partial
from typing import Callable, Dict, Optional, Type

import torch
from botorch.models.transforms.input import AppendFeatures

from bofire.data_models.api import Inputs
from bofire.data_models.features.api import (
    CloneFeature,
    EngineeredFeature,
    InterpolateFeature,
    MeanFeature,
    MolecularWeightedMeanFeature,
    MolecularWeightedSumFeature,
    ProductFeature,
    SumFeature,
    WeightedMeanFeature,
    WeightedSumFeature,
)
from bofire.data_models.types import InputTransformSpecs
from bofire.utils.torch_tools import interp1d


def register(
    data_model_cls: Type[EngineeredFeature],
    map_fn: Optional[Callable] = None,
):
    """Register a custom engineered feature mapping from data model to factory function.

    Can be used as a decorator or as a direct function call::

        # Decorator form
        @register(MyEngineeredFeatureDataModel)
        def map_my_feature(inputs, transform_specs, feature):
            return AppendFeatures(...)

        # Direct call form
        register(MyEngineeredFeatureDataModel, map_my_feature)

    Args:
        data_model_cls: The Pydantic data model class.
        map_fn: A callable that takes ``(inputs, transform_specs, feature)``
            and returns a ``botorch.models.transforms.input.AppendFeatures``
            instance. If not provided, returns a decorator.

    Returns:
        The mapping function (unchanged) when used as a decorator, None otherwise.
    """
    from bofire.data_models.features.api import register_engineered_feature

    def _register(fn: Callable) -> Callable:
        # Register with the data model union first so a discriminator conflict
        # is raised before the functional map is touched (no partial state).
        register_engineered_feature(data_model_cls)
        AGGREGATE_MAP[data_model_cls] = fn

        return fn

    if map_fn is not None:
        _register(map_fn)
        return None

    return _register


def _weighted_features(
    X: torch.Tensor,
    indices: torch.Tensor,
    descriptors: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    weights = X[..., indices]
    weighted_sum = torch.matmul(weights, descriptors)
    if normalize:
        weight_sum = torch.clamp(
            torch.sum(weights, dim=-1, keepdim=True),
            min=torch.finfo(weights.dtype).eps,
        )
        weighted_sum = weighted_sum / weight_sum
    return weighted_sum.unsqueeze(-2)


def _map_reduction_feature(
    inputs: Inputs,
    transform_specs: InputTransformSpecs,
    feature: EngineeredFeature,
    reducer: Callable,
) -> AppendFeatures:
    features2idx, _ = inputs._get_transform_info(transform_specs)
    indices = [features2idx[key][0] for key in feature.features]

    def reduce_features(
        X: torch.Tensor,
        indices: torch.Tensor,
        reducer: Callable,
    ) -> torch.Tensor:
        return reducer(X[..., indices], dim=-1, keepdim=True).unsqueeze(-2)

    return AppendFeatures(
        f=reduce_features,
        fkwargs={"indices": indices, "reducer": reducer},
        transform_on_train=True,
    )


def map_weighted_feature(
    inputs: Inputs,
    transform_specs: InputTransformSpecs,
    feature: WeightedSumFeature,
) -> AppendFeatures:
    features2idx, _ = inputs._get_transform_info(transform_specs)
    indices = [features2idx[key][0] for key in feature.features]
    # one descriptor row per component; the spec filters correlated descriptors
    # once over the combined component structures (when enabled).
    components = [inputs.get_by_key(key) for key in feature.features]
    matrix = feature.component_table(components)
    descriptors = torch.tensor(matrix.values, dtype=torch.double)
    return AppendFeatures(
        f=_weighted_features,
        fkwargs={
            "indices": indices,
            "descriptors": descriptors,
            "normalize": feature.normalize,
        },
        transform_on_train=True,
    )


def _interpolate_features(
    X: torch.Tensor,
    idx_x: torch.Tensor,
    idx_y: torch.Tensor,
    new_x: torch.Tensor,
    prepend_x: torch.Tensor,
    prepend_y: torch.Tensor,
    append_x: torch.Tensor,
    append_y: torch.Tensor,
    normalize_y: float,
    normalize_x: bool,
) -> torch.Tensor:
    x = X[..., idx_x]
    y = X[..., idx_y]

    if prepend_x.numel() > 0:
        px = prepend_x.expand(*x.shape[:-1], -1).to(x)
        x = torch.cat([px, x], dim=-1)
    if prepend_y.numel() > 0:
        py = prepend_y.expand(*y.shape[:-1], -1).to(y)
        y = torch.cat([py, y], dim=-1)
    if append_x.numel() > 0:
        ax = append_x.expand(*x.shape[:-1], -1).to(x)
        x = torch.cat([x, ax], dim=-1)
    if append_y.numel() > 0:
        ay = append_y.expand(*y.shape[:-1], -1).to(y)
        y = torch.cat([y, ay], dim=-1)

    if normalize_x:
        x_max = x.max(dim=-1, keepdim=True).values
        x = x / torch.clamp(x_max, min=1e-8)

    y = y / normalize_y

    # Flatten batch*q dimensions for vmap
    orig_shape = x.shape
    if x.dim() > 1:
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])
    else:
        x_flat = x.unsqueeze(0)
        y_flat = y.unsqueeze(0)

    # Sort by x-values so interp1d gets monotonically increasing x
    sort_idx = x_flat.argsort(dim=-1)
    x_flat = x_flat.gather(-1, sort_idx)
    y_flat = y_flat.gather(-1, sort_idx)

    new_x_expanded = new_x.expand(x_flat.shape[0], -1)
    result = torch.vmap(interp1d)(x_flat, y_flat, new_x_expanded)

    # Reshape back and add q=1 dimension for AppendFeatures convention
    if len(orig_shape) > 1:
        result = result.reshape(*orig_shape[:-1], -1)
    else:
        result = result.squeeze(0)

    return result.unsqueeze(-2)


def map_interpolate_feature(
    inputs: Inputs,
    transform_specs: InputTransformSpecs,
    feature: InterpolateFeature,
) -> AppendFeatures:
    features2idx, _ = inputs._get_transform_info(transform_specs)
    idx_x = [features2idx[key][0] for key in feature.x_keys]
    idx_y = [features2idx[key][0] for key in feature.y_keys]

    lower, upper = feature.interpolation_range
    new_x = torch.linspace(
        lower, upper, feature.n_interpolation_points, dtype=torch.double
    )

    return AppendFeatures(
        f=_interpolate_features,
        fkwargs={
            "idx_x": torch.tensor(idx_x, dtype=torch.long),
            "idx_y": torch.tensor(idx_y, dtype=torch.long),
            "new_x": new_x,
            "prepend_x": torch.tensor(feature.prepend_x, dtype=torch.double),
            "prepend_y": torch.tensor(feature.prepend_y, dtype=torch.double),
            "append_x": torch.tensor(feature.append_x, dtype=torch.double),
            "append_y": torch.tensor(feature.append_y, dtype=torch.double),
            "normalize_y": feature.normalize_y,
            "normalize_x": feature.normalize_x,
        },
        transform_on_train=True,
    )


def map_clone_feature(
    inputs: Inputs,
    transform_specs: InputTransformSpecs,
    feature: CloneFeature,
) -> AppendFeatures:
    features2idx, _ = inputs._get_transform_info(transform_specs)
    indices = [features2idx[key][0] for key in feature.features]

    def clone_features(X: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return X[..., indices].unsqueeze(-2)

    return AppendFeatures(
        f=clone_features,
        fkwargs={"indices": indices},
        transform_on_train=True,
    )


map_sum_feature = partial(_map_reduction_feature, reducer=torch.sum)
map_product_feature = partial(_map_reduction_feature, reducer=torch.prod)
map_mean_feature = partial(_map_reduction_feature, reducer=torch.mean)

AGGREGATE_MAP: Dict[Type[EngineeredFeature], Callable] = {
    SumFeature: map_sum_feature,
    ProductFeature: map_product_feature,
    MeanFeature: map_mean_feature,
    WeightedSumFeature: map_weighted_feature,
    WeightedMeanFeature: map_weighted_feature,
    MolecularWeightedSumFeature: map_weighted_feature,
    MolecularWeightedMeanFeature: map_weighted_feature,
    InterpolateFeature: map_interpolate_feature,
    CloneFeature: map_clone_feature,
}


def map(
    data_model: EngineeredFeature, inputs: Inputs, transform_specs: InputTransformSpecs
) -> AppendFeatures:
    return AGGREGATE_MAP[type(data_model)](
        inputs=inputs,
        transform_specs=transform_specs,
        feature=data_model,
    )
