from functools import partial
from typing import Callable

import pandas as pd
import torch
from botorch.models.transforms.input import AppendFeatures

from bofire.data_models.api import Inputs
from bofire.data_models.features.api import (
    EngineeredFeature,
    InterpolateFeature,
    MeanFeature,
    MolecularWeightedSumFeature,
    ProductFeature,
    SumFeature,
    WeightedSumFeature,
)
from bofire.data_models.types import InputTransformSpecs
from bofire.utils.torch_tools import interp1d


def _weighted_sum_features(
    X: torch.Tensor,
    indices: torch.Tensor,
    descriptors: torch.Tensor,
) -> torch.Tensor:
    result = torch.matmul(
        X[..., indices],
        descriptors,
    ).unsqueeze(-2)
    return result.expand(*result.shape[:-2], 1, -1)


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
        result = reducer(X[..., indices], dim=-1, keepdim=True).unsqueeze(-2)
        return result.expand(*result.shape[:-2], 1, -1)

    return AppendFeatures(
        f=reduce_features,
        fkwargs={"indices": indices, "reducer": reducer},
        transform_on_train=True,
    )


map_sum_feature = partial(_map_reduction_feature, reducer=torch.sum)
map_product_feature = partial(_map_reduction_feature, reducer=torch.prod)
map_mean_feature = partial(_map_reduction_feature, reducer=torch.mean)


def map_weighted_sum_feature(
    inputs: Inputs,
    transform_specs: InputTransformSpecs,
    feature: WeightedSumFeature,
) -> AppendFeatures:
    # use get_feature_indices

    features2idx, _ = inputs._get_transform_info(transform_specs)
    indices = [features2idx[key][0] for key in feature.features]

    descriptors = torch.tensor(
        [
            inputs.get_by_key(key)
            .to_df()[feature.descriptors]  # ty: ignore[unresolved-attribute]
            .values[0]
            for key in feature.features
        ],
        dtype=torch.double,
    )

    # we need to get the descriptors into one tensor
    return AppendFeatures(
        f=_weighted_sum_features,
        fkwargs={"indices": indices, "descriptors": descriptors},
        transform_on_train=True,
    )


def map_molecular_weighted_sum_feature(
    inputs: Inputs,
    transform_specs: InputTransformSpecs,
    feature: MolecularWeightedSumFeature,
) -> AppendFeatures:
    features2idx, _ = inputs._get_transform_info(transform_specs)
    indices = [features2idx[key][0] for key in feature.features]

    molecules = [
        inputs.get_by_key(key).molecule  # ty: ignore[unresolved-attribute]
        for key in feature.features
    ]
    # filter out the highly-correlated descriptors
    feature.molfeatures.remove_correlated_descriptors(molecules)
    descriptors_df = feature.molfeatures.get_descriptor_values(pd.Series(molecules))
    descriptors = torch.tensor(descriptors_df.values, dtype=torch.double)

    return AppendFeatures(
        f=_weighted_sum_features,
        fkwargs={"indices": indices, "descriptors": descriptors},
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


AGGREGATE_MAP = {
    SumFeature: map_sum_feature,
    ProductFeature: map_product_feature,
    MeanFeature: map_mean_feature,
    WeightedSumFeature: map_weighted_sum_feature,
    MolecularWeightedSumFeature: map_molecular_weighted_sum_feature,
    InterpolateFeature: map_interpolate_feature,
}


def map(
    data_model: EngineeredFeature, inputs: Inputs, transform_specs: InputTransformSpecs
) -> AppendFeatures:
    return AGGREGATE_MAP[type(data_model)](
        inputs=inputs,
        transform_specs=transform_specs,
        feature=data_model,
    )
