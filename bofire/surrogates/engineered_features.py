from functools import partial
from typing import Callable

import pandas as pd
import torch
from botorch.models.transforms.input import AppendFeatures

from bofire.data_models.api import Inputs
from bofire.data_models.features.api import (
    CloneFeature,
    EngineeredFeature,
    MeanFeature,
    MolecularWeightedSumFeature,
    ProductFeature,
    SumFeature,
    WeightedSumFeature,
)
from bofire.data_models.types import InputTransformSpecs


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


def map_clone_feature(
    inputs: Inputs,
    transform_specs: InputTransformSpecs,
    feature: CloneFeature,
) -> AppendFeatures:
    features2idx, _ = inputs._get_transform_info(transform_specs)
    indices = [features2idx[key][0] for key in feature.features]

    def clone_features(X: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        result = X[..., indices].unsqueeze(-2)
        return result.expand(*result.shape[:-2], 1, -1)

    return AppendFeatures(
        f=clone_features,
        fkwargs={"indices": indices},
        transform_on_train=True,
    )


AGGREGATE_MAP = {
    SumFeature: map_sum_feature,
    ProductFeature: map_product_feature,
    MeanFeature: map_mean_feature,
    WeightedSumFeature: map_weighted_sum_feature,
    MolecularWeightedSumFeature: map_molecular_weighted_sum_feature,
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
