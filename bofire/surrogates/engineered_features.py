import pandas as pd
import torch
from botorch.models.transforms.input import AppendFeatures

from bofire.data_models.api import Inputs
from bofire.data_models.features.api import (
    EngineeredFeature,
    MeanFeature,
    MolecularWeightedSumFeature,
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


def map_sum_feature(
    inputs: Inputs, transform_specs: InputTransformSpecs, feature: SumFeature
) -> AppendFeatures:
    # Get indices of features to be summed
    features2idx, _ = inputs._get_transform_info(transform_specs)
    indices = [features2idx[key][0] for key in feature.features]

    def sum_features(X: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        result = torch.sum(X[..., indices], dim=-1, keepdim=True).unsqueeze(-2)
        return result.expand(*result.shape[:-2], 1, -1)

    return AppendFeatures(
        f=sum_features,  # type: ignore
        fkwargs={"indices": indices},
        transform_on_train=True,
    )


def map_mean_feature(
    inputs: Inputs, transform_specs: InputTransformSpecs, feature: MeanFeature
) -> AppendFeatures:
    # Get indices of features to be summed
    features2idx, _ = inputs._get_transform_info(transform_specs)
    indices = [features2idx[key][0] for key in feature.features]

    def average_features(X: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        result = torch.mean(X[..., indices], dim=-1, keepdim=True).unsqueeze(-2)
        return result.expand(*result.shape[:-2], 1, -1)

    return AppendFeatures(
        f=average_features,  # type: ignore
        fkwargs={"indices": indices},
        transform_on_train=True,
    )


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
            inputs.get_by_key(key).to_df()[feature.descriptors].values[0]  # type: ignore
            for key in feature.features
        ],  # type: ignore
        dtype=torch.double,
    )

    # we need to get the descriptors into one tensor
    return AppendFeatures(
        f=_weighted_sum_features,  # type: ignore
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

    molecules = [inputs.get_by_key(key).molecule for key in feature.features]  # type: ignore
    # filter out the highly-correlated descriptors
    feature.molfeatures._update_uncorrelated_descriptors(molecules)
    descriptors_df = feature.molfeatures.get_descriptor_values(pd.Series(molecules))
    descriptors = torch.tensor(descriptors_df.values, dtype=torch.double)

    return AppendFeatures(
        f=_weighted_sum_features,  # type: ignore
        fkwargs={"indices": indices, "descriptors": descriptors},
        transform_on_train=True,
    )


AGGREGATE_MAP = {
    SumFeature: map_sum_feature,
    MeanFeature: map_mean_feature,
    WeightedSumFeature: map_weighted_sum_feature,
    MolecularWeightedSumFeature: map_molecular_weighted_sum_feature,
}


def map(
    data_model: EngineeredFeature, inputs: Inputs, transform_specs: InputTransformSpecs
) -> AppendFeatures:
    return AGGREGATE_MAP[type(data_model)](  # type: ignore
        inputs=inputs,
        transform_specs=transform_specs,
        feature=data_model,
    )
