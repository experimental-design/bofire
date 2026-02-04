from typing import List, Union

import torch
from botorch.models.transforms.input import (
    ChainedInputTransform,
    FilterFeatures,
    InputStandardize,
    InputTransform,
    Normalize,
)

from bofire.data_models.domain.api import EngineeredFeatures, Inputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.molfeatures.api import CompositeMolFeatures, MordredDescriptors
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.engineered_features import map as map_feature
from bofire.utils.torch_tools import get_NumericToCategorical_input_transform


def get_continuous_feature_keys(
    inputs: Inputs,
    specs: InputTransformSpecs,
) -> List[str]:
    """Returns a list of continuous feature keys in the input data.
    These features include continuous inputs, categorical inputs with transform
    type CategoricalEncodingEnum.DESCRIPTOR, and Mordred descriptors.

    Args:
        inputs (Inputs): Input features.
        specs (InputTransformSpecs): Dictionary specifying which
            input feature is transformed by which encoder.

    Returns:
        List[str]: The list of continuous feature keys.

    """
    non_continuous_feature_key_list = [
        key
        for key, value in specs.items()
        if value != CategoricalEncodingEnum.DESCRIPTOR
        and not isinstance(value, MordredDescriptors)
        and not (
            isinstance(value, CompositeMolFeatures)
            and any(
                isinstance(molfeature, MordredDescriptors)
                for molfeature in value.features
            )
        )
    ]
    continuous_feature_key_list = [
        feat.key
        for feat in inputs.get()
        if feat.key not in non_continuous_feature_key_list
    ]
    return sorted(continuous_feature_key_list)


def get_scaler(
    inputs: Inputs,
    engineered_features: EngineeredFeatures,
    categorical_encodings: InputTransformSpecs,
    scaler_type: ScalerEnum,
) -> Union[InputStandardize, Normalize, None]:
    """Returns the instanitated scaler object for a set of input features and
    categorical_encodings.


    Args:
        inputs: Input features.
        engineered_features: Engineered features.
        categorical_encodings: Dictionary how to treat
            the categoricals and/or molecules.
        scaler_type (ScalerEnum): Enum indicating the scaler of interest.

    Returns:
        The instantiated botorch scaler object or None if no scaling is to be
            applied.

    """
    if scaler_type == ScalerEnum.IDENTITY:
        return None
    features2idx, _ = inputs._get_transform_info(categorical_encodings)

    d = 0
    for indices in features2idx.values():
        d += len(indices)

    # now we get the engineered feature information
    offset = d
    efeatures2idx = engineered_features.get_features2idx(offset=offset)
    for indices in efeatures2idx.values():
        d += len(indices)

    continuous_feature_keys = get_continuous_feature_keys(
        inputs=inputs,
        specs=categorical_encodings,
    )

    ord_dims = inputs.get_feature_indices(
        specs=categorical_encodings,
        feature_keys=continuous_feature_keys,
    ) + engineered_features.get_feature_indices(
        offset=offset, feature_keys=engineered_features.get_keys()
    )

    if len(ord_dims) == 0:
        return None

    if scaler_type == ScalerEnum.NORMALIZE:
        return Normalize(
            d=d,
            # bounds=torch.tensor([lower, upper]).to(**tkwargs),
            indices=ord_dims,
            batch_shape=torch.Size(),
        )
    # it has to be standardize
    return InputStandardize(
        d=d,
        indices=ord_dims,
        batch_shape=torch.Size(),
    )


def get_input_transform(
    inputs: Inputs,
    engineered_features: EngineeredFeatures,
    scaler_type: ScalerEnum,
    categorical_encodings: InputTransformSpecs,
) -> Union[InputTransform, None]:
    """Creates the botorch input transform on the basis of
    the specified inputs, engineered features and categorical
    encodings.

    Args:
        inputs: Input features.
        engineered_features: Engineered features.
        scaler_type: The scaler enum to be used.
        categorical_encodings: Dictionary how to treat
            the categoricals and/or molecules.

    Returns:
        The created input transform or None.
    """
    transforms = {}
    ignored = []

    # first categorical encodings
    categorical_transform = get_NumericToCategorical_input_transform(
        inputs, categorical_encodings
    )
    if categorical_transform is not None:
        transforms["cat"] = categorical_transform

    # second engineered features
    for feature in engineered_features.get():
        transforms[feature.key] = map_feature(
            data_model=feature, inputs=inputs, transform_specs=categorical_encodings
        )
        if not feature.keep_features:
            ignored.extend(feature.features)

    # third scaler
    scaler = get_scaler(
        inputs=inputs,
        engineered_features=engineered_features,
        categorical_encodings=categorical_encodings,
        scaler_type=scaler_type,
    )
    if scaler is not None:
        transforms["scaler"] = scaler

    # fourth remove ignored features
    if len(ignored) > 0:
        ignored_idx = inputs.get_feature_indices(
            specs=categorical_encodings,
            feature_keys=list(set(ignored)),
        )
        original_idx = inputs.get_feature_indices(
            specs=categorical_encodings, feature_keys=inputs.get_keys()
        )
        engineered_idx = engineered_features.get_feature_indices(
            offset=len(original_idx),
            feature_keys=engineered_features.get_keys(),
        )
        all_idx = original_idx + engineered_idx
        keep_idx = list(set(all_idx) - set(ignored_idx))
        transforms["filter_engineered"] = FilterFeatures(
            feature_indices=torch.tensor(keep_idx, dtype=torch.int64)
        )

    # now chain them
    if len(transforms) == 0:
        return None
    if len(transforms) == 1:
        return list(transforms.values())[0]
    return ChainedInputTransform(**transforms)
