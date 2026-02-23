from typing import List, Mapping, Union

import pandas as pd
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
from bofire.data_models.surrogates.scaler import AnyScaler
from bofire.data_models.surrogates.scaler import Normalize as NormalizeScaler
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.engineered_features import map as map_feature
from bofire.utils.torch_tools import get_NumericToCategorical_input_transform, tkwargs


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
    scaler_type: AnyScaler,
    X: pd.DataFrame | None = None,
) -> Mapping[str, Normalize | InputStandardize]:
    """Returns the instanitated scaler object for a set of input features and
    categorical_encodings.


    Args:
        inputs: Input features.
        engineered_features: Engineered features.
        categorical_encodings: Dictionary how to treat
            the categoricals and/or molecules.
        scaler_type (AnyScaler): The scaler object indicating the scaler of interest.
        X: Experimental values of input features

    Returns:
        Dictionary of the instantiated botorch scaler object(s), can be empty.

    """
    if scaler_type is None:
        return {}
    features2idx, _ = inputs._get_transform_info(categorical_encodings)

    d = 0
    for indices in features2idx.values():
        d += len(indices)

    # now we get the engineered feature information
    offset = d
    efeatures2idx = engineered_features.get_features2idx(offset=offset)
    for indices in efeatures2idx.values():
        d += len(indices)

    if len(scaler_type.features) == 0:
        # if no features are specified, we scale all features that
        # behave like continuous features, i.e. all features that are
        # not encoded as categoricals
        continuous_feature_keys = get_continuous_feature_keys(
            inputs=inputs,
            specs=categorical_encodings,
        )

        cont_feat_dims = inputs.get_feature_indices(
            specs=categorical_encodings,
            feature_keys=continuous_feature_keys,
        )
        engineered_feat_dims = engineered_features.get_feature_indices(
            offset=offset, feature_keys=engineered_features.get_keys()
        )

    else:
        # if features are specified, we only scale those
        # for this we have to find its indices in the transformed space
        # and for this we have to first find out which features are original
        # and which are engineered
        cont_feat_dims = inputs.get_feature_indices(
            specs=categorical_encodings,
            feature_keys=[
                feat for feat in scaler_type.features if feat in inputs.get_keys()
            ],
        )
        engineered_feat_dims = engineered_features.get_feature_indices(
            offset=offset,
            feature_keys=[
                feat
                for feat in scaler_type.features
                if feat in engineered_features.get_keys()
            ],
        )

    ord_dims = cont_feat_dims + engineered_feat_dims

    if len(ord_dims) == 0:
        return {}

    if isinstance(scaler_type, NormalizeScaler):
        # We create a separate Normalize for non-engineered features,
        # since bounds are known for these features.
        lower, upper = inputs.get_bounds(
            specs=categorical_encodings,
            experiments=X,
        )
        input_tfs: dict[str, Normalize] = {}
        if cont_feat_dims:
            input_tfs["scaler"] = Normalize(
                d=d,
                bounds=torch.tensor([lower, upper]).to(**tkwargs)[:, cont_feat_dims],
                indices=cont_feat_dims,
                batch_shape=torch.Size(),
            )

        if engineered_feat_dims:
            input_tfs["engineered_scaler"] = Normalize(
                d=d,
                indices=engineered_feat_dims,
                batch_shape=torch.Size(),
            )

        return input_tfs

    # it has to be standardize
    return {
        "scaler": InputStandardize(
            d=d,
            indices=ord_dims,
            batch_shape=torch.Size(),
        )
    }


def get_input_transform(
    inputs: Inputs,
    engineered_features: EngineeredFeatures,
    scaler_type: AnyScaler,
    categorical_encodings: InputTransformSpecs,
    X: pd.DataFrame | None = None,
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
        X: Experimental values of input features

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
        X=X,
    )

    transforms.update(scaler)

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
