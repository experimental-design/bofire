from typing import List, Union

import pandas as pd
import torch
from botorch.models.transforms.input import InputStandardize, Normalize

from bofire.data_models.domain.api import Inputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MordredDescriptors,
)
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.types import InputTransformSpecs
from bofire.utils.torch_tools import tkwargs


def get_molecular_feature_keys(
    specs: InputTransformSpecs,
) -> List[str]:
    """Returns a list of molecular feature keys in the input data.
    These features are features with transform type Fingerprints, Fragments,
    or FingerprintsFragments in `specs`.

    Args:
        specs (InputTransformSpecs): Dictionary specifying which
            input feature is transformed by which encoder.

    Returns:
        List[str]: The list of molecular feature keys.

    """
    molecular_feature_key_list = [
        key
        for key, value in specs.items()
        if isinstance(value, Fingerprints)
        or isinstance(value, Fragments)
        or isinstance(value, FingerprintsFragments)
    ]
    return sorted(molecular_feature_key_list)


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
    ]
    continuous_feature_key_list = [
        feat.key
        for feat in inputs.get()
        if feat.key not in non_continuous_feature_key_list
    ]
    return sorted(continuous_feature_key_list)


def get_categorical_feature_keys(
    specs: InputTransformSpecs,
) -> List[str]:
    """Returns a list of categorical feature keys in the input data.
    These features are not descriptor-based and are not of type Fingerprints, Fragments, FingerprintsFragments,
    or MordredDescriptors.

    Args:
        specs (InputTransformSpecs): Dictionary specifying which
            input feature is transformed by which encoder.

    Returns:
        List[str]: The list of categorical feature keys.

    """
    categorical_feature_key_list = [
        key
        for key, value in specs.items()
        if value != CategoricalEncodingEnum.DESCRIPTOR
        and not isinstance(value, Fingerprints)
        and not isinstance(value, Fragments)
        and not isinstance(value, FingerprintsFragments)
        and not isinstance(value, MordredDescriptors)
    ]
    return sorted(categorical_feature_key_list)


def get_scaler(
    inputs: Inputs,
    input_preprocessing_specs: InputTransformSpecs,
    scaler: ScalerEnum,
    X: pd.DataFrame,
) -> Union[InputStandardize, Normalize, None]:
    """Returns the instanitated scaler object for a set of input features and
    input_preprocessing_specs.


    Args:
        inputs (Inputs): Input features.
        input_preprocessing_specs (InputTransformSpecs): Dictionary how to treat
            the categoricals and/or molecules.
        scaler (ScalerEnum): Enum indicating the scaler of interest.
        X (pd.DataFrame): The dataset of interest.

    Returns:
        Union[InputStandardize, Normalize]: The instantiated scaler class

    """
    if scaler != ScalerEnum.IDENTITY:
        features2idx, _ = inputs._get_transform_info(input_preprocessing_specs)

        d = 0
        for indices in features2idx.values():
            d += len(indices)

        continuous_feature_keys = get_continuous_feature_keys(
            inputs=inputs,
            specs=input_preprocessing_specs,
        )

        ord_dims = inputs.get_feature_indices(
            specs=input_preprocessing_specs,
            feature_keys=continuous_feature_keys,
        )

        if len(ord_dims) == 0:
            return None

        if scaler == ScalerEnum.NORMALIZE:
            lower, upper = inputs.get_bounds(
                specs=input_preprocessing_specs,
                experiments=X,
            )
            scaler_transform = Normalize(
                d=d,
                bounds=torch.tensor([lower, upper]).to(**tkwargs),
                indices=ord_dims,
                batch_shape=torch.Size(),
            )
        elif scaler == ScalerEnum.STANDARDIZE:
            scaler_transform = InputStandardize(
                d=d,
                indices=ord_dims,
                batch_shape=torch.Size(),
            )
        else:
            raise ValueError("Scaler enum not known.")
        return scaler_transform
    return None
