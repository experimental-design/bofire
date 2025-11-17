from typing import Type

from pydantic import Field, field_validator

from bofire.data_models.domain.features import Inputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalMolecularInput,
    NumericalInput,
    TaskInput,
)
from bofire.data_models.molfeatures.api import Fingerprints
from bofire.data_models.surrogates.surrogate import Surrogate
from bofire.data_models.types import InputTransformSpecs


class BotorchSurrogate(Surrogate):
    """Base class for all botorch based surrogates, that can be used in botorch
    based strategies.

    Attributes:
        input_preprocessing_specs: A dictionary specifying how categorical features are to be
            preprocessed **before** being passed to the surrogate. For all botorch based surrogates, an
            ordinal encoding (`CategoricalEncodingEnum.ORDINAL`) has to be used for all
            categorical features, which is also set as default if nothing is provided.
        categorical_encodings: A dictionary specifying how
            categorical features are to be encoded **within** the botorch based surrogate.
            Keys are the feature keys and values are the encoding types. If no surrogate specific
            default is defined, by default categorical features are one-hot encoded, categorical
            descriptor features are descriptor encoded and categorical molecular features
            are fingerprint encoded. If a feature is not specified in the dictionary, the default
            encoding for the feature type is used.
    """

    categorical_encodings: InputTransformSpecs = Field(
        default_factory=dict, validate_default=True
    )

    @field_validator("input_preprocessing_specs")
    @classmethod
    def validate_input_preprocessing_specs(cls, v, info):
        # when validator for inputs fails, this validator is still checked and causes an Exception error instead of a ValueError
        # fix this by checking if inputs is in info.data
        if "inputs" not in info.data:
            return None

        inputs = info.data["inputs"]
        categorical_keys = inputs.get_keys(CategoricalInput, exact=False)
        for key in categorical_keys:
            if (
                v.get(key, CategoricalEncodingEnum.ORDINAL)
                != CategoricalEncodingEnum.ORDINAL
            ):
                raise ValueError(
                    "Botorch based models have to use ordinal encodings for categoricals",
                )
            v[key] = CategoricalEncodingEnum.ORDINAL
        for key in inputs.get_keys(NumericalInput):
            if v.get(key) is not None:
                raise ValueError(
                    "Botorch based models have to use internal transforms to preprocess numerical features.",
                )
        return v

    @classmethod
    def _default_categorical_encodings(
        cls,
    ) -> dict[Type[CategoricalInput], CategoricalEncodingEnum | Fingerprints]:
        return {
            CategoricalInput: CategoricalEncodingEnum.ONE_HOT,
            CategoricalMolecularInput: Fingerprints(),
            CategoricalDescriptorInput: CategoricalEncodingEnum.DESCRIPTOR,
            TaskInput: CategoricalEncodingEnum.ONE_HOT,
        }

    @classmethod
    def _generate_default_categorical_encodings(
        cls, inputs: Inputs, categorical_encodings: InputTransformSpecs
    ) -> InputTransformSpecs:
        default_encodings = cls._default_categorical_encodings()
        categorical_keys = inputs.get_keys(CategoricalInput, exact=False)
        for key in categorical_keys:
            if key not in categorical_encodings:
                feat = inputs.get_by_key(key)
                categorical_encodings[key] = default_encodings[type(feat)]  # type: ignore
        return categorical_encodings

    @field_validator("categorical_encodings")
    @classmethod
    def validate_categorical_encodings(cls, v, info):
        # when validator for inputs fails, this validator is still checked and causes an Exception error instead of a ValueError
        # fix this by checking if inputs is in info.data
        if "inputs" not in info.data:
            return None

        inputs: Inputs = info.data["inputs"]
        v = cls._generate_default_categorical_encodings(inputs, v)
        inputs._validate_transform_specs(v)
        return v
