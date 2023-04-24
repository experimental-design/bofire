from pydantic import validator

from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    NumericalInput,
)
from bofire.data_models.surrogates.surrogate import Surrogate


class BotorchSurrogate(Surrogate):
    @validator("input_preprocessing_specs", always=True)
    def validate_input_preprocessing_specs(cls, v, values):
        inputs = values["inputs"]
        categorical_keys = inputs.get_keys(CategoricalInput, exact=True)
        descriptor_keys = inputs.get_keys(CategoricalDescriptorInput, exact=True)
        for key in categorical_keys:
            if (
                v.get(key, CategoricalEncodingEnum.ONE_HOT)
                != CategoricalEncodingEnum.ONE_HOT
            ):
                raise ValueError(
                    "Botorch based models have to use one hot encodings for categoricals"
                )
            else:
                v[key] = CategoricalEncodingEnum.ONE_HOT
        # TODO: include descriptors into probabilistic reparam via OneHotToDescriptor input transform
        for key in descriptor_keys:
            if v.get(key, CategoricalEncodingEnum.DESCRIPTOR) not in [
                CategoricalEncodingEnum.DESCRIPTOR,
                CategoricalEncodingEnum.ONE_HOT,
            ]:
                raise ValueError(
                    "Botorch based models have to use one hot encodings or descriptor encodings for categoricals."
                )
            elif v.get(key) is None:
                v[key] = CategoricalEncodingEnum.DESCRIPTOR
        for key in inputs.get_keys(NumericalInput):
            if v.get(key) is not None:
                raise ValueError(
                    "Botorch based models have to use internal transforms to preprocess numerical features."
                )
        return v
