from pydantic import validator

from bofire.data_models.enum import CategoricalEncodingEnum, MolecularEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    NumericalInput,
    MolecularInput,
)
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    Fragments,
    FingerprintsFragments,
    MordredDescriptors,
)  # BagOfCharacters
from bofire.data_models.surrogates.surrogate import Surrogate


class BotorchSurrogate(Surrogate):
    @validator("input_preprocessing_specs", always=True)
    def validate_input_preprocessing_specs(cls, v, values):
        inputs = values["inputs"]
        categorical_keys = inputs.get_keys(CategoricalInput, exact=True)
        descriptor_keys = inputs.get_keys(CategoricalDescriptorInput, exact=True)
        molecular_keys = inputs.get_keys(MolecularInput, exact=True)
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
        # TODO: include descriptors into probabilistic reparam via OneHotToDescriptor input transform
        for key in molecular_keys:
            # if v.get(key, MolecularEncodingEnum.FINGERPRINTS) not in [
            mol_encoding = v.get(key)
            if mol_encoding not in [
                MolecularEncodingEnum.FINGERPRINTS,
                MolecularEncodingEnum.FRAGMENTS,
                MolecularEncodingEnum.FINGERPRINTS_FRAGMENTS,
                MolecularEncodingEnum.MOL_DESCRIPTOR,
            ]:
                raise ValueError(
                    "Botorch based models have to use fingerprints, fragments, fingerprints_fragments, or molecular descriptors for molecular inputs"
                )
            for feat in inputs:
                if key == feat.key:
                    if (
                        (
                            mol_encoding == MolecularEncodingEnum.FINGERPRINTS
                            and not isinstance(feat.molfeatures, Fingerprints)
                        )
                        or (
                            mol_encoding == MolecularEncodingEnum.FRAGMENTS
                            and not isinstance(feat.molfeatures, Fragments)
                        )
                        or (
                            mol_encoding == MolecularEncodingEnum.FINGERPRINTS_FRAGMENTS
                            and not isinstance(feat.molfeatures, FingerprintsFragments)
                        )
                        or (
                            mol_encoding == MolecularEncodingEnum.MOL_DESCRIPTOR
                            and not isinstance(feat.molfeatures, MordredDescriptors)
                        )
                    ):
                        raise ValueError(
                            f"The molecular feature type selected ({feat.molfeatures.type}) for '{feat.key}' is not the same as the molecular encoding type chosen ({mol_encoding})"
                        )

            # elif mol_encoding is None:
            #     v[key] = MolecularEncodingEnum.FINGERPRINTS
        return v
