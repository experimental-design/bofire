from pydantic import Field, field_validator, model_validator

from bofire.data_models.domain.api import EngineeredFeatures
from bofire.data_models.domain.features import Inputs
from bofire.data_models.encodings._migrate import migrate_legacy_encodings
from bofire.data_models.encodings.api import (
    DescriptorEncoding,
    MolecularEncoding,
    OneHotEncoding,
    OrdinalEncoding,
)
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalMolecularInput,
    CategoricalTaskInput,
    NumericalInput,
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
            ordinal encoding (`OrdinalEncoding`) has to be used for all
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
    engineered_features: EngineeredFeatures = Field(
        default_factory=lambda: EngineeredFeatures()
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
            spec = v.get(key)
            if spec is not None and not isinstance(spec, OrdinalEncoding):
                raise ValueError(
                    "Botorch based models have to use ordinal encodings for categoricals",
                )
            v[key] = OrdinalEncoding()
        for key in inputs.get_keys(NumericalInput):
            if v.get(key) is not None:
                raise ValueError(
                    "Botorch based models have to use internal transforms to preprocess numerical features.",
                )
        return v

    @field_validator("categorical_encodings", mode="before")
    @classmethod
    def migrate_legacy_categorical_encodings(cls, v):
        return migrate_legacy_encodings(v)

    @classmethod
    def _default_categorical_encodings(
        cls,
    ) -> dict:
        return {
            CategoricalInput: OneHotEncoding(),
            CategoricalMolecularInput: MolecularEncoding(generator=Fingerprints()),
            CategoricalDescriptorInput: DescriptorEncoding(),
            CategoricalTaskInput: OneHotEncoding(),
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
                default = default_encodings[type(feat)]
                # deep-copy so per-feature encoders (and their stateful generators)
                # are not shared between features.
                categorical_encodings[key] = (
                    default.model_copy(deep=True)
                    if hasattr(default, "model_copy")
                    else default
                )
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

    @model_validator(mode="after")
    def validate_engineered_features(self):
        self.engineered_features.validate_inputs(self.inputs)
        return self
