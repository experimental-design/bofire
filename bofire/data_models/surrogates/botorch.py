from pydantic import Field, field_validator, model_validator

from bofire.data_models.descriptors.api import GeneratedSource, StaticSource
from bofire.data_models.domain.api import EngineeredFeatures
from bofire.data_models.domain.features import Inputs
from bofire.data_models.encodings._migrate import migrate_legacy_encodings
from bofire.data_models.encodings.api import (
    DescriptorEncoding,
    OneHotEncoding,
    OrdinalEncoding,
)
from bofire.data_models.features.api import (
    CategoricalInput,
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
            Keys are the feature keys and values are the encoding types. If a feature is
            not specified, a default is chosen from the descriptor *data* the feature
            carries: a feature with a structure column (e.g. ``smiles``) is fingerprint
            encoded, one with numeric descriptor columns is descriptor encoded, and a
            plain categorical falls back to the surrogate-specific default (one-hot here).
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
    def _default_plain_categorical_encodings(cls) -> dict:
        """Fallback encodings for categoricals *without* descriptor data, keyed by type.

        Descriptor-carrying features are resolved from their data (see
        :meth:`_resolve_default_categorical_encoding`); this map only covers the
        non-descriptor case, where surrogates differ (one-hot vs ordinal) and task
        inputs may want their own default.
        """
        return {
            CategoricalInput: OneHotEncoding(),
            CategoricalTaskInput: OneHotEncoding(),
        }

    @classmethod
    def _resolve_default_categorical_encoding(cls, feat: CategoricalInput):
        """Pick the default encoding for ``feat`` from the descriptor data it carries.

        Task inputs never descriptor-encode. Otherwise a structure column implies a
        molecular (fingerprint) generator, numeric descriptor columns imply a static
        source, and a plain categorical uses the surrogate's non-descriptor fallback.
        """
        fallbacks = cls._default_plain_categorical_encodings()
        if not isinstance(feat, CategoricalTaskInput):
            if feat.descriptor_columns(role="structure"):
                return DescriptorEncoding(
                    source=GeneratedSource(generator=Fingerprints())
                )
            if feat.descriptor_columns(role="descriptor"):
                return DescriptorEncoding(source=StaticSource())
        for klass in type(feat).__mro__:
            if klass in fallbacks:
                return fallbacks[klass]
        return OneHotEncoding()

    @classmethod
    def _generate_default_categorical_encodings(
        cls, inputs: Inputs, categorical_encodings: InputTransformSpecs
    ) -> InputTransformSpecs:
        categorical_keys = inputs.get_keys(CategoricalInput, exact=False)
        for key in categorical_keys:
            if key not in categorical_encodings:
                default = cls._resolve_default_categorical_encoding(
                    inputs.get_by_key(key)
                )
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
