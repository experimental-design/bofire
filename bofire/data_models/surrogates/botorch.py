from pydantic import Field, field_validator, model_validator

from bofire.data_models.descriptor_generators.api import Fingerprints
from bofire.data_models.domain.api import EngineeredFeatures
from bofire.data_models.domain.features import Inputs
from bofire.data_models.encodings._migrate import migrate_legacy_encodings
from bofire.data_models.encodings.api import DescriptorEncoding, OneHotEncoding
from bofire.data_models.features.api import CategoricalInput, CategoricalTaskInput
from bofire.data_models.surrogates.surrogate import Surrogate
from bofire.data_models.types import InputTransformSpecs


class BotorchSurrogate(Surrogate):
    """Base class for all botorch based surrogates, that can be used in botorch
    based strategies.

    Attributes:
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

        A ``structure`` column implies a molecular (fingerprint) generator, numeric
        descriptor columns imply a static source, and a plain categorical uses the
        surrogate's non-descriptor fallback.

        Task inputs never descriptor-encode: they are indices, not described entities,
        and ``TaskInput.validate_no_descriptor_data`` already guarantees they carry no
        descriptor data. The check is kept explicit here so the intent is local.
        """
        is_task = isinstance(feat, CategoricalTaskInput)
        if not is_task:
            if feat.structure is not None:
                # fingerprint-encode from the structure column, static columns excluded
                return DescriptorEncoding(columns=[], generators=[Fingerprints()])
            if feat.descriptor_columns():
                return DescriptorEncoding()  # all numeric descriptor columns
        # look up the fallback by kind, not by exact type, so deprecated subclasses
        # (CategoricalDescriptorInput, CategoricalMolecularInput) resolve too.
        fallbacks = cls._default_plain_categorical_encodings()
        kind = CategoricalTaskInput if is_task else CategoricalInput
        return fallbacks.get(kind, OneHotEncoding())

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
