from pydantic import Field, field_validator, model_validator

from bofire.data_models.descriptor_generators.api import Fingerprints
from bofire.data_models.domain.api import EngineeredFeatures
from bofire.data_models.domain.features import Inputs
from bofire.data_models.encodings.api import DescriptorEncoding, OneHotEncoding
from bofire.data_models.features.api import CategoricalInput, CategoricalTaskInput
from bofire.data_models.surrogates.surrogate import Surrogate
from bofire.data_models.types import InputTransformSpecs


class BotorchSurrogate(Surrogate):
    """Surrogate built on BoTorch, usable in the Bayesian optimization strategies.

    Categorical inputs are passed through their encoding before they reach the
    model. Engineered features are computed from the inputs inside the model, and
    are not set directly by the optimizer.
    """

    categorical_encodings: InputTransformSpecs = Field(
        default={},
        validate_default=True,
        description="How each categorical input is represented to the model, by input "
        "key. An input not listed gets a default from the descriptor data it carries: "
        "a structure column (e.g. SMILES) gives fingerprints, numeric descriptor "
        "columns give a descriptor encoding, and a plain categorical gets the "
        "surrogate's default.",
    )
    engineered_features: EngineeredFeatures = Field(
        default=EngineeredFeatures(),
        description="Quantities derived from the inputs, such as sums or "
        "amount-weighted descriptor blends, that the model sees as further inputs.",
    )

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

        A structure implies a molecular (fingerprint) generator alongside any numeric
        columns, numeric columns alone imply a static source, and a feature with no
        descriptor block uses the surrogate's non-descriptor fallback.
        """
        # `descriptors` is None by type on task inputs, so they fall through to the
        # non-descriptor fallback without needing a special case here.
        if feat.descriptors is not None:
            if feat.descriptors.structure is not None:
                # fingerprint from the structure, *plus* any numeric columns the feature
                # carries (columns=None means "all of them") — a feature with both must
                # not silently lose the handcrafted half.
                return DescriptorEncoding(generators=[Fingerprints()])
            return DescriptorEncoding()  # all numeric descriptor columns
        fallbacks = cls._default_plain_categorical_encodings()
        kind = (
            CategoricalTaskInput
            if isinstance(feat, CategoricalTaskInput)
            else CategoricalInput
        )
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
