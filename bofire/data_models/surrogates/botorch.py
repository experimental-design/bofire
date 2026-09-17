from pydantic import Field, field_validator, model_validator

from bofire.data_models.descriptor_generators.api import Fingerprints
from bofire.data_models.domain.api import EngineeredFeatures
from bofire.data_models.domain.features import Inputs
from bofire.data_models.encodings.api import DescriptorEncoding, OneHotEncoding
from bofire.data_models.features.api import CategoricalInput, CategoricalTaskInput
from bofire.data_models.surrogates.surrogate import Surrogate
from bofire.data_models.types import InputTransformSpecs


# reused by the surrogates that expose a kernel, each of which narrows its type and so
# has to redeclare the field, which drops any inherited description
KERNEL_DESCRIPTION = (
    "The module computing the covariance matrix, which encodes the similarity between "
    "inputs."
)


class BotorchSurrogate(Surrogate):
    """Surrogate built on BoTorch, and so usable by the BoTorch-based strategies.

    Everything the model sees is numeric, so this is the level at which categoricals
    are encoded and extra columns are derived from the inputs.
    """

    categorical_encodings: InputTransformSpecs = Field(
        default={},
        validate_default=True,
        description="How each categorical feature is turned into model columns, keyed "
        "by feature key. A feature left out is defaulted from the descriptor data it "
        "carries: a structure column gives fingerprints, numeric columns give "
        "descriptors, and a plain categorical falls back to what the surrogate "
        "requires.",
    )
    engineered_features: EngineeredFeatures = Field(
        default=EngineeredFeatures(),
        description="Quantities computed from the inputs and appended to what this "
        "surrogate sees, such as a sum or an interaction. They do not become degrees of "
        "freedom of the optimization.",
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
