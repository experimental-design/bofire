from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint
from bofire.data_models.types import NonRestrictedFeatureKeys


if TYPE_CHECKING:
    from bofire.data_models.domain.api import EngineeredFeatures, Inputs
    from bofire.data_models.encodings.api import AnyCategoricalEncoding
    from bofire.data_models.features.api import AnyInput
    from bofire.data_models.types import InputTransformSpecs


class Kernel(BaseModel):
    r"""Covariance function of a Gaussian process.

    A kernel $k(\mathbf x, \mathbf x')$ gives the prior covariance between the function
    values at two points of the input space. It encodes the assumptions the surrogate
    makes about the response: how smooth it is, over what distance observations carry
    information, and which inputs matter at all.
    """

    type: Any


class AggregationKernel(Kernel):
    """Kernel built by combining other kernels rather than acting on inputs directly."""

    pass


class FeatureSpecificKernel(Kernel):
    """Kernel that can be restricted to a subset of the input dimensions."""

    features: Optional[NonRestrictedFeatureKeys] = Field(
        default=None,
        description="Keys of the features this kernel is evaluated on; an engineered "
        "feature contributes every dimension it expands to. Defaults to every feature "
        "this kernel is able to act on.",
    )

    @classmethod
    def can_consume(
        cls,
        feat: "AnyInput",
        encoding: "Optional[AnyCategoricalEncoding]" = None,
    ) -> bool:
        """Whether this kernel can act on a feature, given how it is encoded.

        A kernel sees numbers, not features, so the answer depends on the encoding as
        much as on the feature: a categorical is meaningful to a distance-based kernel
        when it is one-hot encoded and meaningless when it carries integer codes.

        Args:
            feat: The input feature in question.
            encoding: How that feature is encoded for the surrogate, if it is encoded
                at all. Numerical features pass `None`.

        Returns:
            Whether the feature can be part of this kernel's inputs.
        """
        return True

    @classmethod
    def accepted_encodings(
        cls,
        feat: "AnyInput",
        candidates: "Sequence[AnyCategoricalEncoding]",
    ) -> "Tuple[AnyCategoricalEncoding, ...]":
        """Which of the offered encodings this kernel could work with.

        The first entry is the one the kernel would ask for, and the number of entries
        says how particular the kernel is about this feature -- a kernel that accepts it
        under one encoding has a stronger claim to it than one that accepts it under
        several.

        Args:
            feat: The input feature in question.
            candidates: The encodings on offer for that feature, in the order they
                should be preferred.

        Returns:
            The acceptable ones, in the order they were offered.
        """
        return tuple(c for c in candidates if cls.can_consume(feat, c))

    def resolve_features(
        self,
        inputs: "Inputs",
        encodings: "InputTransformSpecs",
        engineered_features: "Optional[EngineeredFeatures]" = None,
    ) -> List[str]:
        """The feature keys this kernel acts on in a given domain.

        Args:
            inputs: The inputs of the surrogate this kernel belongs to.
            encodings: How each categorical input is encoded.
            engineered_features: Quantities derived from the inputs, if any.

        Returns:
            The selected keys, in the order the inputs declare them.
        """
        engineered_keys = (
            engineered_features.get_keys() if engineered_features is not None else []
        )
        if self.features is not None:
            return list(self.features)
        return [
            key
            for key in inputs.get_keys()
            if type(self).can_consume(inputs.get_by_key(key), encodings.get(key))
        ] + engineered_keys

    def validate_inputs(
        self,
        inputs: "Inputs",
        encodings: "InputTransformSpecs",
        engineered_features: "Optional[EngineeredFeatures]" = None,
    ) -> None:
        """Check that the features this kernel selects are ones it can act on.

        Args:
            inputs: The inputs of the surrogate this kernel belongs to.
            encodings: How each categorical input is encoded.
            engineered_features: Quantities derived from the inputs, if any.

        Raises:
            ValueError: If an explicitly named feature cannot be consumed.
        """
        if self.features is not None:
            selected = self.resolve_features(inputs, encodings, engineered_features)
            engineered_keys = (
                engineered_features.get_keys()
                if engineered_features is not None
                else []
            )
            rejected = [
                key
                for key in selected
                if key not in engineered_keys
                and not type(self).can_consume(
                    inputs.get_by_key(key), encodings.get(key)
                )
            ]
            if rejected:
                raise ValueError(
                    f"{type(self).__name__} cannot act on {sorted(rejected)}. Either "
                    f"drop them from `features` or encode them differently."
                )


class ARDKernel(BaseModel):
    r"""Mixin for a kernel supporting automatic relevance determination."""

    ard: bool = Field(
        default=True,
        description="Whether to fit a separate lengthscale per input dimension rather "
        "than one shared lengthscale, letting the model down-weight dimensions the "
        "response does not depend on.",
    )


class LengthscaleKernel(BaseModel):
    r"""Mixin for a kernel parametrized by a lengthscale $\ell$."""

    lengthscale_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the lengthscale, which governs the distance over which "
        "function values stay correlated: a short lengthscale means a rapidly varying "
        "response.",
    )
    lengthscale_constraint: Optional[AnyPriorConstraint] = Field(
        default=None,
        description="Bounds the lengthscale is restricted to during fitting. Unlike a "
        "prior, which only shifts the optimum, values outside cannot be reached.",
    )
