from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import EngineeredFeatures, Inputs
from bofire.data_models.encodings.api import AnyCategoricalEncoding
from bofire.data_models.features.api import AnyFeature
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint
from bofire.data_models.types import InputTransformSpecs, NonRestrictedFeatureKeys


@dataclass(frozen=True)
class KernelInputs:
    """What a kernel is applied to: the features on offer and how they are encoded.

    Attributes:
        inputs: The inputs of the surrogate the kernel belongs to.
        encodings: How each categorical input is encoded, with defaults already filled
            in.
        engineered_features: Quantities the surrogate derives from its inputs.
        offered: Keys of the features a kernel receives when it leaves `features`
            unset. This is decided by the surrogate, not by the kernel.
    """

    inputs: Inputs
    encodings: InputTransformSpecs
    engineered_features: EngineeredFeatures
    offered: Tuple[str, ...]

    def keys(self) -> List[str]:
        """Keys of every input and engineered feature."""
        return self.inputs.get_keys() + self.engineered_features.get_keys()

    def get(self, key: str) -> AnyFeature:
        """The input or engineered feature with this key.

        Raises:
            KeyError: If no input or engineered feature has this key.
        """
        if key in self.inputs.get_keys():
            return self.inputs.get_by_key(key)
        if key in self.engineered_features.get_keys():
            return self.engineered_features.get_by_key(key)
        raise KeyError(key)

    def encoding(self, key: str) -> Optional[AnyCategoricalEncoding]:
        """How the feature with this key is encoded; `None` for non-categoricals."""
        return self.encodings.get(key)


class Kernel(BaseModel):
    r"""Covariance function of a Gaussian process.

    A kernel $k(\mathbf x, \mathbf x')$ gives the prior covariance between the function
    values at two points of the input space. It encodes the assumptions the surrogate
    makes about the response: how smooth it is, over what distance observations carry
    information, and which inputs matter at all.
    """

    type: Any

    def children(self) -> List["Kernel"]:
        """The kernels this kernel is composed of; empty for a kernel on inputs."""
        return []

    def validate_inputs(self, context: KernelInputs) -> None:
        """Check that this kernel can work on what it is applied to.

        Checks the kernel itself, then every kernel it is composed of.

        Args:
            context: The features on offer and how they are encoded.

        Raises:
            ValueError: If this kernel, or one it contains, cannot work on its features.
        """
        for child in self.children():
            child.validate_inputs(context)


class AggregationKernel(Kernel):
    """Kernel built by combining other kernels rather than acting on inputs directly."""

    pass


class FeatureSpecificKernel(Kernel):
    """Kernel that can be restricted to a subset of the input dimensions."""

    features: Optional[NonRestrictedFeatureKeys] = Field(
        default=None,
        description="Keys of the features this kernel is evaluated on; an engineered "
        "feature contributes every dimension it expands to. Defaults to all features "
        "the surrogate offers the kernel.",
    )

    @classmethod
    def can_consume(
        cls,
        feat: AnyFeature,
        encoding: Optional[AnyCategoricalEncoding] = None,
    ) -> bool:
        """Whether this kernel can work on a feature at all, given how it is encoded.

        Only combinations the kernel cannot meaningfully compute are rejected; whether a
        combination is a good modelling choice is left to the caller.

        Args:
            feat: The input or engineered feature in question.
            encoding: How that feature is encoded if it is categorical, else `None`.

        Returns:
            Whether the feature can be part of this kernel's inputs.
        """
        return True

    def selected_features(self, context: KernelInputs) -> List[str]:
        """Keys of the features this kernel is applied to.

        Args:
            context: The features on offer and how they are encoded.

        Returns:
            `features` if set, else everything the surrogate offers.
        """
        if self.features is not None:
            return list(self.features)
        return list(context.offered)

    def validate_inputs(self, context: KernelInputs) -> None:
        """Check that every selected feature exists and can be worked on.

        Raises:
            ValueError: If a selected key names no feature, or names one this kernel
                cannot work on.
        """
        selected = self.selected_features(context)
        known = set(context.keys())
        if unknown := [key for key in selected if key not in known]:
            raise ValueError(
                f"{type(self).__name__} names {sorted(unknown)}, which are neither "
                f"inputs nor engineered features."
            )
        if rejected := [
            key
            for key in selected
            if not self.can_consume(context.get(key), context.encoding(key))
        ]:
            raise ValueError(
                f"{type(self).__name__} cannot work on {sorted(rejected)}."
            )
        super().validate_inputs(context)


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
