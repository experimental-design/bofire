from typing import Any, Optional

from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint
from bofire.data_models.types import NonRestrictedFeatureKeys


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
        "feature contributes every dimension it expands to. Defaults to all inputs.",
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
