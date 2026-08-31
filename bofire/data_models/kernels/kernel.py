from typing import Any, Optional

from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint
from bofire.data_models.types import NonRestrictedFeatureKeys


class Kernel(BaseModel):
    """Defines how similar the surrogate expects two candidates to behave.

    The kernel is the assumption a Gaussian process makes about the response surface:
    how smooth it is, over what distance points inform each other, and which inputs
    matter at all. It is the main thing to change when a surrogate fits the data badly.
    """

    type: Any


class AggregationKernel(Kernel):
    """Kernel built by combining other kernels rather than acting on inputs directly."""

    pass


class FeatureSpecificKernel(Kernel):
    """Kernel that can be restricted to a subset of the inputs."""

    features: Optional[NonRestrictedFeatureKeys] = Field(
        default=None,
        description="Keys of the features this kernel acts on. Naming an engineered "
        "feature selects every input it contributes. If not provided, the kernel acts "
        "on all inputs of the surrogate.",
    )


class ARDKernel(BaseModel):
    """Mixin for a kernel that can learn a separate lengthscale per input."""

    ard: bool = Field(
        default=True,
        description="Whether to fit one lengthscale per input rather than a single "
        "shared one. Separate lengthscales let the model discover that some inputs "
        "matter more than others, at the cost of more parameters to fit, so turning "
        "this off can help when there is little data.",
    )


class LengthscaleKernel(BaseModel):
    """Mixin for a kernel with a fitted lengthscale."""

    lengthscale_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the lengthscale, which sets how far apart two "
        "candidates can be and still inform each other. Worth setting when there is "
        "little data, since the fitted value is otherwise driven by whatever few "
        "points are available. If not provided, the surrogate's default is used.",
    )
    lengthscale_constraint: Optional[AnyPriorConstraint] = Field(
        default=None,
        description="Hard bounds on the lengthscale, enforced during fitting rather "
        "than merely preferred as a prior is. If not provided, the surrogate's default "
        "is used.",
    )
