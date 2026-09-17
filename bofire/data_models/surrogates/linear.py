from typing import Literal, Optional, Type

from pydantic import Field

# from bofire.data_models.strategies.api import FactorialStrategy
from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import LinearKernel
from bofire.data_models.priors.api import (
    THREESIX_NOISE_PRIOR,
    AnyPrior,
    AnyPriorConstraint,
    GreaterThan,
)
from bofire.data_models.surrogates.botorch import KERNEL_DESCRIPTION
from bofire.data_models.surrogates.trainable_botorch import (
    NOISE_CONSTRAINT_DESCRIPTION,
    NOISE_PRIOR_DESCRIPTION,
    TrainableBotorchSurrogate,
)


class LinearSurrogate(TrainableBotorchSurrogate):
    """Gaussian process restricted to linear responses.

    Still predicts an uncertainty, so it can be used in a Bayesian optimization loop,
    but it cannot represent curvature. Pick it when the response is known to be linear,
    or when there are too few experiments to support anything richer.
    """

    type: Literal["LinearSurrogate"] = "LinearSurrogate"

    kernel: LinearKernel = Field(
        default_factory=lambda: LinearKernel(),
        description=KERNEL_DESCRIPTION
        + " Fixed to the linear kernel, which is what restricts the response to a "
        "linear one.",
    )
    noise_prior: AnyPrior = Field(
        default_factory=lambda: THREESIX_NOISE_PRIOR(),
        description=NOISE_PRIOR_DESCRIPTION,
    )
    noise_constraint: Optional[AnyPriorConstraint] = Field(
        default_factory=lambda: GreaterThan(lower_bound=1e-4),
        description=NOISE_CONSTRAINT_DESCRIPTION,
    )

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
