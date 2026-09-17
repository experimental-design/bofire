from typing import Literal, Optional, Type

from pydantic import Field

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import PolynomialKernel
from bofire.data_models.priors.api import (
    THREESIX_NOISE_PRIOR,
    AnyPrior,
    AnyPriorConstraint,
    GreaterThan,
)
from bofire.data_models.surrogates.trainable_botorch import (
    NOISE_CONSTRAINT_DESCRIPTION,
    NOISE_PRIOR_DESCRIPTION,
    TrainableBotorchSurrogate,
)


class PolynomialSurrogate(TrainableBotorchSurrogate):
    """Gaussian process restricted to polynomial responses of a fixed degree.

    Expresses curvature and interactions between inputs, but only up to that degree, so
    it stays interpretable where an RBF kernel would fit an arbitrary shape. Pick it
    when a response surface of a known order is expected, as in a classical DoE.

    Examples:
        >>> surrogate = PolynomialSurrogate(
        ...     inputs=inputs, outputs=outputs, kernel=PolynomialKernel(power=3)
        ... )
    """

    type: Literal["PolynomialSurrogate"] = "PolynomialSurrogate"

    kernel: PolynomialKernel = Field(
        default_factory=lambda: PolynomialKernel(power=2),
        description="Covariance function. Fixed to the polynomial kernel, whose "
        "`power` sets the degree of the response.",
    )
    noise_prior: AnyPrior = Field(
        default_factory=lambda: THREESIX_NOISE_PRIOR(),
        description=NOISE_PRIOR_DESCRIPTION,
    )
    noise_constraint: Optional[AnyPriorConstraint] = Field(
        default_factory=lambda: GreaterThan(lower_bound=1e-4),
        description=NOISE_CONSTRAINT_DESCRIPTION,
    )

    @staticmethod
    def from_power(power: int, inputs: Inputs, outputs: Outputs):
        """Build a surrogate whose polynomial kernel has the given degree.

        Args:
            power: Degree of the polynomial response.
            inputs: Input features the surrogate is fitted on.
            outputs: Output feature the surrogate predicts.

        Returns:
            The configured surrogate.
        """
        return PolynomialSurrogate(
            kernel=PolynomialKernel(power=power),
            inputs=inputs,
            outputs=outputs,
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
