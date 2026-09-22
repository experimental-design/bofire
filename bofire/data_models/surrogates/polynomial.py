from typing import Literal, Optional

from pydantic import Field

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.kernels.api import PolynomialKernel
from bofire.data_models.priors.api import THREESIX_NOISE_PRIOR, AnyPrior
from bofire.data_models.surrogates.botorch import KERNEL_DESCRIPTION
from bofire.data_models.surrogates.single_task_gp import (
    SingleTaskGPHyperconfig,
    SingleTaskGPSurrogate,
)
from bofire.data_models.surrogates.trainable_botorch import (
    HYPERCONFIG_DESCRIPTION,
    NOISE_PRIOR_DESCRIPTION,
)


class PolynomialSurrogate(SingleTaskGPSurrogate):
    """Gaussian process restricted to polynomial responses of a fixed degree.

    The polynomial kernel expresses curvature and interactions between inputs, but only
    up to the kernel's `power`, so the fit stays interpretable where an RBF kernel would
    take an arbitrary shape. Pick it when a response surface of a known order is
    expected, as in a classical DoE.

    Examples:
        >>> surrogate = PolynomialSurrogate(
        ...     inputs=inputs, outputs=outputs, kernel=PolynomialKernel(power=3)
        ... )
    """

    type: Literal["PolynomialSurrogate"] = "PolynomialSurrogate"

    kernel: PolynomialKernel = Field(
        default_factory=lambda: PolynomialKernel(power=2),
        description=KERNEL_DESCRIPTION
        + " Fixed to the polynomial kernel, whose `power` sets the degree of the "
        "response.",
    )
    noise_prior: AnyPrior = Field(
        default_factory=lambda: THREESIX_NOISE_PRIOR(),
        description=NOISE_PRIOR_DESCRIPTION
        + " Defaults to the three-six gamma prior rather than to the log-normal one a "
        "single-task GP uses.",
    )
    hyperconfig: Optional[SingleTaskGPHyperconfig] = Field(
        default=None,
        description=HYPERCONFIG_DESCRIPTION
        + " There is no default one, because the single-task GP config varies over RBF "
        "and Matern and would discard the polynomial kernel.",
    )

    @staticmethod
    def from_power(power: int, inputs: Inputs, outputs: Outputs):
        """Build a surrogate whose polynomial kernel has the given degree.

        Args:
            power: Degree of the polynomial response.
            inputs: Input features the surrogate acts on.
            outputs: Output feature the surrogate predicts.

        Returns:
            The configured surrogate.
        """
        return PolynomialSurrogate(
            kernel=PolynomialKernel(power=power),
            inputs=inputs,
            outputs=outputs,
        )
