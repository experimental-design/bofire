"""Surrogate presets.

A preset is a named configuration of an existing surrogate. These two fix the kernel of
a `SingleTaskGPSurrogate` and adjust the defaults that choice implies, and they are
classes rather than functions because narrowing `kernel` is a *type* constraint: it is
enforced on assignment and on deserialization, it survives a round trip, and it reaches
`model_json_schema()`, none of which a function returning a configured surrogate can do.

The priors draw the line on the other side -- `THREESIX_LENGTHSCALE_PRIOR` and its
siblings are `partial`s over `GammaPrior`, because they only choose values and forbid
nothing.
"""

from typing import Literal, Optional

from pydantic import Field

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.kernels.api import LinearKernel, PolynomialKernel
from bofire.data_models.priors.api import THREESIX_NOISE_PRIOR, AnyPrior
from bofire.data_models.surrogates.botorch import KERNEL_DESCRIPTION
from bofire.data_models.surrogates.single_task_gp import (
    SingleTaskGPHyperconfig,
    SingleTaskGPSurrogate,
)


HYPERCONFIG_DESCRIPTION = (
    "Configuration of a hyperparameter optimization for this surrogate. There is no "
    "default one, because the single-task GP config varies over RBF and Matern and "
    "would discard the kernel this surrogate fixes."
)
NOISE_PRIOR_DESCRIPTION = (
    "Prior over the observation noise, which sets how much of the spread in the data "
    "the model attributes to measurement error rather than to the response. Defaults "
    "to the three-six gamma prior rather than to the log-normal one a single-task GP "
    "uses."
)


class LinearSurrogate(SingleTaskGPSurrogate):
    """Gaussian process restricted to linear responses.

    The linear kernel still yields a predicted uncertainty, so the surrogate can be used
    in a Bayesian optimization loop, but it cannot represent curvature. Pick it when the
    response is known to be linear, or when there are too few experiments to support
    anything richer.

    Examples:
        >>> surrogate = LinearSurrogate(inputs=inputs, outputs=outputs)
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
    hyperconfig: Optional[SingleTaskGPHyperconfig] = Field(
        default=None,
        description=HYPERCONFIG_DESCRIPTION,
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
        description=NOISE_PRIOR_DESCRIPTION,
    )
    hyperconfig: Optional[SingleTaskGPHyperconfig] = Field(
        default=None,
        description=HYPERCONFIG_DESCRIPTION,
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
