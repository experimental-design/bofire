from typing import Literal, Optional

from pydantic import Field

from bofire.data_models.kernels.api import LinearKernel
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
        description=NOISE_PRIOR_DESCRIPTION
        + " Defaults to the three-six gamma prior rather than to the log-normal one a "
        "single-task GP uses.",
    )
    hyperconfig: Optional[SingleTaskGPHyperconfig] = Field(
        default=None,
        description=HYPERCONFIG_DESCRIPTION
        + " There is no default one, because the single-task GP config varies over RBF "
        "and Matern and would discard the linear kernel.",
    )
