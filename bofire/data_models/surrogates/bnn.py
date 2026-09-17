from typing import Literal, Optional

from pydantic import Field

from bofire.data_models.kernels.api import InfiniteWidthBNNKernel
from bofire.data_models.surrogates.botorch import KERNEL_DESCRIPTION
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate
from bofire.data_models.surrogates.trainable import Hyperconfig


class SingleTaskIBNNSurrogate(SingleTaskGPSurrogate):
    """Gaussian process that behaves like a Bayesian neural network of infinite width.

    Unlike the stationary kernels, it does not assume one notion of "close" holds
    everywhere, so it suits a response that is flat over part of the space and sharp
    over another.
    """

    type: Literal["SingleTaskIBNNSurrogate"] = "SingleTaskIBNNSurrogate"
    kernel: InfiniteWidthBNNKernel = Field(
        default=InfiniteWidthBNNKernel(),
        description=KERNEL_DESCRIPTION
        + " Only the depth of the equivalent network is configurable; there is no "
        "lengthscale to set.",
    )
    hyperconfig: Optional[Hyperconfig] = Field(
        default=None,
        description="Configuration of a hyperparameter optimization for this surrogate. "
        "There is no default one, since the hyperparameters the single-task GP config "
        "varies do not apply to this kernel.",
    )
