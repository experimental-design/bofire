from typing import List, Literal, Optional

from bofire.data_models.kernels.kernel import Kernel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint


class WassersteinKernel(Kernel):
    """Kernel based on the Wasserstein distance.

    It only works for 1D data that is monotonically increasing, as it is just
    calculating the integral of the absolute difference between two shapes.
    Only when both shapes are monotonically increasing, this integral is also
    a Wasserstein distance (https://arxiv.org/abs/2002.01878).

    The shape are assumed to be discretized as a set of points. Make sure that
    the discretization is fine enough to capture the shape of the data.

    Attributes:
        squared: If True, the squared exponential Wasserstein distance is used. Note
            that the squared exponential Wasserstein distance kernel is not positive
            definite for all lengthscales. For this reason, as default the absolute
            exponential Wasserstein distance is used.
        lengthscale_prior: Prior for the lengthscale of the kernel.

    """

    type: Literal["WassersteinKernel"] = "WassersteinKernel"
    squared: bool = False
    lengthscale_prior: Optional[AnyPrior] = None
    lengthscale_constraint: Optional[AnyPriorConstraint] = None


class ExactWassersteinKernel(Kernel):
    """Kernel based on the exact 1D Wasserstein distance for piecewise-linear curves."""

    type: Literal["ExactWassersteinKernel"] = "ExactWassersteinKernel"
    squared: bool = False
    lengthscale_prior: Optional[AnyPrior] = None
    lengthscale_constraint: Optional[AnyPriorConstraint] = None
    idx_x: List[int]
    idx_y: List[int]
    prepend_x: List[float] = []
    prepend_y: List[float] = []
    append_x: List[float] = []
    append_y: List[float] = []
    normalize_y: float = 1.0
    normalize_x: bool = True
    order: Literal[1, 2] = 1
