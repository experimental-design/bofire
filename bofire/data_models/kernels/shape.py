from typing import List, Literal, Optional

from bofire.data_models.kernels.kernel import FeatureSpecificKernel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint


class WassersteinKernel(FeatureSpecificKernel):
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
        lengthscale_constraint: Optional constraint for the lengthscale (e.g.
            positivity or bounds).

    """

    type: Literal["WassersteinKernel"] = "WassersteinKernel"
    squared: bool = False
    lengthscale_prior: Optional[AnyPrior] = None
    lengthscale_constraint: Optional[AnyPriorConstraint] = None


class ExactWassersteinKernel(FeatureSpecificKernel):
    """Kernel based on the Exact 1 or 2 1D Wasserstein distance.

    Attributes:
        squared: If True, use the squared-exponential Wasserstein kernel.
        lengthscale_prior: Prior for the kernel lengthscale.
        lengthscale_constraint: Optional constraint for the lengthscale.
        idx_x: List of indices to select x coordinates from input vectors.
        idx_y: List of indices to select y coordinates from input vectors.
        prepend_x: Values to prepend to x vectors prior to distance computation.
        prepend_y: Values to prepend to y vectors.
        append_x: Values to append to x vectors prior to distance computation.
        append_y: Values to append to y vectors.
        normalize_y: Scaling factor to divide y values by before computing the
            distance (1.0 means no scaling).
        normalize_x: Whether to normalize x coordinates to unit range before
            computing the distance.
        order: Order of the Wasserstein distance (1 or 2).

    """

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
