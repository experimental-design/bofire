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

    """

    type: Literal["WassersteinKernel"] = "WassersteinKernel"
    squared: bool = False
    lengthscale_prior: Optional[AnyPrior] = None
    lengthscale_constraint: Optional[AnyPriorConstraint] = None


class ExactWassersteinKernel(FeatureSpecificKernel):
    """Kernel based on the exact 1D Wasserstein distance for piecewise-linear curves.

    Each input row encodes a piecewise-linear curve via the (x, y) coordinates of
    its break points. The kernel evaluates the Wasserstein distance between two
    such curves exactly by interpolating both on the union of their x-grids and
    integrating the (absolute or squared) difference analytically.

    Attributes:
        squared: If True, the squared exponential Wasserstein distance is used.
         By default the absolute exponential form is used.
        lengthscale_prior: Prior for the lengthscale of the kernel.
        lengthscale_constraint: Constraint applied to the lengthscale prior.
        idx_x: Column indices in the input tensor holding the x-coordinates of
            the curves.
        idx_y: Column indices in the input tensor holding the y-coordinates of
            the curves.
        prepend_x: Fixed x-coordinates prepended to every curve before the
            variable section (e.g. a left anchor point).
        prepend_y: Fixed y-coordinates prepended to every curve, paired with
            `prepend_x`.
        append_x: Fixed x-coordinates appended to every curve after the variable
            section (e.g. a right anchor point).
        append_y: Fixed y-coordinates appended to every curve, paired with
            `append_x`.
        normalize_y: Scalar by which the y-coordinates are divided before the
            distance is computed.
        normalize_x: If True, x-coordinates are rescaled by their per-curve
            maximum so that the integration domain is normalized to [0, 1].
        order: Order of the Wasserstein distance. `1` computes W1 (integral of
            the absolute difference); `2` computes W2 (square root of the
            integral of the squared difference).

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
