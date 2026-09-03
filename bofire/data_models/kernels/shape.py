from typing import List, Literal

from pydantic import Field

from bofire.data_models.kernels.kernel import FeatureSpecificKernel, LengthscaleKernel


class WassersteinKernel(LengthscaleKernel, FeatureSpecificKernel):
    """Kernel based on the Wasserstein distance.

    It only works for 1D data that is monotonically increasing, as it is just
    calculating the integral of the absolute difference between two shapes.
    Only when both shapes are monotonically increasing, this integral is also
    a Wasserstein distance (https://arxiv.org/abs/2002.01878).

    The shape are assumed to be discretized as a set of points. Make sure that
    the discretization is fine enough to capture the shape of the data.

    """

    type: Literal["WassersteinKernel"] = "WassersteinKernel"
    squared: bool = Field(
        default=False,
        description="Whether to use the squared exponential form rather than the "
        "absolute exponential one. The squared form is not positive definite at every "
        "lengthscale, which is why the absolute form is the default.",
    )


class ExactWassersteinKernel(LengthscaleKernel, FeatureSpecificKernel):
    """Kernel based on the exact 1D Wasserstein distance for piecewise-linear curves.

    Each input row encodes a piecewise-linear curve via the (x, y) coordinates of
    its break points. The kernel evaluates the Wasserstein distance between two
    such curves exactly by interpolating both on the union of their x-grids and
    integrating the (absolute or squared) difference analytically.

    Unlike `WassersteinKernel`, which approximates the distance from a discretization,
    this computes it analytically from the curves' break points.
    """

    type: Literal["ExactWassersteinKernel"] = "ExactWassersteinKernel"
    squared: bool = Field(
        default=False,
        description="Whether to use the squared exponential form rather than the "
        "absolute exponential one.",
    )
    idx_x: List[int] = Field(
        description="Positions of the x-coordinates of the curve among the kernel's "
        "inputs.",
    )
    idx_y: List[int] = Field(
        description="Positions of the y-coordinates of the curve among the kernel's "
        "inputs, paired with `idx_x`.",
    )
    prepend_x: List[float] = Field(
        default=[],
        description="Fixed x-coordinates placed before the variable ones, for anchoring "
        "every curve at a known starting point.",
    )
    prepend_y: List[float] = Field(
        default=[],
        description="Fixed y-coordinates placed before the variable ones, paired with "
        "`prepend_x`.",
    )
    append_x: List[float] = Field(
        default=[],
        description="Fixed x-coordinates placed after the variable ones, for anchoring "
        "every curve at a known end point.",
    )
    append_y: List[float] = Field(
        default=[],
        description="Fixed y-coordinates placed after the variable ones, paired with "
        "`append_x`.",
    )
    normalize_y: float = Field(
        default=1.0,
        description="Divisor applied to the y-coordinates before the distance is "
        "computed, for bringing curves onto a comparable scale.",
    )
    normalize_x: bool = Field(
        default=True,
        description="Whether to rescale each curve's x-coordinates by its own maximum, "
        "so that only the shape of the curve is compared and not its extent.",
    )
    order: Literal[1, 2] = Field(
        default=1,
        description="Order of the Wasserstein distance: 1 integrates the absolute "
        "difference between the curves, 2 the squared difference.",
    )
