from typing import List, Literal

from pydantic import Field, model_validator

from bofire.data_models.kernels.kernel import FeatureSpecificKernel, LengthscaleKernel
from bofire.data_models.types import NonRestrictedFeatureKeys


class WassersteinKernel(LengthscaleKernel, FeatureSpecificKernel):
    """Kernel based on the Wasserstein distance.

    It only works for 1D data that is monotonically increasing, as it is just
    calculating the integral of the absolute difference between two shapes.
    Only when both shapes are monotonically increasing, this integral is also
    a Wasserstein distance (https://arxiv.org/abs/2002.01878).

    The shape are assumed to be discretized as a set of points. Make sure that
    the discretization is fine enough to capture the shape of the data.
    A single lengthscale scales the distance between whole curves; its prior is
    mapped with dimensionality one, independently of the number of points.

    """

    type: Literal["WassersteinKernel"] = "WassersteinKernel"
    squared: bool = Field(
        default=False,
        description="Whether to use the squared exponential form rather than the "
        "absolute exponential one. The squared form is not positive definite at every "
        "lengthscale.",
    )


class ExactWassersteinKernel(LengthscaleKernel, FeatureSpecificKernel):
    """Kernel based on the exact 1D Wasserstein distance for piecewise-linear curves.

    Each input row encodes a piecewise-linear curve via the (x, y) coordinates of
    its break points. The kernel evaluates the Wasserstein distance between two
    such curves exactly by interpolating both on the union of their x-grids and
    integrating the (absolute or squared) difference analytically.

    Unlike `WassersteinKernel`, which approximates the distance from a discretization,
    this computes it analytically from the curves' break points.
    A single lengthscale scales the distance between whole curves; its prior is
    mapped with dimensionality one, independently of the number of break points.

    Examples:
        >>> ExactWassersteinKernel(
        ...     x_keys=["x_start", "x_end"],
        ...     y_keys=["y_start", "y_end"],
        ... )
    """

    type: Literal["ExactWassersteinKernel"] = "ExactWassersteinKernel"
    squared: bool = Field(
        default=False,
        description="Whether to use the squared exponential form rather than the "
        "absolute exponential one.",
    )
    x_keys: NonRestrictedFeatureKeys = Field(
        description="Ordered feature keys holding the curve's x-coordinates. Each "
        "key must resolve to one scalar input column and must be included in "
        "`features` when a subset is selected. Must not overlap `y_keys`.",
    )
    y_keys: NonRestrictedFeatureKeys = Field(
        description="Ordered feature keys holding the curve's y-coordinates. Each "
        "key must resolve to one scalar input column and must be included in "
        "`features` when a subset is selected. Coordinates are paired positionally "
        "after prepending and appending fixed values; both totals must match.",
    )
    prepend_x: List[float] = Field(
        default=[],
        description="Fixed x-coordinates placed before the variable ones, anchoring "
        "every curve at a known starting point.",
    )
    prepend_y: List[float] = Field(
        default=[],
        description="Fixed y-coordinates placed before the variable ones, paired with "
        "`prepend_x`.",
    )
    append_x: List[float] = Field(
        default=[],
        description="Fixed x-coordinates placed after the variable ones, anchoring "
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
        "computed, bringing curves onto a comparable scale.",
    )
    normalize_x: bool = Field(
        default=True,
        description="Whether to rescale each curve's x-coordinates by its own maximum, "
        "so only the shape is compared and not the extent.",
    )
    order: Literal[1, 2] = Field(
        default=1,
        description="Order of the Wasserstein distance: 1 integrates the absolute "
        "difference between the curves, 2 the squared difference.",
    )

    @model_validator(mode="after")
    def validate_keys(self) -> "ExactWassersteinKernel":
        if set(self.x_keys) & set(self.y_keys):
            raise ValueError("x_keys and y_keys must not overlap.")
        if self.features:
            missing = set(self.x_keys + self.y_keys) - set(self.features)
            if missing:
                raise ValueError(
                    f"Coordinate keys must be included in features: {sorted(missing)}."
                )
        n_x = len(self.x_keys) + len(self.prepend_x) + len(self.append_x)
        n_y = len(self.y_keys) + len(self.prepend_y) + len(self.append_y)
        if n_x != n_y:
            raise ValueError("Total number of x and y values must be equal.")
        return self
