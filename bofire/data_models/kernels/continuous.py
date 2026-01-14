from typing import List, Literal, Optional, Union

from pydantic import PositiveInt, field_validator, model_validator

from bofire.data_models.kernels.kernel import FeatureSpecificKernel
from bofire.data_models.priors.api import AnyGeneralPrior, AnyPrior, AnyPriorConstraint


class ContinuousKernel(FeatureSpecificKernel):
    pass


class RBFKernel(ContinuousKernel):
    type: Literal["RBFKernel"] = "RBFKernel"
    ard: bool = True
    lengthscale_prior: Optional[AnyPrior] = None
    lengthscale_constraint: Optional[AnyPriorConstraint] = None


class MaternKernel(ContinuousKernel):
    type: Literal["MaternKernel"] = "MaternKernel"
    ard: bool = True
    nu: float = 2.5
    lengthscale_prior: Optional[AnyPrior] = None
    lengthscale_constraint: Optional[AnyPriorConstraint] = None

    @field_validator("nu")
    def validate_nu(cls, nu):
        if nu not in {0.5, 1.5, 2.5}:
            raise ValueError("nu expected to be 0.5, 1.5, or 2.5")
        return nu


class LinearKernel(ContinuousKernel):
    type: Literal["LinearKernel"] = "LinearKernel"
    variance_prior: Optional[AnyGeneralPrior] = None


class PolynomialKernel(ContinuousKernel):
    type: Literal["PolynomialKernel"] = "PolynomialKernel"
    offset_prior: Optional[AnyGeneralPrior] = None
    power: int = 2


class InfiniteWidthBNNKernel(ContinuousKernel):
    features: Optional[List[str]] = None
    type: Literal["InfiniteWidthBNNKernel"] = "InfiniteWidthBNNKernel"
    depth: PositiveInt = 3


class SphericalLinearKernel(ContinuousKernel):
    """Spherical linear kernel for continuous inputs.
    This kernel projects the inputs onto a unit sphere and computes the linear kernel in this space.
    Attributes:
        ard: Whether to use Automatic Relevance Determination. If True, separate lengthscales
            are learned for each input dimension. Defaults to True.
        lengthscale_prior: Optional prior distribution for the lengthscale parameter(s).
        lengthscale_constraint: Optional constraint on the lengthscale parameter(s).
        bounds: Bounds for the input features. Can be a single tuple for all dimensions
            or a list of tuples for per-dimension bounds. Defaults to (0.0, 1.0).
    Raises:
        ValueError: If ard is False and bounds is not a list with length equal to input dimension.
    """

    type: Literal["SphericalLinearKernel"] = "SphericalLinearKernel"
    ard: bool = True
    lengthscale_prior: Optional[AnyPrior] = None
    lengthscale_constraint: Optional[AnyPriorConstraint] = None
    bounds: Union[tuple[float, float], List[tuple[float, float]]] = (0.0, 1.0)

    @model_validator(mode="after")
    def validate_ard_bounds(self):
        if not self.ard and not isinstance(self.bounds, list):
            raise ValueError(
                "Cannot determine number of dimensions. If ard=False then list of bounds should have length equal to the input dimension."
            )
        return self
