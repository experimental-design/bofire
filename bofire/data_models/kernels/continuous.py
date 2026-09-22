from typing import List, Literal, Optional, Union

from pydantic import Field, PositiveInt, model_validator

from bofire.data_models.kernels.kernel import (
    ARDKernel,
    FeatureSpecificKernel,
    LengthscaleKernel,
)
from bofire.data_models.priors.api import AnyPrior


class ContinuousKernel(FeatureSpecificKernel):
    """Kernel acting on continuous inputs.

    Compares inputs by numeric distance, so it can act on a categorical only when the
    encoding turns it into coordinates that a distance is meaningful over -- one-hot or
    descriptor columns, but not the integer codes of an ordinal encoding. A task input
    is excluded: which task an observation belongs to is not a position in the space.
    """

    @classmethod
    def can_consume(cls, feat, encoding=None) -> bool:
        from bofire.data_models.encodings.api import DescriptorEncoding, OneHotEncoding
        from bofire.data_models.features.api import (
            CategoricalInput,
            NumericalInput,
            TaskInput,
        )

        if isinstance(feat, TaskInput):
            return False
        if isinstance(feat, NumericalInput):
            return True
        if isinstance(feat, CategoricalInput):
            return isinstance(encoding, (OneHotEncoding, DescriptorEncoding))
        return False


class RBFKernel(ARDKernel, LengthscaleKernel, ContinuousKernel):
    r"""Radial basis function kernel, the usual default for continuous inputs.

    $$
    k(\mathbf x, \mathbf x') = \exp\left(-\frac{\lVert \mathbf x - \mathbf x' \rVert^2}
                                              {2\ell^2}\right)
    $$

    Samples from this kernel are infinitely differentiable, so it assumes a very smooth
    response. Use `MaternKernel` where the response is expected to be rougher.
    """

    type: Literal["RBFKernel"] = "RBFKernel"


class MaternKernel(ARDKernel, LengthscaleKernel, ContinuousKernel):
    r"""Matern kernel, a less smooth alternative to the RBF kernel.

    $$
    k(\mathbf x, \mathbf x') = \frac{2^{1-\nu}}{\Gamma(\nu)}
        \left(\sqrt{2\nu}\,\frac{d}{\ell}\right)^{\nu}
        K_{\nu}\!\left(\sqrt{2\nu}\,\frac{d}{\ell}\right),
    \qquad d = \lVert \mathbf x - \mathbf x' \rVert
    $$

    Samples are $\lceil \nu \rceil - 1$ times differentiable, so $\nu$ sets how rough
    the response may be. Reach for this when an RBF fit looks implausibly smooth
    between observations.
    """

    type: Literal["MaternKernel"] = "MaternKernel"
    nu: Literal[0.5, 1.5, 2.5] = Field(
        default=2.5,
        description="Smoothness parameter. 0.5 gives a nowhere-differentiable "
        "response, 1.5 a once-differentiable one and 2.5 a twice-differentiable one.",
    )


class LinearKernel(ContinuousKernel):
    r"""Linear kernel, equivalent to Bayesian linear regression on the inputs.

    $$
    k(\mathbf x, \mathbf x') = v\,\mathbf x^{\top} \mathbf x'
    $$
    """

    type: Literal["LinearKernel"] = "LinearKernel"
    variance_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the variance $v$, which scales the whole kernel and so "
        "sets the magnitude of the linear response.",
    )


class PolynomialKernel(ContinuousKernel):
    r"""Polynomial kernel, of a fixed degree in the inputs.

    $$
    k(\mathbf x, \mathbf x') = (\mathbf x^{\top} \mathbf x' + c)^{p}
    $$
    """

    type: Literal["PolynomialKernel"] = "PolynomialKernel"
    offset_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the offset $c$, which weights the lower-order terms "
        "against the highest one: a large offset makes the kernel more nearly linear.",
    )
    power: int = Field(
        default=2,
        description="Degree $p$ of the polynomial. $p=2$ captures pairwise "
        "interactions and quadratic curvature; higher degrees fit more complex "
        "functions, but lead to worse extrapolation and overfitting.",
    )


class InfiniteWidthBNNKernel(ContinuousKernel):
    """Kernel equivalent to a Bayesian neural network of infinite width.

    Captures the kind of hierarchical, non-stationary structure a deep network would,
    rather than assuming one lengthscale applies across the whole space.
    """

    type: Literal["InfiniteWidthBNNKernel"] = "InfiniteWidthBNNKernel"
    depth: PositiveInt = Field(
        default=3,
        description="Number of layers in the equivalent network. More layers allow a "
        "less stationary response.",
    )


class SphericalLinearKernel(ARDKernel, LengthscaleKernel, ContinuousKernel):
    """Spherical linear kernel for continuous inputs.

    This kernel projects the inputs onto a unit sphere and computes the linear kernel in
    this space, so it responds to the direction of an input vector rather than its
    magnitude.
    """

    type: Literal["SphericalLinearKernel"] = "SphericalLinearKernel"
    bounds: Union[tuple[float, float], List[tuple[float, float]]] = Field(
        default=(0.0, 1.0),
        description="Range the inputs are rescaled from before projection onto the "
        "sphere. A single pair applies to every input; a list gives one pair per input, "
        "and is required when `ard` is disabled.",
    )

    @model_validator(mode="after")
    def validate_ard_bounds(self):
        if not self.ard and not isinstance(self.bounds, list):
            raise ValueError(
                "Cannot determine number of dimensions. If ard=False then list of bounds should have length equal to the input dimension."
            )
        return self
