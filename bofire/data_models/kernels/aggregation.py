from collections.abc import Sequence
from typing import Literal, Optional, Union

from pydantic import Field

from bofire.data_models.kernels.categorical import (
    HammingDistanceKernel,
    IndexKernel,
    PositiveIndexKernel,
)
from bofire.data_models.kernels.conditional import WedgeKernel
from bofire.data_models.kernels.continuous import (
    InfiniteWidthBNNKernel,
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    RBFKernel,
    SphericalLinearKernel,
)
from bofire.data_models.kernels.fidelity import DownsamplingKernel
from bofire.data_models.kernels.kernel import AggregationKernel
from bofire.data_models.kernels.molecular import TanimotoKernel
from bofire.data_models.kernels.shape import ExactWassersteinKernel, WassersteinKernel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint


class AdditiveKernel(AggregationKernel):
    r"""Sum of several kernels, $k(\mathbf x, \mathbf x') = \sum_i k_i(\mathbf x, \mathbf x')$.

    A sum models a response made of independent additive contributions, and the
    covariance stays high if any one of the summed kernels is high.
    """

    type: Literal["AdditiveKernel"] = "AdditiveKernel"
    kernels: Sequence[
        Union[
            RBFKernel,
            SphericalLinearKernel,
            MaternKernel,
            LinearKernel,
            HammingDistanceKernel,
            IndexKernel,
            PositiveIndexKernel,
            TanimotoKernel,
            WassersteinKernel,
            ExactWassersteinKernel,
            DownsamplingKernel,
            WedgeKernel,
            "AdditiveKernel",
            "MultiplicativeKernel",
            "ScaleKernel",
        ]
    ] = Field(description="The kernels to sum.")


class MultiplicativeKernel(AggregationKernel):
    r"""Product of several kernels, $k(\mathbf x, \mathbf x') = \prod_i k_i(\mathbf x, \mathbf x')$.

    A product models an interaction: the covariance is high only where every one of the
    multiplied kernels is high, so this is how a dependence across input types is
    expressed.
    """

    type: Literal["MultiplicativeKernel"] = "MultiplicativeKernel"
    kernels: Sequence[
        Union[
            RBFKernel,
            SphericalLinearKernel,
            MaternKernel,
            LinearKernel,
            HammingDistanceKernel,
            IndexKernel,
            PositiveIndexKernel,
            AdditiveKernel,
            TanimotoKernel,
            WassersteinKernel,
            ExactWassersteinKernel,
            DownsamplingKernel,
            WedgeKernel,
            "MultiplicativeKernel",
            "ScaleKernel",
        ]
    ] = Field(description="The kernels to multiply.")


class ScaleKernel(AggregationKernel):
    r"""Wraps another kernel with a fitted output scale,
    $k(\mathbf x, \mathbf x') = \theta\, k_{\text{base}}(\mathbf x, \mathbf x')$.

    The base kernel sets the shape of the covariance and this sets its magnitude, which
    is the variance of the response. Most kernels are wrapped in one before use, since
    the base kernels are normalized.
    """

    type: Literal["ScaleKernel"] = "ScaleKernel"
    base_kernel: Union[
        RBFKernel,
        SphericalLinearKernel,
        MaternKernel,
        LinearKernel,
        HammingDistanceKernel,
        IndexKernel,
        PositiveIndexKernel,
        AdditiveKernel,
        MultiplicativeKernel,
        TanimotoKernel,
        DownsamplingKernel,
        WedgeKernel,
        "ScaleKernel",
        WassersteinKernel,
        ExactWassersteinKernel,
    ] = Field(description="The kernel whose output is scaled.")
    # the ScaleKernel mapper forwards the dimensionality d to the outputscale prior, so
    # dimensionality-scaled priors are supported here.
    outputscale_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the output scale $\\theta$, which sets the variance of "
        "the response. If not provided, no prior is placed on it and it is fitted from "
        "the data alone.",
    )
    outputscale_constraint: Optional[AnyPriorConstraint] = Field(
        default=None,
        description="Bounds the output scale is reparametrized into, making values "
        "outside them unreachable during fitting, unlike a prior which only shifts the "
        "fitted value.",
    )


class PolynomialFeatureInteractionKernel(AggregationKernel):
    """
    This kernel efficiently computes degree-n interactions between different
    kernels, possibly including self-interactions. This is most useful when
    there are different kernels for different feature types (e.g. continuous,
    and categorical) and we want to compute interactions between them.

    For example, given three input kernels k1, k2, and k3, this kernel with
    `max_degree=2` and `include_self_interactions=True` would be equivalent
    to the following kernel, but much faster to compute:

    ```
    k = AdditiveKernel(kernels=[
        # constant (degree-0)
        ConstantKernel(),

        # individual kernels (degree-1)
        ScaleKernel(base_kernel=k1),
        ScaleKernel(base_kernel=k2),
        ScaleKernel(base_kernel=k3),

        # interactions (degree-2)
        ScaleKernel(base_kernel=MultiplicativeKernel(kernels=[k1, k2])),
        ScaleKernel(base_kernel=MultiplicativeKernel(kernels=[k1, k3])),
        ScaleKernel(base_kernel=MultiplicativeKernel(kernels=[k2, k3])),

        # self-interactions (degree-2)
        ScaleKernel(base_kernel=MultiplicativeKernel(kernels=[k1, k1])),
        ScaleKernel(base_kernel=MultiplicativeKernel(kernels=[k2, k2])),
        ScaleKernel(base_kernel=MultiplicativeKernel(kernels=[k3, k3])),
    ])
    ```

    """

    type: Literal["PolynomialFeatureInteractionKernel"] = (
        "PolynomialFeatureInteractionKernel"
    )
    kernels: Sequence[
        Union[
            AdditiveKernel,
            MultiplicativeKernel,
            ScaleKernel,
            HammingDistanceKernel,
            LinearKernel,
            PolynomialKernel,
            MaternKernel,
            RBFKernel,
            SphericalLinearKernel,
            TanimotoKernel,
            InfiniteWidthBNNKernel,
            WassersteinKernel,
            ExactWassersteinKernel,
        ]
    ] = Field(
        description="The kernels whose interactions are computed. Typically one per "
        "kind of input, so that the interactions run across the kinds.",
    )
    max_degree: int = Field(
        description="Highest interaction order computed. 1 keeps the kernels "
        "independent; 2 adds every pairwise interaction, and so on. The number of terms "
        "grows quickly with this.",
    )
    include_self_interactions: bool = Field(
        description="Whether a kernel may interact with itself, which adds the "
        "quadratic and higher powers of each kernel alongside the cross terms.",
    )
    outputscale_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the output scale applied to each interaction term "
        "before the terms are summed. If not provided, no prior is placed on them and "
        "they are fitted from the data alone.",
    )


AdditiveKernel.model_rebuild()
MultiplicativeKernel.model_rebuild()
