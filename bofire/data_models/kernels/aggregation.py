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
    """Sum of several kernels.

    Two candidates count as similar if they are similar under any one of the summed
    kernels, which suits a response made of separable contributions.
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
    """Product of several kernels.

    Two candidates count as similar only if they are similar under every one of the
    multiplied kernels, which is how an interaction between input types is expressed.
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
    """Wraps another kernel with a fitted output scale.

    The base kernel decides which candidates are similar; this decides how much the
    response actually varies, which is why most kernels are wrapped in one before use.
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
        description="Prior over the output scale, which sets how far the response is "
        "expected to vary. If not provided, the surrogate's default is used.",
    )
    outputscale_constraint: Optional[AnyPriorConstraint] = Field(
        default=None,
        description="Hard bounds on the output scale, enforced during fitting rather "
        "than merely preferred as a prior is.",
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
        "before the terms are summed. If not provided, the surrogate's default is used.",
    )


AdditiveKernel.model_rebuild()
MultiplicativeKernel.model_rebuild()
