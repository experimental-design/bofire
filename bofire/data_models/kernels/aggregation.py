from collections.abc import Sequence
from typing import Literal, Optional, Union

from bofire.data_models.kernels.categorical import HammingDistanceKernel
from bofire.data_models.kernels.continuous import (
    InfiniteWidthBNNKernel,
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    RBFKernel,
)
from bofire.data_models.kernels.kernel import AggregationKernel
from bofire.data_models.kernels.molecular import TanimotoKernel
from bofire.data_models.kernels.shape import WassersteinKernel
from bofire.data_models.priors.api import AnyGeneralPrior, AnyPriorConstraint


class AdditiveKernel(AggregationKernel):
    type: Literal["AdditiveKernel"] = "AdditiveKernel"
    kernels: Sequence[
        Union[
            RBFKernel,
            MaternKernel,
            LinearKernel,
            HammingDistanceKernel,
            TanimotoKernel,
            "AdditiveKernel",
            "MultiplicativeKernel",
            "ScaleKernel",
        ]
    ]
    type: Literal["AdditiveKernel"] = "AdditiveKernel"


class MultiplicativeKernel(AggregationKernel):
    type: Literal["MultiplicativeKernel"] = "MultiplicativeKernel"
    kernels: Sequence[
        Union[
            RBFKernel,
            MaternKernel,
            LinearKernel,
            HammingDistanceKernel,
            AdditiveKernel,
            TanimotoKernel,
            "MultiplicativeKernel",
            "ScaleKernel",
        ]
    ]


class ScaleKernel(AggregationKernel):
    type: Literal["ScaleKernel"] = "ScaleKernel"
    base_kernel: Union[
        RBFKernel,
        MaternKernel,
        LinearKernel,
        HammingDistanceKernel,
        AdditiveKernel,
        MultiplicativeKernel,
        TanimotoKernel,
        "ScaleKernel",
        WassersteinKernel,
    ]
    outputscale_prior: Optional[AnyGeneralPrior] = None
    outputscale_constraint: Optional[AnyPriorConstraint] = None


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

    Attributes:
        kernels: The base kernels that should interact.
        max_degree: Maximum degree of interactions computed.
        include_self_interactions: Whether a kernel is allowed to interact with itself.
        outputscale_prior: The prior used to scale each interaction term before summing.
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
            TanimotoKernel,
            InfiniteWidthBNNKernel,
            WassersteinKernel,
        ]
    ]
    max_degree: int
    include_self_interactions: bool
    outputscale_prior: Optional[AnyGeneralPrior] = None


AdditiveKernel.model_rebuild()
MultiplicativeKernel.model_rebuild()
