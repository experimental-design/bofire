from collections.abc import Sequence
from typing import Dict, List, Literal, Optional, Union

from pydantic import Field, PositiveInt

from bofire.data_models.domain.api import Inputs
from bofire.data_models.encodings.api import AnyCategoricalEncoding, OrdinalEncoding
from bofire.data_models.feature_context import FeatureContext, task_input_key
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.kernels.categorical import (
    HammingDistanceKernel,
    IndexKernel,
    PositiveIndexKernel,
)
from bofire.data_models.kernels.conditional import WedgeKernel
from bofire.data_models.kernels.continuous import (
    AdditiveMapSaasKernel,
    InfiniteWidthBNNKernel,
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    RBFKernel,
    SphericalLinearKernel,
)
from bofire.data_models.kernels.fidelity import DownsamplingKernel
from bofire.data_models.kernels.kernel import AggregationKernel, Kernel
from bofire.data_models.kernels.molecular import TanimotoKernel
from bofire.data_models.kernels.shape import ExactWassersteinKernel, WassersteinKernel
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    AnyPrior,
    AnyPriorConstraint,
    GreaterThan,
)


class AdditiveKernel(AggregationKernel):
    r"""Sum of several kernels, $k(\mathbf x, \mathbf x') = \sum_i k_i(\mathbf x, \mathbf x')$."""

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
            AdditiveMapSaasKernel,
            WassersteinKernel,
            ExactWassersteinKernel,
            DownsamplingKernel,
            WedgeKernel,
            "AdditiveKernel",
            "MultiplicativeKernel",
            "ScaleKernel",
        ]
    ] = Field(description="The kernels to sum.")

    def children(self) -> List[Kernel]:
        return list(self.kernels)


class MultiplicativeKernel(AggregationKernel):
    r"""Product of several kernels, $k(\mathbf x, \mathbf x') = \prod_i k_i(\mathbf x, \mathbf x')$."""

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
            AdditiveMapSaasKernel,
            WassersteinKernel,
            ExactWassersteinKernel,
            DownsamplingKernel,
            WedgeKernel,
            "MultiplicativeKernel",
            "ScaleKernel",
        ]
    ] = Field(description="The kernels to multiply.")

    def children(self) -> List[Kernel]:
        return list(self.kernels)


class ScaleKernel(AggregationKernel):
    r"""Wraps another kernel with a fitted output scale,
    $k(\mathbf x, \mathbf x') = \theta\, k_{\text{base}}(\mathbf x, \mathbf x')$.

    The base kernel sets the shape of the covariance and this sets its magnitude, which
    is the variance of the noiseless signal.
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
        AdditiveMapSaasKernel,
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
        "the noiseless signal.",
    )
    outputscale_constraint: Optional[AnyPriorConstraint] = Field(
        default=None,
        description="Bounds the output scale $\\theta$ is restricted to during "
        "fitting.",
    )

    def children(self) -> List[Kernel]:
        return [self.base_kernel]


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
            AdditiveMapSaasKernel,
            InfiniteWidthBNNKernel,
            WassersteinKernel,
            ExactWassersteinKernel,
        ]
    ] = Field(
        description="The kernels whose interactions are computed.",
    )
    max_degree: int = Field(
        description="Highest interaction order computed. 1 keeps the kernels "
        "independent; 2 adds every pairwise interaction, and so on.",
    )
    include_self_interactions: bool = Field(
        description="Whether a kernel may interact with itself, adding the quadratic "
        "and higher powers of each kernel alongside the cross terms.",
    )
    outputscale_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the output scale applied to each interaction term "
        "before the terms are summed.",
    )

    def children(self) -> List[Kernel]:
        return list(self.kernels)


class ICMKernel(Kernel):
    r"""Kernel sharing information between tasks, the intrinsic coregionalization model.

    $$
    k((\mathbf x, t), (\mathbf x', t')) = k_{\text{base}}(\mathbf x, \mathbf x')\,
    B_{t t'}
    $$

    where $t$ is the task an observation belongs to and $B$ is a learned positive
    matrix of how strongly the tasks are correlated. Observations of one task then
    inform predictions for the others, in proportion to that correlation. The base
    kernel acts on every feature except the task input, and a task without
    observations is predicted from the prior.
    """

    type: Literal["ICMKernel"] = "ICMKernel"
    base_kernel: Union[
        RBFKernel,
        SphericalLinearKernel,
        MaternKernel,
        LinearKernel,
        HammingDistanceKernel,
        TanimotoKernel,
        AdditiveMapSaasKernel,
        WassersteinKernel,
        ExactWassersteinKernel,
        WedgeKernel,
        AdditiveKernel,
        MultiplicativeKernel,
        ScaleKernel,
    ] = Field(
        description="Kernel over the features other than the task input. It is "
        "offered every feature except the task input.",
    )
    rank: Optional[PositiveInt] = Field(
        default=None,
        description="Rank of the learned task correlation matrix, at most the number "
        "of tasks. A lower rank forces the tasks to share fewer patterns. If not "
        "provided, it is the number of tasks.",
    )
    task_feature: Optional[str] = Field(
        default=None,
        description="Key of the task input. If not provided, the single task input "
        "of the domain.",
    )

    def children(self) -> List[Kernel]:
        return [self.base_kernel]

    def encoding_requests(self, inputs: Inputs) -> Dict[str, AnyCategoricalEncoding]:
        requests = super().encoding_requests(inputs)
        key = task_input_key(inputs, self.task_feature)
        if key is not None:
            requests[key] = OrdinalEncoding()
        return requests

    def validate_inputs(self, context: FeatureContext) -> None:
        """Check the task input and the rank, then the base kernel without the task.

        Raises:
            ValueError: If there is no usable task input, if `rank` exceeds the
                number of tasks, or if the base kernel cannot work on its features.
        """
        task = context.task_feature(self.task_feature)
        if self.rank is not None and self.rank > len(task.categories):
            raise ValueError(
                f"ICMKernel has rank={self.rank}, but '{task.key}' has only "
                f"{len(task.categories)} tasks."
            )
        self.base_kernel.validate_inputs(context.without(task.key))


class MixedKernel(Kernel):
    r"""Kernel over a mix of continuous and categorical inputs.

    $$
    k = s_1\,(k_{\text{cont}} + s_2\,k_{\text{cat}}) + s_3\,(k_{\text{cont}} \cdot
    k_{\text{cat}})
    $$

    with $s_1$ to $s_3$ fitted output scales, so the model can express both an effect
    common to every category and one that differs between them. The categorical kernel
    acts on the categoricals encoded as ordinal codes, the continuous kernel on every
    other feature, engineered ones included. With no continuous features it reduces to
    $s_1\,k_{\text{cat}}$. Categoricals without descriptor data default to ordinal
    codes. A sub-kernel with explicit `features` uses those instead of its share.
    """

    type: Literal["MixedKernel"] = "MixedKernel"
    continuous_kernel: Union[
        RBFKernel,
        MaternKernel,
        LinearKernel,
        PolynomialKernel,
        SphericalLinearKernel,
        AdditiveMapSaasKernel,
        InfiniteWidthBNNKernel,
    ] = Field(
        default=RBFKernel(
            ard=True,
            lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
            lengthscale_constraint=GreaterThan(lower_bound=2.5e-2),
        ),
        description="Kernel over the continuous features, and the categoricals not "
        "encoded as ordinal codes.",
    )
    categorical_kernel: Union[
        HammingDistanceKernel,
        IndexKernel,
        PositiveIndexKernel,
    ] = Field(
        default=HammingDistanceKernel(
            ard=True, lengthscale_constraint=GreaterThan(lower_bound=1e-6)
        ),
        description="Kernel over the categoricals encoded as ordinal codes.",
    )

    def children(self) -> List[Kernel]:
        return [self.continuous_kernel, self.categorical_kernel]

    @staticmethod
    def categorical_share(context: FeatureContext) -> List[str]:
        """The offered features the categorical kernel acts on: ordinal-coded categoricals."""
        return [
            key
            for key in context.offered
            if isinstance(context.get(key), CategoricalInput)
            and isinstance(context.encoding(key), OrdinalEncoding)
        ]

    def encoding_requests(self, inputs: Inputs) -> Dict[str, AnyCategoricalEncoding]:
        requests = super().encoding_requests(inputs)
        for feat in inputs.get(CategoricalInput, exact=False):
            if feat.descriptors is None:
                requests[feat.key] = OrdinalEncoding()
        return requests

    def validate_inputs(self, context: FeatureContext) -> None:
        """Check that there is a categorical side, then each kernel against its share.

        Raises:
            ValueError: If no categorical is encoded as ordinal codes, or a kernel
                cannot work on its features.
        """
        categorical = self.categorical_share(context)
        if not categorical:
            raise ValueError(
                "MixedKernel needs at least one categorical input encoded as ordinal "
                "codes."
            )
        continuous = [k for k in context.offered if k not in categorical]
        self.continuous_kernel.validate_inputs(context.only(continuous))
        self.categorical_kernel.validate_inputs(context.only(categorical))


AdditiveKernel.model_rebuild()
MultiplicativeKernel.model_rebuild()
