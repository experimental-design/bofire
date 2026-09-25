from typing import List, Literal, Optional, Sequence, Union

from pydantic import Field, field_validator

from bofire.data_models.constraints.condition import Condition
from bofire.data_models.feature_context import FeatureContext
from bofire.data_models.kernels.categorical import HammingDistanceKernel
from bofire.data_models.kernels.continuous import (
    LinearKernel,
    MaternKernel,
    RBFKernel,
    SphericalLinearKernel,
)
from bofire.data_models.kernels.kernel import ARDKernel, Kernel, LengthscaleKernel
from bofire.data_models.priors.api import AnyPrior


class ConditionalEmbeddingKernel(Kernel):
    """A kernel that transforms inputs into an embedding space, to encode conditional
    dependence on other input features.

    By default, all features are passed to the base kernel. It is generally advised
    that indicator features - those that only exist to indicate whether another
    feature is active - not be included in `base_kernel.features`, since they
    will not provide any useful information beyond their role as an indicator. This
    avoids "double-dipping" these indicator features.

    Examples:
        A feature conditional on another, and one conditional on itself:

        >>> conditions = [
        ...     # only use the catalyst concentration if a catalyst is present
        ...     (
        ...         "catalyst_concentration",
        ...         "catalyst",
        ...         SelectionCondition(selection=["Pt", "Pd"]),
        ...     ),
        ...     # only use the acid concentration where it is non-zero
        ...     ("acid_concentration", "acid_concentration", NonZeroCondition()),
        ... ]
        >>> ConditionalEmbeddingKernel(
        ...     base_kernel=LinearKernel(), conditions=conditions
        ... )
    """

    base_kernel: Union[
        RBFKernel,
        SphericalLinearKernel,
        MaternKernel,
        LinearKernel,
        HammingDistanceKernel,
        # AdditiveKernel,
        # MultiplicativeKernel,
        # ScaleKernel,
    ] = Field(
        description="Kernel applied to the embedded inputs. Its own lengthscale "
        "settings are ignored; configure the lengthscale on this kernel instead.",
    )

    conditions: Sequence[tuple[str, str, Condition]] = Field(
        description="Which feature is active under which circumstances, as triples of "
        "the dependent feature, the feature it depends on, and the condition that must "
        "hold. A feature may depend on itself, which expresses that it is only relevant "
        "to the model under some conditions, e.g. if it is positive.",
    )

    def children(self) -> List[Kernel]:
        return [self.base_kernel]

    def validate_inputs(self, context: FeatureContext) -> None:
        """Check that every feature a condition names exists, then the base kernel.

        Raises:
            ValueError: If a condition names a key that is neither an input nor an
                engineered feature, or the base kernel cannot work on its features.
        """
        known = set(context.keys())
        named = {
            key
            for dependent, indicator, _ in self.conditions
            for key in (dependent, indicator)
        }
        if unknown := sorted(named - known):
            raise ValueError(
                f"{type(self).__name__} has conditions on {unknown}, which are "
                f"neither inputs nor engineered features."
            )
        super().validate_inputs(context)


class WedgeKernel(ARDKernel, LengthscaleKernel, ConditionalEmbeddingKernel):
    """Conditional kernel embedding each input into a wedge-shaped space.

    Two points that both leave a feature inactive have the same embedding for it,
    whatever value that inactive feature nominally holds. That is what stops a
    conditionally irrelevant dimension from contributing to the covariance.
    """

    type: Literal["WedgeKernel"] = "WedgeKernel"
    angle_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the wedge's opening angle, which sets the distance "
        "between the lower and upper bound of the feature, when active, in the "
        "embedded space.",
    )
    radius_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the wedge's radius, which sets the distance between "
        "the inactive point and the active points in the embedded space.",
    )

    @field_validator("base_kernel")
    @classmethod
    def validate_base_kernel(cls, base_kernel):
        lengthscale_attrs = ("lengthscale_prior", "lengthscale_constraint")
        for attr in lengthscale_attrs:
            if getattr(base_kernel, attr, None) is not None:
                raise ValueError(
                    f"When using a {cls.__name__}, the base_kernel must not have "
                    f"a {attr} provided, since this will be ignored."
                )
        return base_kernel
