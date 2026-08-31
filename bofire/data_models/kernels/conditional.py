from typing import Literal, Optional, Sequence, Union

from pydantic import Field, field_validator

from bofire.data_models.constraints.condition import Condition
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

    Example:
        >>> # Feature that is conditional on another (indicator) feature.
        >>> # eg. only include catalyst concentration if catalyst != None
        >>> inter_dependent_condition = (
        >>>     "catalyst_concentration", "catalyst", SelectionCondition(selection=["Pt", "Pd"])
        >>> )
        >>> # Feature that depends on itself taking certain values
        >>> self_dependent_condition = (
        >>>     "acid_concentration", "acid_concentration", NonZeroCondition()
        >>> )
        >>> conditions = [inter_dependent_condition, self_dependent_condition]
        >>> conditional_kernel = ConditionalEmebeddingKernel(
        >>>     base_kernel=LinearKernel(),
        >>>     conditions=conditions
        >>> )
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
        "settings are ignored, so leave them unset and configure the lengthscale on "
        "this kernel instead.",
    )

    conditions: Sequence[tuple[str, str, Condition]] = Field(
        description="Which feature is active under which circumstances, as triples of "
        "the dependent feature, the feature it depends on, and the condition that must "
        "hold. A feature may depend on itself, which expresses that it is only "
        "meaningful at certain of its own values.",
    )


class WedgeKernel(ARDKernel, LengthscaleKernel, ConditionalEmbeddingKernel):
    """Conditional kernel embedding each input into a wedge-shaped space.

    Two candidates that both switch a feature off are treated as alike regardless of
    what value that inactive feature nominally holds, which is what stops a
    conditionally irrelevant input from driving the similarity.
    """

    type: Literal["WedgeKernel"] = "WedgeKernel"
    angle_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the wedge's opening angle, which sets how different an "
        "active candidate is taken to be from an inactive one.",
    )
    radius_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the wedge's radius, which sets how much the value of "
        "an active feature matters relative to whether it is active at all.",
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
