from typing import Literal, Optional, Sequence, Union

from pydantic import field_validator

from bofire.data_models.constraints.condition import Condition
from bofire.data_models.kernels.categorical import HammingDistanceKernel
from bofire.data_models.kernels.continuous import LinearKernel, MaternKernel, RBFKernel
from bofire.data_models.kernels.kernel import Kernel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint


class ConditionalEmbeddingKernel(Kernel):
    """A kernel that transforms inputs into an embedding space, to encode conditional
    dependence on other input features.

    By default, all features are passed to the base kernel. It is generally advised
    that indicator features - those that only exist to indicate whether another
    feature is active - not be included in `base_kernel.features`, since they
    will not provide any useful information beyond their role as an indicator. This
    avoids "double-dipping" these indicator features."""

    base_kernel: Union[
        RBFKernel,
        MaternKernel,
        LinearKernel,
        HammingDistanceKernel,
        # AdditiveKernel,
        # MultiplicativeKernel,
        # ScaleKernel,
    ]

    conditions: Sequence[tuple[str, str, Condition]]


class WedgeKernel(ConditionalEmbeddingKernel):
    type: Literal["WedgeKernel"] = "WedgeKernel"
    ard: bool = True
    lengthscale_prior: Optional[AnyPrior] = None
    lengthscale_constraint: Optional[AnyPriorConstraint] = None
    angle_prior: Optional[AnyPrior] = None
    radius_prior: Optional[AnyPrior] = None

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
