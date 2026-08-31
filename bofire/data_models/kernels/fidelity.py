from typing import Any, Literal

from pydantic import Field, field_validator

from bofire.data_models.kernels.kernel import FeatureSpecificKernel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint


class FidelityKernel(FeatureSpecificKernel):
    """Abstract base class for kernels that encode a continuous task feature
    representing different fidelities.

    This class is intentionally not part of the ``AnyKernel`` union and cannot
    be instantiated directly. Use a concrete subclass such as
    :class:`DownsamplingKernel`. The base class exists only so that strategies
    can use ``isinstance(kernel, FidelityKernel)`` to identify fidelity kernels
    when validating surrogate specifications.
    """

    type: Any


class DownsamplingKernel(FidelityKernel):
    """Kernel encoding that a task becomes more like the target as fidelity rises.

    Rather than learning the relationship between fidelities freely, it assumes the
    approximation improves monotonically, which needs far less data than an
    unconstrained multi-task kernel.
    """

    type: Literal["DownsamplingKernel"] = "DownsamplingKernel"
    offset_prior: AnyPrior | None = Field(
        default=None,
        description="Prior over the offset, which sets how much the cheapest fidelity "
        "still tells you about the target. If not provided, the surrogate's default is "
        "used.",
    )
    offset_constraint: AnyPriorConstraint | None = Field(
        default=None,
        description="Hard bounds on the offset, enforced during fitting rather than "
        "merely preferred as a prior is.",
    )
    power_prior: AnyPrior | None = Field(
        default=None,
        description="Prior over the power, which sets how quickly the approximation "
        "improves as fidelity rises. If not provided, the surrogate's default is used.",
    )
    power_constraint: AnyPriorConstraint | None = Field(
        default=None,
        description="Hard bounds on the power, enforced during fitting rather than "
        "merely preferred as a prior is.",
    )

    @field_validator("features", mode="after")
    @classmethod
    def validate_one_task_feature(cls, features: list[str] | None) -> list[str]:
        if features is None or len(features) != 1:
            raise ValueError(
                f"{cls.__name__} requires a single task feature to be provided."
            )

        return features
