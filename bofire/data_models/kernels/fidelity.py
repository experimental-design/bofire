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
    r"""Kernel encoding that a task approaches the target as its fidelity rises.

    $$
    k(s, s') = c + (1 - s)^{1+\delta}(1 - s')^{1+\delta}
    $$

    where $s$ is the fidelity input, $c$ the offset and $\delta$ the power. Assuming the
    approximation improves monotonically with fidelity needs far less data than learning
    an unconstrained relationship between the tasks would.
    """

    type: Literal["DownsamplingKernel"] = "DownsamplingKernel"
    offset_prior: AnyPrior | None = Field(
        default=None,
        description="Prior over the offset $c$, which sets how much a low-fidelity "
        "observation still says about the target fidelity. If not provided, no prior is "
        "placed on it and it is fitted from the data alone.",
    )
    offset_constraint: AnyPriorConstraint | None = Field(
        default=None,
        description="Bounds the offset $c$ is restricted to during fitting.",
    )
    power_prior: AnyPrior | None = Field(
        default=None,
        description="Prior over the power $\\delta$, which sets how quickly the "
        "approximation improves as the fidelity rises. If not provided, no prior is "
        "placed on it and it is fitted from the data alone.",
    )
    power_constraint: AnyPriorConstraint | None = Field(
        default=None,
        description="Bounds the power $\\delta$ is restricted to during fitting.",
    )

    @field_validator("features", mode="after")
    @classmethod
    def validate_one_task_feature(cls, features: list[str] | None) -> list[str]:
        if features is None or len(features) != 1:
            raise ValueError(
                f"{cls.__name__} requires a single task feature to be provided."
            )

        return features
