from pydantic import field_validator

from bofire.data_models.kernels.kernel import FeatureSpecificKernel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint


class FidelityKernel(FeatureSpecificKernel):
    """A kernel that encodes a continuous task feature, representing different fidelities."""

    pass


class DownsamplingKernel(FidelityKernel):
    offset_prior: AnyPrior | None = None
    offset_constraint: AnyPriorConstraint | None = None
    power_prior: AnyPrior | None = None
    power_constraint: AnyPriorConstraint | None = None

    @field_validator("features", mode="after")
    @classmethod
    def validate_one_task_feature(cls, features: list[str] | None) -> list[str]:
        if features is None or len(features) != 1:
            raise ValueError(
                f"{cls.__name__} requires a single task feature to be provided."
            )

        return features
