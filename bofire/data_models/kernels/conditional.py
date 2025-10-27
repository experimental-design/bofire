from typing import Literal, Optional, Union

from pydantic import field_validator

from bofire.data_models.constraints.categorical import Condition
from bofire.data_models.kernels.aggregation import (
    AdditiveKernel,
    MultiplicativeKernel,
    ScaleKernel,
)
from bofire.data_models.kernels.categorical import HammingDistanceKernel
from bofire.data_models.kernels.continuous import LinearKernel, MaternKernel, RBFKernel
from bofire.data_models.kernels.kernel import FeatureSpecificKernel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint


class WedgeKernel(FeatureSpecificKernel):
    type: Literal["WedgeKernel"] = "WedgeKernel"
    ard: bool = True
    lengthscale_prior: Optional[AnyPrior] = None
    lengthscale_constraint: Optional[AnyPriorConstraint] = None
    base_kernel: Union[
        RBFKernel,
        MaternKernel,
        LinearKernel,
        HammingDistanceKernel,
        AdditiveKernel,
        MultiplicativeKernel,
        ScaleKernel,
    ]
    conditions: list[tuple[str, Condition]]

    # Indicator features are used to determine whether the conditional features are
    # active or not. It is generally advised to remove these features from the base
    # kernel, to avoid "double-dipping" these features.
    drop_indicator_features_in_base_kernel: bool = True

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
