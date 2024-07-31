from typing import Annotated, Literal, Tuple, Type, Union

from pydantic import Field, field_validator

from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.features.api import (
    CategoricalOutput,
    ContinuousInput,
    ContinuousOutput,
    Feature,
)
from bofire.data_models.strategies.strategy import Strategy


class SpaceFillingStrategy(Strategy):
    """Stratey that generates space filling samples by optimization in IPOPT.

    Attributes:
        domain (Domain): Domain defining the constrained input space
        sampling_fraction (float, optional): Fraction of sampled points to total points generated in
            the sampling process. Defaults to 0.3.
        ipopt_options (dict, optional): Dictionary containing options for the IPOPT solver. Defaults to {"maxiter":200, "disp"=0}.
    """

    type: Literal["SpaceFillingStrategy"] = "SpaceFillingStrategy"
    sampling_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.3
    ipopt_options: dict = {"maxiter": 200, "disp": 0}

    transform_range: Union[None, Tuple[float, float]] = None

    @field_validator("transform_range")
    @classmethod
    def validate_feature_range(cls, value):
        if value is not None:
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("feature_range must be a tuple of length 2")
            if value[0] >= value[1]:
                raise ValueError(
                    "feature_range[0] must be smaller than feature_range[1]"
                )
            return value

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        return my_type in [
            LinearEqualityConstraint,
            LinearInequalityConstraint,
            NonlinearInequalityConstraint,
            NonlinearEqualityConstraint,
            NChooseKConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [ContinuousInput, ContinuousOutput, CategoricalOutput]
