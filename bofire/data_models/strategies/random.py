from typing import Annotated, Literal, Optional, Type

from pydantic import Field

from bofire.data_models.constraints.api import (
    CategoricalExcludeConstraint,
    Constraint,
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearInequalityConstraint,
    ProductInequalityConstraint,
)
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.strategy import Strategy


class RandomStrategy(Strategy):
    type: Literal["RandomStrategy"] = "RandomStrategy"
    fallback_sampling_method: SamplingMethodEnum = SamplingMethodEnum.UNIFORM
    n_burnin: Annotated[int, Field(ge=1)] = 1000
    n_thinning: Annotated[int, Field(ge=1)] = 32
    num_base_samples: Optional[Annotated[int, Field(gt=0)]] = None
    max_iters: Annotated[int, Field(gt=0)] = 1000

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        return my_type in [
            LinearInequalityConstraint,
            LinearEqualityConstraint,
            NChooseKConstraint,
            InterpointEqualityConstraint,
            NonlinearInequalityConstraint,
            ProductInequalityConstraint,
            CategoricalExcludeConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return True
