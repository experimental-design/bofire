from typing import Any, Dict, Literal, Type

from entmoot.models.model_params import EntingParams
from pydantic import Field

from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
)
from bofire.data_models.objectives.api import (
    MaximizeObjective,
    MinimizeObjective,
    Objective,
)
from bofire.data_models.strategies.predictives.predictive import PredictiveStrategy


class EntingStrategy(PredictiveStrategy):
    type: Literal["EntingStrategy"] = "EntingStrategy"
    enting_params: EntingParams = Field(default_factory=EntingParams)
    solver_params: Dict[str, Any] = {}
    learn_from_candidates_coeff: float = 10.0

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        return my_type in [
            LinearEqualityConstraint,
            LinearInequalityConstraint,
            NChooseKConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            CategoricalInput,
            DiscreteInput,
            CategoricalDescriptorInput,
            ContinuousInput,
            ContinuousOutput,
        ]

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return my_type in [MinimizeObjective, MaximizeObjective]
