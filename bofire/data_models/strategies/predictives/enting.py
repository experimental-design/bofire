from typing import Any, Dict, Literal, Type

from pydantic import PositiveFloat, PositiveInt

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

    # uncertainty model parameters
    beta: PositiveFloat = 1.96
    bound_coeff: PositiveFloat = 0.5
    acq_sense: Literal["exploration", "penalty"] = "exploration"
    dist_trafo: Literal["normal", "standard"] = "normal"
    dist_metric: Literal["euclidean_squared", "l1", "l2"] = "euclidean_squared"
    cat_metric: Literal["overlap", "of", "goodall4"] = "overlap"

    # lightgbm training hyperparameters
    # see https://lightgbm.readthedocs.io/en/latest/Parameters.html
    num_boost_round: PositiveInt = 100
    max_depth: PositiveInt = 3
    min_data_in_leaf: PositiveInt = 1
    min_data_per_group: PositiveInt = 1
    verbose: Literal[-1, 0, 1, 2] = -1

    # pyomo parameters
    solver_name: str = "gurobi"
    solver_verbose: bool = False
    solver_params: Dict[str, Any] = {}

    # kappa_fantasy determines a bound on the predicted value of an unseen point
    # used for making batch predictions, y* = mean + kappa_fantasy * std
    # for a both min and max problems, a positive value is 'pesimistic'
    # and a negative value is 'optimistic'
    # a value of zero implies future observations will be exactly the mean
    kappa_fantasy: float = 1.96

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
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
