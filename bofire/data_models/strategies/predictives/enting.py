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
    solver_params: Dict[str, Any] = {}
    learn_from_candidates_coeff: float = 10.0

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

    def dump_enting_params(self) -> dict:
        """Dump the model in the nested structure required for ENTMOOT.

        Returns:
            dict: the nested dictionary of entmoot params.
        """
        return {
            "unc_params": {
                "beta": self.beta,
                "bound_coeff": self.bound_coeff,
                "acq_sense": self.acq_sense,
                "dist_trafo": self.dist_trafo,
                "dist_metric": self.dist_metric,
                "cat_metric": self.cat_metric,
            },
            "tree_train_params": {
                "train_params": {
                    "num_boost_round": self.num_boost_round,
                    "max_depth": self.max_depth,
                    "min_data_in_leaf": self.min_data_in_leaf,
                    "min_data_per_group": self.min_data_per_group,
                    "verbose": self.verbose,
                },
            },
        }

    def dump_solver_params(self) -> dict:
        """Dump the solver parameters for pyomo.

        Returns:
            dict: the nested dictionary of solver params.
        """
        return {
            "solver_name": self.solver_name,
            "verbose": self.solver_verbose,
            **self.solver_params,
        }
