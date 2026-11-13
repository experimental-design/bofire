from typing import Type

from pydantic import field_validator

from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
    Objective,
    TargetObjective,
)
from bofire.data_models.strategies.convergence_criteria.api import ConvergenceCriterion
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy


class MultiobjectiveStrategy(BotorchStrategy):
    """Bayesian optimization of several objectives at once, searching for the best
    trade-offs between them (the Pareto front).

    At least two outputs need a maximize, minimize or close-to-target objective.
    Outputs with a sigmoid or target objective act as constraints.
    """

    @field_validator("domain")
    @classmethod
    def validate_domain_is_multiobjective(cls, v):
        """Validate that the domain is multiobjective."""
        feats = v.outputs.get_by_objective(
            [MaximizeObjective, MinimizeObjective, CloseToTargetObjective],
        )
        if len(feats) < 2:
            raise ValueError(
                "At least two output features with MaximizeObjective, MinimizeObjective or CloseToTargetObjective have to be defined in the domain.",
            )
        return v

    @classmethod
    def is_criterion_implemented(cls, my_type: Type[ConvergenceCriterion]) -> bool:
        """Check if a convergence criterion type is applicable for the strategy.

        Multi-objective strategies accept a criterion if it declares itself
        applicable to multi-objective optimization.

        Args:
            my_type: ConvergenceCriterion class

        Returns:
            bool: True if the convergence criterion type is valid for the strategy chosen, False otherwise

        """
        return my_type.is_applicable_to_multiobjective()

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        """Method to check if a objective type is implemented for the strategy

        Args:
            my_type (Type[Objective]): Objective class

        Returns:
            bool: True if the objective type is valid for the strategy chosen, False otherwise

        """
        return my_type in [
            MaximizeObjective,
            MinimizeObjective,
            MinimizeSigmoidObjective,
            MaximizeSigmoidObjective,
            TargetObjective,
            CloseToTargetObjective,
        ]
