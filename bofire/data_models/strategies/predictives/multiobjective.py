from typing import Type

from pydantic import field_validator

from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MaximizeObjective,
    MinimizeObjective,
)
from bofire.data_models.strategies.convergence_criteria.api import ConvergenceCriterion
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy


class MultiobjectiveStrategy(BotorchStrategy):
    @field_validator("domain")
    @classmethod
    def validate_domain_is_multiobjective(cls, v):
        """Validate that the domain is multiobjective."""
        feats = v.outputs.get_by_objective(
            [MaximizeObjective, MinimizeObjective, CloseToTargetObjective],
        )
        if len(feats) < 2:
            raise ValueError(
                "At least two output features with MaximizeObjective or MinimizeObjective has to be defined in the domain.",
            )
        for feat in feats:
            if feat.objective.w != 1.0:
                raise ValueError(
                    f"Only objectives with weight 1 are supported. Violated by feature {feat.key}.",
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
