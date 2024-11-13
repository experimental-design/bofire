from pydantic import field_validator

from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MaximizeObjective,
    MinimizeObjective,
)
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
