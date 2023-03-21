from pydantic import validator

from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy


class MultiobjectiveStrategy(BotorchStrategy):
    """
    The MultiobjectiveStrategy class is a subclass of the BotorchStrategy class. This class checks that the domain is multiobjective by validating that it has at least two output features with either a MaximizeObjective or a MinimizeObjective objective type, and that all objectives have a weight of 1. If these conditions are not met, a ValueError is raised.

    Attributes:
    - domain: An instance of the Domain class, which specifies the search space for optimization.

    Methods:
    - validate_domain_is_multiobjective: A method that validates whether the domain is multiobjective by checking the number of output features with MaximizeObjective or MinimizeObjective objective types and their weights.
    """

    @validator("domain")
    def validate_domain_is_multiobjective(cls, v, values):
        """Validate that the domain is multiobjective."""
        feats = v.outputs.get_by_objective([MaximizeObjective, MinimizeObjective])
        if len(feats) < 2:
            raise ValueError(
                "At least two output features with MaximizeObjective or MinimizeObjective has to be defined in the domain."
            )
        for feat in feats:
            if feat.objective.w != 1.0:  # type: ignore
                raise ValueError(
                    f"Only objectives with weight 1 are supported. Violated by feature {feat.key}."
                )
        return v
