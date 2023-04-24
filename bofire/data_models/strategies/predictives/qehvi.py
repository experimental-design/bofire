from typing import Dict, Literal, Optional, Type

from pydantic import validator

from bofire.data_models.constraints.api import Constraint, NChooseKConstraint
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import (
    MaximizeObjective,
    MinimizeObjective,
    Objective,
)
from bofire.data_models.strategies.predictives.multiobjective import (
    MultiobjectiveStrategy,
)


class QehviStrategy(MultiobjectiveStrategy):
    type: Literal["QehviStrategy"] = "QehviStrategy"

    ref_point: Optional[Dict[str, float]] = None

    @validator("ref_point")
    def validate_ref_point(cls, v, values):
        """Validate that the provided refpoint matches the provided domain."""
        if v is None:
            return v
        keys = values["domain"].outputs.get_keys_by_objective(
            [MaximizeObjective, MinimizeObjective]
        )
        if sorted(keys) != sorted(v.keys()):
            raise ValueError(
                f"Provided refpoint do not match the domain, expected keys: {keys}"
            )
        return v

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise
        """
        if my_type == NChooseKConstraint:
            return False
        return True

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """Method to check if a specific feature type is implemented for the strategy

        Args:
            my_type (Type[Feature]): Feature class

        Returns:
            bool: True if the feature type is valid for the strategy chosen, False otherwise
        """
        return True

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
        ]
