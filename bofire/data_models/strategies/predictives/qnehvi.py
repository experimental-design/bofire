from typing import Literal, Type

from pydantic import confloat

from bofire.data_models.objectives.api import (
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
    Objective,
    TargetObjective,
)
from bofire.data_models.strategies.predictives.qehvi import QehviStrategy


class QnehviStrategy(QehviStrategy):
    """
    QnehviStrategy class implements a variant of the Quality-Expected Hypervolume Improvement (QEHVI) strategy for Bayesian optimization.

    Attributes:
    type (Literal["QnehviStrategy"]): A literal indicating the strategy type as "QnehviStrategy".
    alpha (float): A confidence parameter used in the calculation of the acquisition function.

    Methods:
    is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
    Check if a given objective type is implemented for the strategy.
    """

    type: Literal["QnehviStrategy"] = "QnehviStrategy"
    alpha: confloat(ge=0, le=0.5) = 0.0  # type: ignore

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
        ]
