from abc import abstractmethod
from typing import Any, Dict, Literal, Optional, Type, Union

from pydantic import Field, model_validator

from bofire.data_models.acquisition_functions.api import (
    AnyMultiObjectiveAcquisitionFunction,
    qLogNEHVI,
)
from bofire.data_models.base import BaseModel
from bofire.data_models.features.api import CategoricalOutput, Feature
from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
    Objective,
    TargetObjective,
)
from bofire.data_models.strategies.predictives.multiobjective import (
    MultiobjectiveStrategy,
)


class ReferenceValue(BaseModel):
    type: Any

    @abstractmethod
    def get_reference_value(self, best: float, worst: float) -> float:
        """Get the reference value.

        Args:
            best: The best value that has been seen so far for the objective.
            worst: The worst value that has been seen so far for the objective.

        Returns:
            float: The reference value.
        """
        pass


class FixedReferenceValue(ReferenceValue):
    """Reference value that is fixed.

    Args:
        value: The fixed reference value.
    """

    type: Literal["FixedReferenceValue"] = "FixedReferenceValue"
    value: float

    def get_reference_value(self, best: float, worst: float) -> float:
        """Get the reference value.

        Returns:
            float: The reference value.
        """
        return self.value


class AbsoluteMovingReferenceValue(ReferenceValue):
    """Reference value that is changing over execution time of the strategy, where the change is
    parameterized in absolute values.

    Attributes:
        orient_at_best: If True, the reference value is oriented at the best value that has
            been seen so far for the objective. If False, the reference value is oriented
            at the worst value that has been seen so far for the objective. Have a look below
            on the exact formula.
        offset: The offset that is added to the reference value. In case of `orient_at_max==True`,
            it holds that `reference_value = best_value + offset`, else it holds that
            `reference_value = worst_value + offset`, , where best is the best value that has been
            seen so far for the objective and worst is the worst value that has been seen so far
            for the objective.
    """

    type: Literal["AbsoluteMovingReferenceValue"] = "AbsoluteMovingReferenceValue"
    orient_at_best: bool = True
    offset: float

    def get_reference_value(self, best: float, worst: float) -> float:
        if self.orient_at_best:
            return best + self.offset
        return worst + self.offset


class RelativeMovingReferenceValue(ReferenceValue):
    """Reference value that is changing over execution time of the strategy, where the change is
    parameterized in relative values.

    Attributes:
        orient_at_best: If True, the reference value is oriented at the best value that has
            been seen so far for the objective. If False, the reference value is oriented
            at the worst value that has been seen so far for the objective. Have a look below
            on the exact formula.
        scaling: The scaling that is applied to the reference value. In case of `orient_at_max==True`,
            it holds that `reference_value = best + scaling * (best-worst)`, else it holds that
            `reference_value = worst + scaling * (best - worst)`, where best is the best value that has been
            seen so far for the objective and worst is the worst value that has been seen so far for the
            objective.
    """

    type: Literal["RelativeMovingReferenceValue"] = "RelativeMovingReferenceValue"
    orient_at_best: bool = True
    scaling: float = 1.0

    def get_reference_value(self, best: float, worst: float) -> float:
        if self.orient_at_best:
            return best + self.scaling * (best - worst)
        return worst + self.scaling * (best - worst)


class RelativeToMaxMovingReferenceValue(ReferenceValue):
    """Reference value that is changing over execution time of the strategy, where the change is
    parameterized in relative values to the maximum without min/max scaling.

    Attributes:
        orient_at_best: If True, the reference value is oriented at the best value that has
            been seen so far for the objective. If False, the reference value is oriented
            at the worst value that has been seen so far for the objective. Have a look below
            on the exact formula.
        scaling: The scaling that is applied to the reference value. In case of `orient_at_max==True`,
            it holds that `reference_value = best * (1 + scaling)`, else it holds that
            `reference_value = worst * (1 + scaling)`, where best is the best value that has been
            seen so far for the objective and worst is the worst value that has been seen so far for the
            objective.

    Note:
        This reference value is not scaled by the min/max values of the objective.
        This means you need to be mindful that the values of your objective are positive or negative.
        i.e., scaling a positive number by e.g. scaling=-0.3 will result in a smaller positive number. While scaling a negative number
        by -0.3 will result in a smaller negative number (thus having the opposite effect).
    """

    type: Literal["RelativeToMaxMovingReferenceValue"] = (
        "RelativeToMaxMovingReferenceValue"
    )
    orient_at_best: bool = True
    scaling: float = 1.0

    def get_reference_value(self, best: float, worst: float) -> float:
        if self.orient_at_best:
            return best + best * self.scaling
        return worst + worst * self.scaling


class ReferencePoint(BaseModel):
    type: Any


class ExplicitReferencePoint(ReferencePoint):
    """Data model used to define the reference point (and how it is possibly inferred)
    in an explicit per feature based way.

    Attributes:
        values: The values of the reference point for each output feature.
    """

    type: Literal["ExplicitReferencePoint"] = "ExplicitReferencePoint"
    values: Dict[
        str,
        Union[
            FixedReferenceValue,
            AbsoluteMovingReferenceValue,
            RelativeMovingReferenceValue,
            RelativeToMaxMovingReferenceValue,
        ],
    ]


class MoboStrategy(MultiobjectiveStrategy):
    type: Literal["MoboStrategy"] = "MoboStrategy"  # type: ignore
    ref_point: Optional[Union[ExplicitReferencePoint, Dict[str, float]]] = None
    acquisition_function: AnyMultiObjectiveAcquisitionFunction = Field(
        default_factory=lambda: qLogNEHVI(),
    )

    @model_validator(mode="after")
    def validate_ref_point(self):
        """Validate that the provided refpoint matches the provided domain."""
        if self.ref_point is None:
            self.ref_point = ExplicitReferencePoint(
                values={
                    k: AbsoluteMovingReferenceValue(orient_at_best=False, offset=0.0)
                    for k in self.domain.outputs.get_keys_by_objective(
                        [
                            MaximizeObjective,
                            MinimizeObjective,
                            CloseToTargetObjective,
                        ],
                    )
                }
            )
        if isinstance(self.ref_point, dict):
            self.ref_point = ExplicitReferencePoint(
                values={
                    k: FixedReferenceValue(value=v) for k, v in self.ref_point.items()
                }
            )
        keys = self.domain.outputs.get_keys_by_objective(
            [MaximizeObjective, MinimizeObjective, CloseToTargetObjective],
        )
        ref_point_keys = self.ref_point.values.keys()
        if sorted(keys) != sorted(ref_point_keys):
            raise ValueError(
                f"Provided refpoint do not match the domain, expected keys: {keys}",
            )
        return self

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """Method to check if a specific feature type is implemented for the strategy

        Args:
            my_type (Type[Feature]): Feature class

        Returns:
            bool: True if the feature type is valid for the strategy chosen, False otherwise

        """
        if my_type not in [CategoricalOutput]:
            return True
        return False

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
