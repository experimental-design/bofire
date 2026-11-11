from abc import abstractmethod
from numbers import Real
from typing import Any, Dict, Literal, Type

from pydantic import Field, ValidationInfo, field_validator, model_validator

from bofire.data_models.acquisition_functions.api import (
    AnyMultiObjectiveAcquisitionFunction,
    qLogNEHVI,
)
from bofire.data_models.base import BaseModel
from bofire.data_models.features.api import CategoricalOutput, Feature
from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MaximizeObjective,
    MinimizeObjective,
)
from bofire.data_models.strategies.predictives.multiobjective import (
    MultiobjectiveStrategy,
)
from bofire.data_models.unions import tagged_union


class ReferenceValue(BaseModel):
    """Reference value of one objective, the value an objective has to beat for a
    candidate to add to the hypervolume.

    Moving reference values follow the experiments: they are computed from the best
    and the worst value of the objective seen so far, so they shift as the
    optimization proceeds.
    """

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
    """Reference value that stays fixed, in units of the output.

    Examples:
        >>> FixedReferenceValue(value=0.5)
    """

    type: Literal["FixedReferenceValue"] = "FixedReferenceValue"
    value: float = Field(description="The reference value, in units of the output.")

    def get_reference_value(self, best: float, worst: float) -> float:
        """Get the reference value.

        Returns:
            float: The reference value.
        """
        return self.value


class AbsoluteMovingReferenceValue(ReferenceValue):
    """Reference value at a fixed distance from the best or the worst value seen so
    far.

    The reference value is `best + offset`, or `worst + offset` if not
    `orient_at_best`. A positive offset moves it towards better values, a negative
    one towards worse values.

    Examples:
        >>> AbsoluteMovingReferenceValue(orient_at_best=False, offset=0.0)
    """

    type: Literal["AbsoluteMovingReferenceValue"] = "AbsoluteMovingReferenceValue"
    orient_at_best: bool = Field(
        default=True,
        description="Whether the reference value is placed relative to the best value "
        "seen so far, or else relative to the worst one.",
    )
    offset: float = Field(
        description="Distance from the best or worst value, in units of the output; "
        "positive towards better values.",
    )

    def get_reference_value(self, best: float, worst: float) -> float:
        if self.orient_at_best:
            return best + self.offset
        return worst + self.offset


class RelativeMovingReferenceValue(ReferenceValue):
    """Reference value at a distance from the best or the worst value seen so far
    that is a multiple of the range between them.

    The reference value is `best + scaling * (best - worst)`, or
    `worst + scaling * (best - worst)` if not `orient_at_best`. Relative to the worst
    value, a scaling between 0 and 1 places it within the range seen so far, a
    negative one beyond the worst value.

    Examples:
        >>> RelativeMovingReferenceValue(orient_at_best=False, scaling=-0.1)
    """

    type: Literal["RelativeMovingReferenceValue"] = "RelativeMovingReferenceValue"
    orient_at_best: bool = Field(
        default=True,
        description="Whether the reference value is placed relative to the best value "
        "seen so far, or else relative to the worst one.",
    )
    scaling: float = Field(
        default=1.0,
        description="Distance from the best or worst value, as a multiple of the "
        "range between them; positive towards better values.",
    )

    def get_reference_value(self, best: float, worst: float) -> float:
        if self.orient_at_best:
            return best + self.scaling * (best - worst)
        return worst + self.scaling * (best - worst)


class RelativeToMaxMovingReferenceValue(ReferenceValue):
    """Reference value at a multiple of the best or the worst value seen so far.

    The reference value is `best * (1 + scaling)`, or `worst * (1 + scaling)` if not
    `orient_at_best`. The direction depends on the sign of the value: a negative
    scaling moves a positive value towards zero, but a negative value away from it.
    For a minimized objective, the values count as negated.

    Examples:
        >>> RelativeToMaxMovingReferenceValue(scaling=-0.3)
    """

    type: Literal["RelativeToMaxMovingReferenceValue"] = (
        "RelativeToMaxMovingReferenceValue"
    )
    orient_at_best: bool = Field(
        default=True,
        description="Whether the reference value is placed relative to the best value "
        "seen so far, or else relative to the worst one.",
    )
    scaling: float = Field(
        default=1.0,
        description="Relative change of the best or worst value, e.g. -0.3 for 70% "
        "of it.",
    )

    def get_reference_value(self, best: float, worst: float) -> float:
        if self.orient_at_best:
            return best + best * self.scaling
        return worst + worst * self.scaling


AnyReferenceValue = tagged_union(
    FixedReferenceValue,
    AbsoluteMovingReferenceValue,
    RelativeMovingReferenceValue,
    RelativeToMaxMovingReferenceValue,
)


class ExplicitReferencePoint(BaseModel):
    """Reference point of a multi-objective optimization, given by one reference
    value per objective.

    The hypervolume is measured from the reference point, so only candidates that
    beat it in every objective count as improvements. A reference point a little
    worse than the region of interest focuses the search there.

    Examples:
        >>> ExplicitReferencePoint(
        ...     values={
        ...         "yield": FixedReferenceValue(value=0.5),
        ...         "cost": AbsoluteMovingReferenceValue(orient_at_best=False, offset=0.0),
        ...     }
        ... )
    """

    type: Literal["ExplicitReferencePoint"] = "ExplicitReferencePoint"
    values: Dict[str, AnyReferenceValue] = Field(
        description="Reference value per key of an output with a maximize, minimize "
        "or close-to-target objective; every such output needs one.",
    )


class MoboStrategy(MultiobjectiveStrategy):
    """Multi-objective Bayesian optimization that maximizes the hypervolume improvement.

    The hypervolume is the region dominated by the Pareto front and bounded by a
    reference point; each batch is chosen to enlarge it the most. The cost grows
    quickly with the number of objectives; with many objectives or large batches,
    `QparegoStrategy` is cheaper.

    Examples:
        >>> MoboStrategy(domain=domain, ref_point={"yield": 0.5, "cost": 10.0})
    """

    type: Literal["MoboStrategy"] = "MoboStrategy"
    ref_point: ExplicitReferencePoint = Field(
        default=None,
        validate_default=True,
        description="Reference point of the hypervolume. A mapping from output keys "
        "to numbers is taken as fixed reference values. If not provided, the worst "
        "value of each objective seen so far.",
    )
    acquisition_function: AnyMultiObjectiveAcquisitionFunction = Field(
        default=qLogNEHVI(),
        description="Multi-objective acquisition function scoring the candidates by "
        "their improvement of the Pareto front.",
    )

    @field_validator("ref_point", mode="before")
    @classmethod
    def validate_ref_point(cls, ref_point: Any, info: ValidationInfo):
        """Builds the reference point from a mapping to numbers, or from the domain
        if none is given."""
        if ref_point is None:
            if "domain" not in info.data:
                return ref_point  # the domain is invalid, which is reported instead
            return ExplicitReferencePoint(
                values={
                    k: AbsoluteMovingReferenceValue(orient_at_best=False, offset=0.0)
                    for k in info.data["domain"].outputs.get_keys_by_objective(
                        [MaximizeObjective, MinimizeObjective, CloseToTargetObjective],
                    )
                }
            )
        if isinstance(ref_point, dict) and all(
            isinstance(v, Real) for v in ref_point.values()
        ):
            return ExplicitReferencePoint(
                values={k: FixedReferenceValue(value=v) for k, v in ref_point.items()}
            )
        return ref_point

    @model_validator(mode="after")
    def validate_ref_point_keys(self):
        """Validate that the reference point matches the domain."""
        keys = self.domain.outputs.get_keys_by_objective(
            [MaximizeObjective, MinimizeObjective, CloseToTargetObjective],
        )
        if sorted(keys) != sorted(self.ref_point.values.keys()):
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
        return my_type is not CategoricalOutput
