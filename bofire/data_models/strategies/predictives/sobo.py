from typing import List, Literal, Optional, Type

import pydantic
from pydantic import Field, field_validator, model_validator

from bofire.data_models.acquisition_functions.api import (
    AnySingleObjectiveAcquisitionFunction,
    qLogNEI,
    qLogPF,
)
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import ConstrainedObjective, Objective
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy


class SoboBaseStrategy(BotorchStrategy):
    acquisition_function: AnySingleObjectiveAcquisitionFunction = Field(
        default_factory=lambda: qLogNEI(),
    )

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
        return True


class SoboStrategy(SoboBaseStrategy):
    type: Literal["SoboStrategy"] = "SoboStrategy"

    @model_validator(mode="after")
    def validate_is_singleobjective(self):
        if (
            len(self.domain.outputs.get_by_objective(excludes=ConstrainedObjective))
            - len(
                self.domain.outputs.get_by_objective(includes=None, excludes=Objective)  # type: ignore
            )
        ) > 1 and not isinstance(self.acquisition_function, qLogPF):
            raise ValueError(
                "SOBO strategy can only deal with one no-constraint objective.",
            )
        if isinstance(self.acquisition_function, qLogPF):
            if len(self.domain.outputs.get_by_objective(ConstrainedObjective)) == 0:
                raise ValueError(
                    "At least one constrained objective is required for qLogPF.",
                )
        return self


class _ForbidPFMixin:
    """
    Mixin to forbid the use of qLogPF acquisition function in single-objective strategies
    that are not the SoboStrategy.
    """

    @field_validator("acquisition_function")
    def validate_acquisition_function(cls, acquisition_function):
        if isinstance(acquisition_function, qLogPF):
            raise ValueError(
                "qLogPF acquisition function is only allowed in the ´SoboStrategy´.",
            )
        return acquisition_function


class AdditiveSoboStrategy(SoboBaseStrategy, _ForbidPFMixin):
    type: Literal["AdditiveSoboStrategy"] = "AdditiveSoboStrategy"

    use_output_constraints: bool = True

    @field_validator("domain")
    def validate_is_multiobjective(cls, v, info):
        if (len(v.outputs.get_by_objective(Objective))) < 2:
            raise ValueError(
                "Additive SOBO strategy requires at least 2 outputs with objectives. Consider SOBO strategy instead.",
            )
        return v


class _CheckAdaptableWeightsMixin:
    """
    Contains an additional validator for weights in multiplicative objective merging.

    Additional validation of weights for adaptable weights, in multiplicative calculations. Adaption to (1, inf)
    requires w>=1e-8
    """

    @model_validator(mode="after")
    def check_adaptable_weights(cls, self):
        for obj in self.domain.outputs.get_by_objective():
            if obj.objective.w < 1e-8:
                raise pydantic.ValidationError(
                    f"Weight transformation to (1, inf) requires w>=1e-8 . Violated by feature {obj.key}."
                )
        return self


class MultiplicativeSoboStrategy(
    SoboBaseStrategy, _CheckAdaptableWeightsMixin, _ForbidPFMixin
):
    type: Literal["MultiplicativeSoboStrategy"] = "MultiplicativeSoboStrategy"

    @field_validator("domain")
    def validate_is_multiobjective(cls, v, info):
        if (len(v.outputs.get_by_objective(Objective))) < 2:
            raise ValueError(
                "Multiplicative SOBO strategy requires at least 2 outputs with objectives. Consider SOBO strategy instead.",
            )
        return v


class MultiplicativeAdditiveSoboStrategy(
    SoboBaseStrategy, _CheckAdaptableWeightsMixin, _ForbidPFMixin
):
    """
    Mixed, weighted multiplicative (primary, strict) and additive (secondary, non-strict) objectives.

    The formular for a mixed objective with two multiplicative features (f1, and f2 with weights w1 and w2) and two
    additive features (f3 and f4 with weights w3 and w4) is:

        additive_objective = 1 + f3*w3 + f4*w4

        objective = f1^w1 * f2^w2 * additive_objective

    """

    type: Literal["MultiplicativeAdditiveSoboStrategy"] = (
        "MultiplicativeAdditiveSoboStrategy"
    )

    use_output_constraints: bool = True
    additive_features: List[str] = Field(default_factory=list)

    @field_validator("additive_features")
    def validate_additive_features(cls, v, values):
        domain = values.data["domain"]
        for feature in v:
            if feature not in domain.outputs.get_keys():
                raise ValueError(
                    f"Feature {feature} is not an output feature of the domain."
                )
        return v


class CustomSoboStrategy(SoboBaseStrategy, _ForbidPFMixin):
    type: Literal["CustomSoboStrategy"] = "CustomSoboStrategy"

    use_output_constraints: bool = True
    dump: Optional[str] = None
