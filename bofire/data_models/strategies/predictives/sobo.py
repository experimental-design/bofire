from typing import List, Literal, Optional, Type

from pydantic import Field, field_validator, model_validator

from bofire.data_models.acquisition_functions.api import (
    AnySingleObjectiveAcquisitionFunction,
    qLogNEI,
    qLogPF,
)
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import ConstrainedObjective, Objective
from bofire.data_models.strategies.convergence_criteria.api import ConvergenceCriterion
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy


class SoboBaseStrategy(BotorchStrategy):
    """Bayesian optimization of a single objective, or of several combined into one."""

    acquisition_function: AnySingleObjectiveAcquisitionFunction = Field(
        default=qLogNEI(),
        description="Acquisition function that scores candidates by the predicted "
        "value of the objective.",
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

    @classmethod
    def is_criterion_implemented(cls, my_type: Type[ConvergenceCriterion]) -> bool:
        """Method to check if a convergence criterion type is applicable for the strategy.

        SOBO strategies are single-objective, so they accept criteria that are
        applicable to single-objective optimization.

        Args:
            my_type: ConvergenceCriterion class

        Returns:
            bool: True if the convergence criterion type is valid for the strategy chosen, False otherwise

        """
        return my_type.is_applicable_to_singleobjective()


class SoboStrategy(SoboBaseStrategy):
    """Single-objective Bayesian optimization.

    Exactly one output has an objective to optimize; further outputs may have
    constrained objectives, such as sigmoid ones, which act as constraints the
    candidates should satisfy. With `qLogPF` as acquisition function, the strategy
    only searches for candidates that satisfy those constraints.

    Examples:
        >>> SoboStrategy(domain=domain, acquisition_function=qLogEI())
    """

    type: Literal["SoboStrategy"] = "SoboStrategy"

    @model_validator(mode="after")
    def validate_is_singleobjective(self):
        if (
            len(self.domain.outputs.get_by_objective(excludes=ConstrainedObjective))
            - len(
                self.domain.outputs.get_by_objective(includes=None, excludes=Objective)
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
    """Forbids qLogPF, which is only allowed in the `SoboStrategy`."""

    @field_validator("acquisition_function")
    @classmethod
    def validate_acquisition_function(cls, acquisition_function):
        if isinstance(acquisition_function, qLogPF):
            raise ValueError(
                "qLogPF acquisition function is only allowed in the ´SoboStrategy´.",
            )
        return acquisition_function


class _MultipleObjectivesMixin:
    """Requires at least two outputs with objectives."""

    @field_validator("domain")
    @classmethod
    def validate_multiple_objectives(cls, domain):
        if len(domain.outputs.get_by_objective(Objective)) < 2:
            raise ValueError(
                f"{cls.__name__} requires at least 2 outputs with objectives. "
                "Consider SoboStrategy instead.",
            )
        return domain


class _CheckAdaptableWeightsMixin:
    """Requires objective weights of at least 1e-8, as the weights are divided by
    the smallest one."""

    @model_validator(mode="after")
    def check_adaptable_weights(self):
        for (
            obj
        ) in self.domain.outputs.get_by_objective():  # ty: ignore[unresolved-attribute]
            if obj.objective.w < 1e-8:
                raise ValueError(
                    f"Weight transformation to (1, inf) requires w>=1e-8. Violated by feature {obj.key}."
                )
        return self


class _OutputConstraintsSoboStrategy(SoboBaseStrategy, _ForbidPFMixin):
    """Combines the objectives of several outputs into one, optionally keeping the
    constrained objectives apart as constraints."""

    use_output_constraints: bool = Field(
        default=True,
        description="Whether outputs with a constrained objective, such as a sigmoid "
        "one, act as constraints the candidates should satisfy. If not, they are "
        "combined with the other objectives like any objective.",
    )


class AdditiveSoboStrategy(_OutputConstraintsSoboStrategy, _MultipleObjectivesMixin):
    """Bayesian optimization of the weighted sum of several objectives.

    The objective values are summed, each weighted by its objective's `w`. A poor
    value of one objective can be made up for by good values of the others.

    Examples:
        >>> AdditiveSoboStrategy(domain=domain)
    """

    type: Literal["AdditiveSoboStrategy"] = "AdditiveSoboStrategy"


class MultiplicativeSoboStrategy(
    SoboBaseStrategy,
    _MultipleObjectivesMixin,
    _CheckAdaptableWeightsMixin,
    _ForbidPFMixin,
):
    """Bayesian optimization of the weighted product of several objectives.

    The objective values, each raised to the power of its objective's weight `w`, are
    multiplied; the weights are rescaled so that the smallest is 1. A candidate is only good if all
    objectives are good, since a value near zero in one of them makes the product
    small. Constrained objectives, such as sigmoid ones, are factors like any other.

    Examples:
        >>> MultiplicativeSoboStrategy(domain=domain)
    """

    type: Literal["MultiplicativeSoboStrategy"] = "MultiplicativeSoboStrategy"


class MultiplicativeAdditiveSoboStrategy(
    _OutputConstraintsSoboStrategy,
    _MultipleObjectivesMixin,
    _CheckAdaptableWeightsMixin,
):
    """Bayesian optimization of several objectives, combined partly by product and
    partly by sum.

    The objectives in `additive_features` are summed, the others multiplied, each
    weighted by its objective's `w`, rescaled so that the smallest weight is 1. With
    multiplicative objectives f1 and f2 and additive objectives f3 and f4, the
    combined objective is

        f1^w1 * f2^w2 * (1 + w3 * f3 + w4 * f4)

    so the multiplicative objectives all have to be good, while the additive ones
    can make up for each other.

    Examples:
        >>> MultiplicativeAdditiveSoboStrategy(domain=domain, additive_features=["y3"])
    """

    type: Literal["MultiplicativeAdditiveSoboStrategy"] = (
        "MultiplicativeAdditiveSoboStrategy"
    )

    additive_features: List[str] = Field(
        default=[],
        description="Keys of the outputs whose objectives are summed; all other "
        "objectives are multiplied.",
    )

    @field_validator("additive_features")
    def validate_additive_features(cls, v, values):
        domain = values.data["domain"]
        for feature in v:
            if feature not in domain.outputs.get_keys():
                raise ValueError(
                    f"Feature {feature} is not an output feature of the domain."
                )
        return v


class CustomSoboStrategy(_OutputConstraintsSoboStrategy):
    """Bayesian optimization of several objectives combined by a function of your own.

    The function is assigned to the strategy's `f`, not to this data model, and
    receives the sampled output values, the objective functions and their weights,
    and the candidates; it returns one value per sample to maximize.
    """

    type: Literal["CustomSoboStrategy"] = "CustomSoboStrategy"

    dump: Optional[str] = Field(
        default=None,
        description="The function combining the objectives, as returned by the "
        "strategy's `dumps()`. If not provided, the function has to be set on the "
        "strategy before it can propose candidates.",
    )
