from typing import Literal, Type

from pydantic import Field, model_validator

from bofire.data_models.acquisition_functions.api import (
    AnySingleObjectiveAcquisitionFunction,
    qLogNEI,
    qLogPF,
)
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import ConstrainedObjective, Objective
from bofire.data_models.strategies.convergence_criteria.api import ConvergenceCriterion
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy
from bofire.data_models.strategies.predictives.scalarization import (
    AnyScalarization,
    SingleObjectiveScalarization,
)


class SoboStrategy(BotorchStrategy):
    """Single-objective Bayesian optimization, of one objective or of several
    combined into one by a scalarization.

    By default exactly one output has an objective to optimize; further outputs may
    have constrained objectives, such as sigmoid ones, which act as constraints the
    candidates should satisfy. With `qLogPF` as acquisition function, the strategy
    only searches for candidates that satisfy those constraints.

    Examples:
        >>> SoboStrategy(domain=domain, acquisition_function=qLogEI())
        >>> SoboStrategy(domain=domain, scalarization=WeightedProductScalarization())
    """

    type: Literal["SoboStrategy"] = "SoboStrategy"
    acquisition_function: AnySingleObjectiveAcquisitionFunction = Field(
        default=qLogNEI(),
        description="Acquisition function that scores candidates by the predicted "
        "value of the objective.",
    )
    scalarization: AnyScalarization = Field(
        default=SingleObjectiveScalarization(),
        description="How the objectives are combined into the single value that is "
        "maximized. By default, the domain has a single objective to optimize.",
    )

    @model_validator(mode="after")
    def validate_scalarization(self):
        self.scalarization.validate_domain(self.domain)
        is_single = isinstance(self.scalarization, SingleObjectiveScalarization)
        if isinstance(self.acquisition_function, qLogPF):
            if not is_single:
                raise ValueError(
                    "qLogPF acquisition function is only allowed with the "
                    "SingleObjectiveScalarization.",
                )
            if len(self.domain.outputs.get_by_objective(ConstrainedObjective)) == 0:
                raise ValueError(
                    "At least one constrained objective is required for qLogPF.",
                )
        elif (
            is_single
            and (
                len(self.domain.outputs.get_by_objective(excludes=ConstrainedObjective))
                - len(
                    self.domain.outputs.get_by_objective(
                        includes=None, excludes=Objective
                    )
                )
            )
            > 1
        ):
            raise ValueError(
                "SOBO strategy can only deal with one no-constraint objective.",
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
