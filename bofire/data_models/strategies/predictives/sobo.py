from typing import Literal, Optional, Type

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
from bofire.data_models.strategies.predictives.scalarization import AnyScalarization


class SoboStrategy(BotorchStrategy):
    """Single-objective Bayesian optimization, of one objective or of several
    combined into one by a scalarization.

    Without a scalarization, exactly one output has an objective to optimize;
    further outputs may have constrained objectives, such as sigmoid ones, which act
    as constraints the candidates should satisfy. With `qLogPF` as acquisition
    function, the strategy ignores the objectives and only searches for candidates
    that satisfy those constraints.

    Examples:
        >>> SoboStrategy(domain=domain, acquisition_function=qLogEI())
        >>> SoboStrategy(domain=domain, scalarization=WeightedScalarization())
    """

    type: Literal["SoboStrategy"] = "SoboStrategy"
    acquisition_function: AnySingleObjectiveAcquisitionFunction = Field(
        default=qLogNEI(),
        description="Acquisition function that scores candidates by the predicted "
        "value of the objective.",
    )
    scalarization: Optional[AnyScalarization] = Field(
        default=None,
        description="How the objectives of several outputs are combined into the "
        "single value that is maximized. If not provided, the domain has exactly one "
        "objective that is not a constrained one.",
    )

    @model_validator(mode="after")
    def validate_scalarization(self):
        if self.scalarization is None:
            if (
                len(self.domain.outputs.get_by_objective(excludes=ConstrainedObjective))
                > 1
            ):
                raise ValueError(
                    "SoboStrategy without a scalarization can only optimize one "
                    "objective that is not a constrained one. Combine several with "
                    "a scalarization, e.g. WeightedScalarization.",
                )
        else:
            self.scalarization.validate_domain(self.domain)
        if isinstance(self.acquisition_function, qLogPF):
            if (
                self.scalarization is not None
                and not self.scalarization.separates_output_constraints()
            ):
                raise ValueError(
                    "qLogPF acquisition function needs a scalarization that keeps "
                    "constrained objectives apart as constraints.",
                )
            if len(self.domain.outputs.get_by_objective(ConstrainedObjective)) == 0:
                raise ValueError(
                    "At least one constrained objective is required for qLogPF.",
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
