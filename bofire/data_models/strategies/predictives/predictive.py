from abc import abstractmethod
from typing import Type

from pydantic import field_validator

from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import Output
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.convergence_criteria.api import (
    AnyConvergenceCriterion,
    ConvergenceCriterion,
)
from bofire.data_models.strategies.strategy import Strategy


class PredictiveStrategy(Strategy):
    convergence_criterion: AnyConvergenceCriterion | None = None

    @field_validator("domain")
    @classmethod
    def validate_objectives(cls, domain: Domain):
        """Validator to ensure that all objectives defined in the domain are valid for the chosen strategy

        Args:
            domain (Domain): The domain to be used in the strategy

        Raises:
            ValueError: if a objective type is defined in the domain but is invalid for the strategy chosen

        Returns:
            Domain: the domain

        """
        for feature in domain.outputs.get_by_objective(Objective):
            assert isinstance(feature, Output)
            if not cls.is_objective_implemented(type(feature.objective)):
                raise ValueError(
                    f"Objective `{type(feature.objective)}` is not implemented for strategy `{cls.__name__}`",
                )
        return domain

    @classmethod
    @abstractmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        """Abstract method to check if a objective type is implemented for the strategy

        Args:
            my_type (Type[Objective]): Objective class

        Returns:
            bool: True if the objective type is valid for the strategy chosen, False otherwise

        """

    @field_validator("domain")
    @classmethod
    def validate_output_feature_count(cls, domain: Domain):
        """Validator to ensure that at least one output feature with attached objective is defined.

        Args:
            domain (Domain): The domain to be used in the strategy

        Raises:
            ValueError: if no output feature is specified
            ValueError: if not output feature with an attached objective is specified

        Returns:
            Domain: the domain

        """
        if len(domain.outputs) == 0:
            raise ValueError("no output feature specified")
        if len(domain.outputs.get_by_objective(Objective)) == 0:
            raise ValueError("no output feature with objective specified")
        return domain

    @classmethod
    @abstractmethod
    def is_criterion_implemented(cls, my_type: Type[ConvergenceCriterion]) -> bool:
        """Abstract method to check if a criterion type is applicable for the strategy.

        The decision is driven by whether the strategy is single- or
        multi-objective: single-objective strategies accept criteria that are
        applicable to single-objective optimization, multi-objective strategies
        accept those applicable to multi-objective optimization, and strategies
        that support both accept either.

        Args:
            my_type: ConvergenceCriterion class

        Returns:
            bool: True if the convergence criterion type is valid for the strategy chosen, False otherwise

        """

    @field_validator("convergence_criterion")
    @classmethod
    def validate_convergence_criterion(
        cls, convergence_criterion: ConvergenceCriterion | None
    ):
        """Validator to ensure that the convergence criterion is implemented for the strategy.

        Args:
            convergence_criterion: The convergence criterion to be validated

        Raises:
            ValueError: if the convergence criterion is not implemented for the strategy

        Returns:
            the convergence criterion

        """
        if convergence_criterion is not None and not cls.is_criterion_implemented(
            type(convergence_criterion)
        ):
            raise ValueError(
                f"Convergence criterion `{type(convergence_criterion)}` "
                f"is not implemented for strategy `{cls.__name__}`",
            )
        return convergence_criterion
