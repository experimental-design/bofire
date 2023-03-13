from abc import abstractmethod
from typing import Type

from pydantic import validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import Constraint
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import Feature, Output
from bofire.data_models.objectives.api import Objective
from bofire.strategies.validation import (
    validate_constraints,
    validate_features,
    validate_input_feature_count,
    validate_output_feature_count,
)


class Strategy(BaseModel):
    type: str
    domain: Domain

    @validator("domain")
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
            assert feature.objective is not None
            if not cls.is_objective_implemented(type(feature.objective)):
                raise ValueError(
                    f"Objective `{type(feature.objective)}` is not implemented for strategy `{cls.__name__}`"  # type: ignore
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
        pass

    _validate_constraints = validator("domain", allow_reuse=True)(validate_constraints)
    _validate_features = validator("domain", allow_reuse=True)(validate_features)
    _validate_input_feature_count = validator("domain", allow_reuse=True)(
        validate_input_feature_count
    )
    _validate_output_feature_count = validator("domain", allow_reuse=True)(
        validate_output_feature_count
    )

    @classmethod
    @abstractmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Abstract method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise
        """
        pass

    @classmethod
    @abstractmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """Abstract method to check if a specific feature type is implemented for the strategy

        Args:
            my_type (Type[Feature]): Feature class

        Returns:
            bool: True if the feature type is valid for the strategy chosen, False otherwise
        """
        pass
