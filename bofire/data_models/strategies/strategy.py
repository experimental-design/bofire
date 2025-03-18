from abc import abstractmethod
from typing import Annotated, Optional, Type

from pydantic import Field, field_validator, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import Constraint
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import Feature


class Strategy(BaseModel):
    type: str
    domain: Domain
    seed: Optional[Annotated[int, Field(ge=0)]] = None

    @model_validator(mode="after")
    def validate_constraints(self):
        """Validator to ensure that all constraints defined in the domain are valid for the chosen strategy

        Raises:
            ValueError: if a constraint is defined in the domain but is invalid for the strategy chosen

        Returns:
            Domain: the domain

        """
        for constraint in self.domain.constraints:
            if not self.is_constraint_implemented(type(constraint)):
                raise ValueError(
                    f"constraint `{type(constraint)}` is not implemented for strategy `{type(self).__name__}`",
                )
        return self

    @field_validator("domain")
    @classmethod
    def validate_features(cls, domain: Domain):
        """Validator to ensure that all features defined in the domain are valid for the chosen strategy

        Args:
            domain (Domain): The domain to be used in the strategy

        Raises:
            ValueError: if a feature type is defined in the domain but is invalid for the strategy chosen

        Returns:
            Domain: the domain

        """
        for feature in domain.inputs + domain.outputs:
            if not cls.is_feature_implemented(type(feature)):
                raise ValueError(
                    f"feature `{type(feature)}` is not implemented for strategy `{cls.__name__}`",
                )
        return domain

    @field_validator("domain")
    @classmethod
    def validate_input_feature_count(cls, domain: Domain):
        """Validator to ensure that at least one input is defined.

        Args:
            domain (Domain): The domain to be used in the strategy

        Raises:
            ValueError: if no input feature is specified

        Returns:
            Domain: the domain

        """
        if len(domain.inputs) == 0:
            raise ValueError("no input feature specified")
        return domain

    @abstractmethod
    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
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
