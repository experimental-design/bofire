from typing import Annotated, Dict, Literal, Type

import pandas as pd
from pydantic import Field, field_validator, model_validator

from bofire.data_models.constraints.api import (
    Constraint,
    ConstraintNotFulfilledError,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.domain import Domain
from bofire.data_models.features.api import ContinuousInput, Feature
from bofire.data_models.strategies.strategy import Strategy


def has_local_search_region(domain: Domain) -> bool:
    """Checks if the given domain has a local search region.

    Args:
        domain (Domain): The domain to check.

    Returns:
        bool: True if the domain has a local search region, False otherwise.

    """
    if len(domain.inputs.get(ContinuousInput)) == 0:
        return False
    is_lsr = False
    for feat in domain.inputs.get(ContinuousInput):
        assert isinstance(feat, ContinuousInput)
        if feat.local_relative_bounds is not None:
            is_lsr = True
    return is_lsr


class ShortestPathStrategy(Strategy):
    """Represents a strategy for finding the shortest path between two points.

    Attributes:
        type (Literal["ShortestPathStrategy"]): The type of the strategy.
        start (Annotated[Dict[str, Union[float, str]], Field(min_length=1)]): The starting point of the path.
        end (Annotated[Dict[str, Union[float, str]], Field(min_length=1)]): The ending point of the path.
        atol (Annotated[float, Field(gt=0)]): The absolute tolerance used for numerical comparisons.

    """

    type: Literal["ShortestPathStrategy"] = "ShortestPathStrategy"
    start: Annotated[Dict[str, float | str], Field(min_length=1)]
    end: Annotated[Dict[str, float | str], Field(min_length=1)]
    atol: Annotated[float, Field(gt=0)] = 1e-7

    @model_validator(mode="after")
    def validate_start_end(self):
        """Validates the start and end points of the path.

        Raises:
            ValueError: If the start or end point is not a valid candidate or if they are the same.

        """
        df_start = pd.DataFrame(pd.Series(self.start)).T
        df_end = pd.DataFrame(pd.Series(self.end)).T
        try:
            self.domain.validate_candidates(df_start, only_inputs=True)
        except (ValueError, ConstraintNotFulfilledError):
            raise ValueError("`start` is not a valid candidate.")
        try:
            self.domain.validate_candidates(df_end, only_inputs=True)
        except (ValueError, ConstraintNotFulfilledError):
            raise ValueError("`end` is not a valid candidate.")

        self.domain.inputs.validate_candidates(df_end)
        # check that start and end are not the same
        if df_start[self.domain.inputs.get_keys()].equals(
            df_end[self.domain.inputs.get_keys()],
        ):
            raise ValueError("`start` is equal to `end`.")
        return self

    @field_validator("domain")
    @classmethod
    def validate_lsr(cls, domain):
        """Validates the local search region of the domain.

        Args:
            domain: The domain to validate.

        Raises:
            ValueError: If the domain has no local search region.

        """
        if has_local_search_region(domain=domain) is False:
            raise ValueError("Domain has no local search region.")
        return domain

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        """Checks if a constraint is implemented. Currently only linear constraints are supported.

        Args:
            my_type (Type[Feature]): The type of the constraint.

        Returns:
            bool: True if the constraint is implemented, False otherwise.

        """
        return my_type in [LinearInequalityConstraint, LinearEqualityConstraint]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """Checks if a feature is implemented. Currently all features are supported.

        Args:
            my_type (Type[Feature]): The type of the feature.

        Returns:
            bool: True if the feature is implemented, False otherwise.

        """
        return True
