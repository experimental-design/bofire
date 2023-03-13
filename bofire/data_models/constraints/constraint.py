from abc import abstractmethod
from typing import Annotated, List

import pandas as pd
from pydantic import Field

from bofire.data_models.base import BaseModel


class Constraint(BaseModel):
    """Abstract base class to define constraints on the optimization space."""

    type: str

    @abstractmethod
    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        """Abstract method to check if a constraint is fulfilled for all the rows of the provided dataframe.

        Args:
            experiments (pd.DataFrame): Dataframe to check constraint fulfillment.

        Returns:
            bool: True if fulfilled else False
        """
        pass

    @abstractmethod
    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        """Numerically evaluates the constraint.

        Args:
            experiments (pd.DataFrame): Dataframe to evaluate the constraint on.

        Returns:
            pd.Series: Distance to reach constraint fulfillment.
        """
        pass


FeatureKeys = Annotated[List[str], Field(min_items=2)]
Coefficients = Annotated[List[float], Field(min_items=2)]
