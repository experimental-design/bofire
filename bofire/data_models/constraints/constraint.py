from abc import abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.features import Inputs
from bofire.data_models.types import FeatureKeys


class Constraint(BaseModel):
    """Abstract base class to define constraints on the optimization space."""

    type: Any
    features: FeatureKeys

    @abstractmethod
    def is_fulfilled(
        self,
        experiments: pd.DataFrame,
        tol: Optional[float] = 1e-6,
    ) -> pd.Series:
        """Abstract method to check if a constraint is fulfilled for all the rows of the provided dataframe.

        Args:
            experiments (pd.DataFrame): Dataframe to check constraint fulfillment.
            tol (float, optional): tolerance parameter. A constraint is considered as not fulfilled if
                the violation is larger than tol. Defaults to 0.

        Returns:
            bool: True if fulfilled else False

        """

    @abstractmethod
    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        """Numerically evaluates the constraint.

        Args:
            experiments (pd.DataFrame): Dataframe to evaluate the constraint on.

        Returns:
            pd.Series: Distance to reach constraint fulfillment.

        """

    @abstractmethod
    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Numerically evaluates the jacobian of the constraint
        Args:
            experiments (pd.DataFrame): Dataframe to evaluate the constraint on.

        Returns:
            pd.DataFrame: the i-th row contains the jacobian evaluated at the i-th experiment

        """

    @abstractmethod
    def validate_inputs(self, inputs: Inputs):
        """Validates that the features stored in Inputs are compatible with the constraint.

        Args:
            inputs (Inputs): Inputs to validate.

        """


class IntrapointConstraint(Constraint):
    """An intrapoint constraint describes required relationships within a candidate
    when asking a strategy to return one or more candidates.
    """

    type: str


class EqualityConstraint(IntrapointConstraint):
    type: str

    def is_fulfilled(self, experiments: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
        return pd.Series(
            np.isclose(self(experiments), 0, atol=tol),
            index=experiments.index,
        )


class InequalityConstraint(IntrapointConstraint):
    type: str

    def is_fulfilled(self, experiments: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
        return self(experiments) <= 0 + tol


class ConstraintError(Exception):
    """Base Error for Constraints"""


class ConstraintNotFulfilledError(ConstraintError):
    """Raised when an constraint is not fulfilled."""
