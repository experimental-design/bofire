from abc import abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.features import Inputs
from bofire.data_models.types import FeatureKeys


class Constraint(BaseModel):
    """Abstract base class to define constraints on the optimization space."""

    type: Any
    features: FeatureKeys = Field(
        description="Keys of the input features the constraint acts on. The order is "
        "significant for subclasses that pair each feature with a per-feature value, "
        "such as the coefficients of a LinearConstraint or the exponents of a "
        "ProductConstraint.",
    )
    context: Optional[str] = Field(
        default=None,
        description="Free-text context providing additional information about the "
        "constraint, such as why it exists. Useful for agentic optimization where an "
        "LLM agent can leverage this description to better understand the constraint.",
    )

    @abstractmethod
    def to_description(self) -> str:
        """Return a human-readable description of this constraint."""

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

    type: Any


class EqualityConstraint(IntrapointConstraint):
    """Abstract base class for constraints fulfilled at a constraint value of zero.

    What is evaluated is defined by the `__call__` of the implementing subclass. This
    class only fixes how that value is interpreted: a candidate fulfills the constraint
    if the value is within `tol` of zero.
    """

    type: Any

    def is_fulfilled(self, experiments: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
        return pd.Series(
            np.isclose(self(experiments), 0, atol=tol),
            index=experiments.index,
        )


class InequalityConstraint(IntrapointConstraint):
    """Abstract base class for constraints fulfilled at a non-positive constraint value.

    What is evaluated is defined by the `__call__` of the implementing subclass. This
    class only fixes how that value is interpreted: a candidate fulfills the constraint
    if the value does not exceed `tol`.
    """

    type: Any

    def is_fulfilled(self, experiments: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
        return self(experiments) <= 0 + tol


class ConstraintError(Exception):
    """Base Error for Constraints"""


class ConstraintNotFulfilledError(ConstraintError):
    """Raised when an constraint is not fulfilled."""
