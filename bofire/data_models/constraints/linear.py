from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from pydantic import root_validator, validator

from bofire.data_models.constraints.constraint import (
    Coefficients,
    Constraint,
    FeatureKeys,
)


class LinearConstraint(Constraint):
    """Abstract base class for linear equality and inequality constraints.

    Attributes:
        features (list): list of feature keys (str) on which the constraint works on.
        coefficients (list): list of coefficients (float) of the constraint.
        rhs (float): Right-hand side of the constraint
    """

    type: Literal["LinearConstraint"] = "LinearConstraint"

    features: FeatureKeys
    coefficients: Coefficients
    rhs: float

    @validator("features")
    def validate_features_unique(cls, features):
        """Validate that feature keys are unique."""
        if len(features) != len(set(features)):
            raise ValueError("features must be unique")
        return features

    @root_validator(pre=False, skip_on_failure=True)
    def validate_list_lengths(cls, values):
        """Validate that length of the feature and coefficient lists have the same length."""
        if len(values["features"]) != len(values["coefficients"]):
            raise ValueError(
                f'must provide same number of features and coefficients, got {len(values["features"])} != {len(values["coefficients"])}'
            )
        return values

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        return (
            experiments[self.features] @ self.coefficients - self.rhs
        ) / np.linalg.norm(self.coefficients)

    def __str__(self) -> str:
        """Generate string representation of the constraint.

        Returns:
            str: string representation of the constraint.
        """
        return " + ".join(
            [f"{self.coefficients[i]} * {feat}" for i, feat in enumerate(self.features)]
        )

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile(
                [np.array(self.coefficients) / np.linalg.norm(self.coefficients)],
                [experiments.shape[0], 1],
            ),
            columns=[f"dg/d{name}" for name in self.features],
        )


class LinearEqualityConstraint(LinearConstraint):
    """Linear equality constraint of the form `coefficients * x = rhs`.

    Attributes:
        features (list): list of feature keys (str) on which the constraint works on.
        coefficients (list): list of coefficients (float) of the constraint.
        rhs (float): Right-hand side of the constraint
    """

    type: Literal["LinearEqualityConstraint"] = "LinearEqualityConstraint"

    # def is_fulfilled(self, experiments: pd.DataFrame, complete: bool) -> bool:
    #     """Check if the linear equality constraint is fulfilled for all the rows of the provided dataframe.

    #     Args:
    #         df_data (pd.DataFrame): Dataframe to evaluate constraint on.

    #     Returns:
    #         bool: True if fulfilled else False.
    #     """
    #     fulfilled = np.isclose(self(experiments), 0)
    #     if complete:
    #         return fulfilled.all()
    #     else:
    #         pd.Series(fulfilled, index=experiments.index)

    def is_fulfilled(self, experiments: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
        return pd.Series(
            np.isclose(self(experiments), 0, atol=tol), index=experiments.index
        )

    def __str__(self) -> str:
        """Generate string representation of the constraint.

        Returns:
            str: string representation of the constraint.
        """
        return super().__str__() + f" = {self.rhs}"


class LinearInequalityConstraint(LinearConstraint):
    """Linear inequality constraint of the form `coefficients * x <= rhs`.

    To instantiate a constraint of the form `coefficients * x >= rhs` multiply coefficients and rhs by -1, or
    use the classmethod `from_greater_equal`.

    Attributes:
        features (list): list of feature keys (str) on which the constraint works on.
        coefficients (list): list of coefficients (float) of the constraint.
        rhs (float): Right-hand side of the constraint
    """

    type: Literal["LinearInequalityConstraint"] = "LinearInequalityConstraint"

    def is_fulfilled(self, experiments: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
        return self(experiments) <= 0 + tol

    def as_smaller_equal(self) -> Tuple[List[str], List[float], float]:
        """Return attributes in the smaller equal convention

        Returns:
            Tuple[List[str], List[float], float]: features, coefficients, rhs
        """
        return self.features, self.coefficients, self.rhs

    def as_greater_equal(self) -> Tuple[List[str], List[float], float]:
        """Return attributes in the greater equal convention

        Returns:
            Tuple[List[str], List[float], float]: features, coefficients, rhs
        """
        return self.features, [-1.0 * c for c in self.coefficients], -1.0 * self.rhs

    @classmethod
    def from_greater_equal(
        cls,
        features: List[str],
        coefficients: List[float],
        rhs: float,
    ):
        """Class method to construct linear inequality constraint of the form `coefficients * x >= rhs`.

        Args:
            features (List[str]): List of feature keys.
            coefficients (List[float]): List of coefficients.
            rhs (float): Right-hand side of the constraint.
        """
        return cls(
            features=features,
            coefficients=[-1.0 * c for c in coefficients],
            rhs=-1.0 * rhs,
        )

    @classmethod
    def from_smaller_equal(
        cls,
        features: List[str],
        coefficients: List[float],
        rhs: float,
    ):
        """Class method to construct linear inequality constraint of the form `coefficients * x <= rhs`.

        Args:
            features (List[str]): List of feature keys.
            coefficients (List[float]): List of coefficients.
            rhs (float): Right-hand side of the constraint.
        """
        return cls(
            features=features,
            coefficients=coefficients,
            rhs=rhs,
        )

    def __str__(self):
        """Generate string representation of the constraint.

        Returns:
            str: string representation of the constraint.
        """
        return super().__str__() + f" <= {self.rhs}"
