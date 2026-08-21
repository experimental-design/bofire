from typing import Annotated, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from bofire.data_models.constraints.constraint import (
    EqualityConstraint,
    InequalityConstraint,
    IntrapointConstraint,
)
from bofire.data_models.domain.features import Inputs
from bofire.data_models.features.api import ContinuousInput, DiscreteInput


class LinearConstraint(IntrapointConstraint):
    """Abstract base class for linear equality and inequality constraints."""

    type: Literal["LinearConstraint"] = "LinearConstraint"

    coefficients: Annotated[List[float], Field(min_length=2)] = Field(
        description="Coefficients of the linear combination, one per entry in "
        "`features` and in the same order.",
    )
    rhs: float = Field(description="Right-hand side of the constraint.")

    @model_validator(mode="after")
    def validate_list_lengths(self):
        """Validate that length of the feature and coefficient lists have the same length."""
        if len(self.features) != len(self.coefficients):
            raise ValueError(
                f"must provide same number of features and coefficients, got {len(self.features)} != {len(self.coefficients)}",
            )
        return self

    def validate_inputs(self, inputs: Inputs):
        keys = inputs.get_keys([ContinuousInput, DiscreteInput])
        for f in self.features:
            if f not in keys:
                raise ValueError(
                    f"Feature {f} is not a continuous input feature in the provided Inputs object.",
                )

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        return (
            experiments[self.features] @ self.coefficients - self.rhs
        ) / np.linalg.norm(np.array(self.coefficients))

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile(
                [
                    np.array(self.coefficients)
                    / np.linalg.norm(np.array(self.coefficients)),
                ],
                [experiments.shape[0], 1],
            ),
            columns=[f"dg/d{name}" for name in self.features],
        )

    def hessian(self, experiments: pd.DataFrame) -> Dict[Union[int, str], float]:
        return dict.fromkeys(range(experiments.shape[0]), 0.0)


class LinearEqualityConstraint(LinearConstraint, EqualityConstraint):
    """Linear equality constraint of the form `coefficients * x = rhs`.

    Use for exact linear relationships, such as a mixture whose components must sum to
    one. For a bound rather than an exact relationship, use
    `LinearInequalityConstraint`.

    Examples:
        >>> LinearEqualityConstraint(
        ...     features=["x1", "x2"], coefficients=[1.0, 1.0], rhs=1.0
        ... )
    """

    type: Literal["LinearEqualityConstraint"] = "LinearEqualityConstraint"

    def to_description(self) -> str:
        """Render as ``"1.0*x1 + 2.0*x2 = 5.0"``.

        Example::

            >>> c = LinearEqualityConstraint(features=["x1", "x2"], coefficients=[1.0, 2.0], rhs=5.0)
            >>> c.to_description()
            '1.0*x1 + 2.0*x2 = 5.0'
        """
        terms = " + ".join(f"{c}*{f}" for c, f in zip(self.coefficients, self.features))
        desc = f"{terms} = {self.rhs}"
        if self.context:
            desc += f" — {self.context}"
        return desc


class LinearInequalityConstraint(LinearConstraint, InequalityConstraint):
    """Linear inequality constraint of the form `coefficients * x <= rhs`.

    To instantiate a constraint of the form `coefficients * x >= rhs` multiply coefficients and rhs by -1, or
    use the classmethod `from_greater_equal`.

    Use for linear bounds, such as a budget that must not be overspent. For an exact
    relationship, use `LinearEqualityConstraint`.

    Examples:
        >>> LinearInequalityConstraint(
        ...     features=["x1", "x2"], coefficients=[1.0, 2.0], rhs=5.0
        ... )
    """

    type: Literal["LinearInequalityConstraint"] = "LinearInequalityConstraint"

    def to_description(self) -> str:
        """Render as ``"1.0*x1 + 2.0*x2 <= 5.0"``.

        Example::

            >>> c = LinearInequalityConstraint(features=["x1", "x2"], coefficients=[1.0, 2.0], rhs=5.0)
            >>> c.to_description()
            '1.0*x1 + 2.0*x2 <= 5.0'
        """
        terms = " + ".join(f"{c}*{f}" for c, f in zip(self.coefficients, self.features))
        desc = f"{terms} <= {self.rhs}"
        if self.context:
            desc += f" — {self.context}"
        return desc

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
