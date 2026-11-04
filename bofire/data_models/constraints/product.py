from typing import Annotated, Any, List, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from bofire.data_models.constraints.constraint import (
    EqualityConstraint,
    InequalityConstraint,
    IntrapointConstraint,
)
from bofire.data_models.domain.features import Inputs
from bofire.data_models.features.api import ContinuousInput


class ProductConstraint(IntrapointConstraint):
    """Represents a product constraint of the form `sign * x1**e1 * x2**e2 * ... * xn**en`.

    Abstract base class for the equality and inequality variants.
    """

    type: Any
    exponents: Annotated[List[float], Field(min_length=2)] = Field(
        description="Exponents of the product, one per entry in `features` and in the "
        "same order.",
    )
    rhs: float = Field(description="Right-hand side of the constraint.")
    sign: Literal[1, -1] = Field(
        default=1,
        description="Sign of the left-hand side of the constraint.",
    )

    @model_validator(mode="after")
    def validate_list_lengths(self) -> "ProductConstraint":
        """Validates that the number of features and exponents provided are the same.

        Raises:
            ValueError: If the number of features and exponents are not equal.

        Returns:
            ProductConstraint: The current instance of the class.

        """
        if len(self.features) != len(self.exponents):
            raise ValueError(
                f"must provide same number of features and exponents, got {len(self.features)} != {len(self.exponents)}",
            )
        return self

    def validate_inputs(self, inputs: Inputs):
        keys = inputs.get_keys(ContinuousInput)
        for f in self.features:
            if f not in keys:
                raise ValueError(
                    f"Feature {f} is not a continuous input feature in the provided Inputs object.",
                )

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        """Evaluates the constraint on the given experiments.

        Args:
            experiments (pd.DataFrame): The experiments to evaluate the constraint on.

        Returns:
            pd.Series: The distance to reach constraint fulfillment.

        """
        return pd.Series(
            self.sign
            * np.prod(
                np.power(experiments[self.features].values, np.array(self.exponents)),
                axis=1,
            )
            - self.rhs,
            index=experiments.index,
        )

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "Jacobian for product constraints is not yet implemented.",
        )

    def hessian(self, experiments: pd.DataFrame) -> List[pd.DataFrame]:
        raise NotImplementedError(
            "Hessian for product constraints is not yet implemented.",
        )


class ProductEqualityConstraint(ProductConstraint, EqualityConstraint):
    """Represents a product constraint of the form `sign * x1**e1 * x2**e2 * ... * xn**en == rhs`.

    Use for exact multiplicative relationships between features. For a bound rather
    than an exact relationship, use `ProductInequalityConstraint`.

    Examples:
        >>> ProductEqualityConstraint(
        ...     features=["x1", "x2"], exponents=[2, 3], rhs=1.0, sign=1
        ... )
    """

    type: Literal["ProductEqualityConstraint"] = "ProductEqualityConstraint"

    def to_description(self) -> str:
        """Render as ``"x1^2 * x2^3 = 1.0"``.

        Example::

            >>> c = ProductEqualityConstraint(features=["x1", "x2"], exponents=[2, 3], rhs=1.0, sign=1)
            >>> c.to_description()
            'x1^2 * x2^3 = 1.0'
        """
        sign = "" if self.sign == 1 else "-"
        terms = " * ".join(f"{f}^{e}" for f, e in zip(self.features, self.exponents))
        desc = f"{sign}{terms} = {self.rhs}"
        if self.context:
            desc += f" — {self.context}"
        return desc


class ProductInequalityConstraint(ProductConstraint, InequalityConstraint):
    """Represents a product constraint of the form `sign * x1**e1 * x2**e2 * ... * xn**en <= rhs`.

    Use for multiplicative bounds between features. For an exact relationship, use
    `ProductEqualityConstraint`.

    Examples:
        >>> ProductInequalityConstraint(
        ...     features=["x1", "x2"], exponents=[2, 3], rhs=1.0, sign=1
        ... )
    """

    type: Literal["ProductInequalityConstraint"] = "ProductInequalityConstraint"

    def to_description(self) -> str:
        """Render as ``"x1^2 * x2^3 <= 1.0"``.

        Example::

            >>> c = ProductInequalityConstraint(features=["x1", "x2"], exponents=[2, 3], rhs=1.0, sign=1)
            >>> c.to_description()
            'x1^2 * x2^3 <= 1.0'
        """
        sign = "" if self.sign == 1 else "-"
        terms = " * ".join(f"{f}^{e}" for f, e in zip(self.features, self.exponents))
        desc = f"{sign}{terms} <= {self.rhs}"
        if self.context:
            desc += f" — {self.context}"
        return desc
