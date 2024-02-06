from typing import Annotated, List, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from bofire.data_models.constraints.constraint import (
    EqalityConstraint,
    InequalityConstraint,
    IntrapointConstraint,
)
from bofire.data_models.types import TFeatureKeys


class MultiLinearConstraint(IntrapointConstraint):
    """
    Represents a multi-linear constraint of the form `sign * x1**e1 * x2**e2 * ... * xn**en`.

    Attributes:
        type (str): The type of the constraint.
        features (FeatureKeys): The keys of the features used in the constraint.
        exponents (List[float]): The exponents corresponding to each feature.
        rhs (float): The right-hand side value of the constraint.
        sign (Literal[1, -1], optional): The sign of the left hand side of the constraint.
            Defaults to 1.
    """

    type: str
    features: TFeatureKeys
    exponents: Annotated[List[float], Field(min_length=2)]
    rhs: float
    sign: Literal[1, -1] = 1

    @model_validator(mode="after")
    def validate_list_lengths(self) -> "MultiLinearConstraint":
        """
        Validates that the number of features and exponents provided are the same.

        Raises:
            ValueError: If the number of features and exponents are not equal.

        Returns:
            MultiLinearConstraint: The current instance of the class.
        """
        if len(self.features) != len(self.exponents):
            raise ValueError(
                f"must provide same number of features and exponents, got {len(self.features)} != {len(self.exponents)}"
            )
        return self

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        """
        Evaluates the constraint on the given experiments.

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
            "Jacobian for multilinear constraints is not yet implemented."
        )


class MultiLinearEqualityConstraint(MultiLinearConstraint, EqalityConstraint):
    """
    Represents a multi-linear constraint of the form `sign * x1**e1 * x2**e2 * ... * xn**en == rhs`.

    Attributes:
        type (str): The type of the constraint.
        features (FeatureKeys): The keys of the features used in the constraint.
        exponents (List[float]): The exponents corresponding to each feature.
        rhs (float): The right-hand side value of the constraint.
        sign (Literal[1, -1], optional): The sign of the left hand side of the constraint.
            Defaults to 1.
    """

    type: Literal["MultiLinearEqualityConstraint"] = "MultiLinearEqualityConstraint"


class MultiLinearInequalityConstraint(MultiLinearConstraint, InequalityConstraint):
    """
    Represents a multi-linear constraint of the form `sign * x1**e1 * x2**e2 * ... * xn**en <= rhs`.

    Attributes:
        type (str): The type of the constraint.
        features (FeatureKeys): The keys of the features used in the constraint.
        exponents (List[float]): The exponents corresponding to each feature.
        rhs (float): The right-hand side value of the constraint.
        sign (Literal[1, -1], optional): The sign of the left hand side of the constraint.
            Defaults to 1.
    """

    type: Literal["MultiLinearInequalityConstraint"] = "MultiLinearInequalityConstraint"
