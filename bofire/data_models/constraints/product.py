from typing import Annotated, List, Literal

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

    Attributes:
        type (str): The type of the constraint.
        features (FeatureKeys): The keys of the features used in the constraint.
        exponents (List[float]): The exponents corresponding to each feature.
        rhs (float): The right-hand side value of the constraint.
        sign (Literal[1, -1], optional): The sign of the left hand side of the constraint.
            Defaults to 1.

    """

    type: str
    exponents: Annotated[List[float], Field(min_length=2)]
    rhs: float
    sign: Literal[1, -1] = 1

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

    Attributes:
        type (str): The type of the constraint.
        features (FeatureKeys): The keys of the features used in the constraint.
        exponents (List[float]): The exponents corresponding to each feature.
        rhs (float): The right-hand side value of the constraint.
        sign (Literal[1, -1], optional): The sign of the left hand side of the constraint.
            Defaults to 1.

    """

    type: Literal["ProductEqualityConstraint"] = "ProductEqualityConstraint"


class ProductInequalityConstraint(ProductConstraint, InequalityConstraint):
    """Represents a product constraint of the form `sign * x1**e1 * x2**e2 * ... * xn**en <= rhs`.

    Attributes:
        type (str): The type of the constraint.
        features (FeatureKeys): The keys of the features used in the constraint.
        exponents (List[float]): The exponents corresponding to each feature.
        rhs (float): The right-hand side value of the constraint.
        sign (Literal[1, -1], optional): The sign of the left hand side of the constraint.
            Defaults to 1.

    """

    type: Literal["ProductInequalityConstraint"] = "ProductInequalityConstraint"
