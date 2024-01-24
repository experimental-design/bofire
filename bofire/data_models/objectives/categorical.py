from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd
from pydantic import field_validator

from bofire.data_models.features.feature import TCategoryVals
from bofire.data_models.objectives.objective import (
    ConstrainedObjective,
    Objective,
    TWeight,
)


class CategoricalObjective:
    """Abstract categorical objective class"""


class ConstrainedCategoricalObjective(
    ConstrainedObjective, CategoricalObjective, Objective
):
    """Compute the categorical objective value as:

        Po where P is an [n, c] matrix where each row is a probability vector
        (P[i, :].sum()=1 for all i) and o is a vector of size [c] of objective values

    Attributes:
        w (float): float between zero and one for weighting the objective.
        desirability (list): list of values of size c (c is number of categories) such that the i-th entry is in {True, False}
    """

    w: TWeight = 1.0
    categories: TCategoryVals
    desirability: List[bool]
    eta: float = 1.0
    type: Literal["ConstrainedCategoricalObjective"] = "ConstrainedCategoricalObjective"

    @field_validator(
        "desirability",
    )
    def validate_categories_unique(cls, desirability: List[bool], info) -> List[bool]:
        """validates that desirabilities match the categories

        Args:
            categories (List[str]): List or tuple of category names

        Raises:
            ValueError: when desirability count is not equal to category count

        Returns:
            Tuple[bool]: Tuple of the desirability
        """
        if len(desirability) != len(info.data["categories"]):
            raise ValueError(
                "number of categories differs from number of desirabilities"
            )
        return desirability

    def to_dict(self) -> Dict:
        """Returns the categories and corresponding objective values as dictionary"""
        return dict(zip(self.categories, self.desirability))

    def to_dict_label(self) -> Dict:
        """Returns the catergories and label location of categories"""
        return {c: i for i, c in enumerate(self.categories)}

    def from_dict_label(self) -> Dict:
        """Returns the label location and the categories"""
        d = self.to_dict_label()
        return dict(zip(d.values(), d.keys()))

    def __call__(
        self, x: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray, float]:
        """The call function returning a probabilistic reward for x.

        Args:
            x (np.ndarray): A matrix of x values

        Returns:
            np.ndarray: A reward calculated as inner product of probabilities and feasible objectives.
        """
        return np.dot(x, np.array(self.desirability))
