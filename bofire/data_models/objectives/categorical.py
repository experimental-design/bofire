from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import model_validator

from bofire.data_models.objectives.objective import (
    ConstrainedObjective,
    Objective,
    TWeight,
)
from bofire.data_models.types import CategoryVals


class ConstrainedCategoricalObjective(ConstrainedObjective, Objective):
    """Compute the categorical objective value as:

        Po where P is an [n, c] matrix where each row is a probability vector
        (P[i, :].sum()=1 for all i) and o is a vector of size [c] of objective values

    Attributes:
        w (float): float between zero and one for weighting the objective.
        desirability (list): list of values of size c (c is number of categories) such that the i-th entry is in {True, False}

    """

    w: TWeight = 1.0
    categories: CategoryVals
    desirability: List[bool]
    type: Literal["ConstrainedCategoricalObjective"] = "ConstrainedCategoricalObjective"

    @model_validator(mode="after")
    def validate_desireability(self):
        """Validates that categories have unique names

        Args:
            categories (List[str]): List or tuple of category names

        Raises:
            ValueError: when categories do not match objective categories

        Returns:
            Tuple[str]: Tuple of the categories

        """
        if len(self.desirability) != len(self.categories):
            raise ValueError(
                "number of categories differs from number of desirabilities",
            )
        return self

    def to_dict(self) -> Dict:
        """Returns the categories and corresponding objective values as dictionary"""
        return dict(zip(self.categories, self.desirability))

    def to_dict_label(self) -> Dict:
        """Returns the categories and label location of categories"""
        return {c: i for i, c in enumerate(self.categories)}

    def from_dict_label(self) -> Dict:
        """Returns the label location and the categories"""
        d = self.to_dict_label()
        return dict(zip(d.values(), d.keys()))

    def __call__(
        self,
        x: Union[pd.Series, np.ndarray],
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray, float]:
        """The call function returning a probabilistic reward for x.

        Args:
            x (np.ndarray): A matrix of x values
            x_adapt (Optional[np.ndarray], optional): An array of x values which are used to
                update the objective parameters on the fly. Defaults to None.

        Returns:
            np.ndarray: A reward calculated as inner product of probabilities and feasible objectives.

        """
        return np.dot(x, np.array(self.desirability))
